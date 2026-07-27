from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import os
import sys
import tomllib

import pymysql


ENV_VAR_NAMES = {
    "db_host": "HANOI_SYNTHETIC_DB_HOST",
    "db_user": "HANOI_SYNTHETIC_DB_USER",
    "db_pass": "HANOI_SYNTHETIC_DB_PASS",
    "db_name": "HANOI_SYNTHETIC_DB_NAME",
    "port": "HANOI_SYNTHETIC_DB_PORT",
}


ROW_MATCH_CONDITION = " AND ".join(
    [
        "m.`instr_id` = f.`instr_id`",
        "m.`date1` = f.`date1`",
        "m.`data1` = f.`data1`",
        "COALESCE(m.`custom_fields`, '') = COALESCE(f.`custom_fields`, '')",
    ]
)


def append_stream_log(root: Path, message: str) -> None:
    log_dir = root / "validation_data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "mydata_stream.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"{line}\n")

    print(line, flush=True)


def read_target_db_config_from_env() -> dict[str, object] | None:
    values = {key: os.getenv(env_var, "").strip() for key, env_var in ENV_VAR_NAMES.items()}
    required_keys = ("db_host", "db_user", "db_pass", "db_name")
    present_required_keys = [key for key in required_keys if values[key]]

    if not present_required_keys:
        return None

    missing_required_env_vars = [ENV_VAR_NAMES[key] for key in required_keys if not values[key]]
    if missing_required_env_vars:
        raise RuntimeError(
            "Incomplete database configuration in environment variables. Missing: "
            + ", ".join(missing_required_env_vars)
        )

    config: dict[str, object] = {
        "db_host": values["db_host"],
        "db_user": values["db_user"],
        "db_pass": values["db_pass"],
        "db_name": values["db_name"],
    }
    if values["port"]:
        config["port"] = int(values["port"])
    return config


def read_target_db_config_from_secrets(root: Path) -> dict[str, object]:
    with (root / ".streamlit" / "secrets.toml").open("rb") as file:
        project_data = tomllib.load(file)["project_data"]
    return dict(project_data["hanoi_synthetic"])


def read_target_db_config(root: Path) -> dict[str, object]:
    env_config = read_target_db_config_from_env()
    if env_config is not None:
        return env_config
    return read_target_db_config_from_secrets(root)


def connect_mysql(config: dict[str, object]) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=str(config["db_host"]),
        user=str(config["db_user"]),
        password=str(config["db_pass"]),
        database=str(config["db_name"]),
        port=int(config.get("port", 3306)),
        charset="utf8mb4",
        autocommit=False,
        cursorclass=pymysql.cursors.Cursor,
    )


def fetch_single_int(cursor: pymysql.cursors.Cursor, query: str, params: tuple[object, ...] = ()) -> int:
    cursor.execute(query, params)
    row = cursor.fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def resolve_row_id_column(cursor: pymysql.cursors.Cursor, table_name: str) -> str | None:
    cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
    columns = list(cursor.fetchall() or [])
    if not columns:
        return None

    for column in columns:
        field_name = str(column[0])
        key = str(column[3]) if len(column) > 3 else ""
        extra = str(column[5]).lower() if len(column) > 5 else ""
        if key == "PRI" or "auto_increment" in extra:
            return field_name

    field_names = {str(column[0]) for column in columns}
    if "id" in field_names:
        return "id"
    return None


def dedupe_timeseries_table(cursor: pymysql.cursors.Cursor, table_name: str) -> int:
    """Delete exact duplicate business-key rows, keeping the lowest id."""
    id_column = resolve_row_id_column(cursor, table_name)
    if id_column is None:
        return 0

    dedupe_sql = f"""
        DELETE t1 FROM `{table_name}` t1
        INNER JOIN `{table_name}` t2
          ON t1.`instr_id` = t2.`instr_id`
         AND t1.`date1` = t2.`date1`
         AND t1.`data1` = t2.`data1`
         AND COALESCE(t1.`custom_fields`, '') = COALESCE(t2.`custom_fields`, '')
         AND t1.`{id_column}` > t2.`{id_column}`
    """
    cursor.execute(dedupe_sql)
    return int(cursor.rowcount or 0)


def move_rows_to_mydata(connection: pymysql.connections.Connection, updated_to_time: datetime) -> dict[str, int]:
    cutoff_timestamp = updated_to_time.strftime("%Y-%m-%d %H:%M:%S")

    with connection.cursor() as cursor:
        cursor.execute("CREATE TABLE IF NOT EXISTS `futuredata` LIKE `mydata`")

        futuredata_duplicates_removed = dedupe_timeseries_table(cursor, "futuredata")
        mydata_duplicates_removed = dedupe_timeseries_table(cursor, "mydata")

        rows_to_update = fetch_single_int(
            cursor,
            "SELECT COUNT(*) FROM `futuredata` WHERE `date1` <= %s",
            (cutoff_timestamp,),
        )
        if rows_to_update == 0:
            connection.commit()
            return {
                "rows_to_update": 0,
                "rows_inserted": 0,
                "rows_deleted": 0,
                "rows_missing_after_copy": 0,
                "futuredata_duplicates_removed": futuredata_duplicates_removed,
                "mydata_duplicates_removed": mydata_duplicates_removed,
            }

        insert_sql = f"""
            INSERT INTO `mydata` (`instr_id`, `date1`, `data1`, `custom_fields`)
            SELECT DISTINCT f.`instr_id`, f.`date1`, f.`data1`, f.`custom_fields`
            FROM `futuredata` f
            LEFT JOIN `mydata` m
              ON {ROW_MATCH_CONDITION}
            WHERE f.`date1` <= %s
              AND m.`instr_id` IS NULL
        """
        cursor.execute(insert_sql, (cutoff_timestamp,))
        rows_inserted = int(cursor.rowcount or 0)

        missing_after_copy_sql = f"""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT f.`instr_id`, f.`date1`, f.`data1`, f.`custom_fields`
                FROM `futuredata` f
                WHERE f.`date1` <= %s
            ) f
            LEFT JOIN `mydata` m
              ON m.`instr_id` = f.`instr_id`
             AND m.`date1` = f.`date1`
             AND m.`data1` = f.`data1`
             AND COALESCE(m.`custom_fields`, '') = COALESCE(f.`custom_fields`, '')
            WHERE m.`instr_id` IS NULL
        """
        rows_missing_after_copy = fetch_single_int(cursor, missing_after_copy_sql, (cutoff_timestamp,))
        if rows_missing_after_copy != 0:
            raise RuntimeError(
                "Copy verification failed: "
                f"{rows_missing_after_copy} rows from futuredata are still missing in mydata."
            )

        cursor.execute("DELETE FROM `futuredata` WHERE `date1` <= %s", (cutoff_timestamp,))
        rows_deleted = int(cursor.rowcount or 0)
        if rows_deleted != rows_to_update:
            raise RuntimeError(
                "Delete verification failed: "
                f"expected to delete {rows_to_update} rows from futuredata but deleted {rows_deleted}."
            )

    connection.commit()
    return {
        "rows_to_update": rows_to_update,
        "rows_inserted": rows_inserted,
        "rows_deleted": rows_deleted,
        "rows_missing_after_copy": 0,
        "futuredata_duplicates_removed": futuredata_duplicates_removed,
        "mydata_duplicates_removed": mydata_duplicates_removed,
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    current_timestamp = datetime.now()
    updated_to_time = current_timestamp - timedelta(hours=24)
    connection: pymysql.connections.Connection | None = None

    append_stream_log(
        root,
        (
            "Scheduled target database update started. "
            f"current_timestamp={current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"updated_to_time={updated_to_time.strftime('%Y-%m-%d %H:%M:%S')}."
        ),
    )

    try:
        target_config = read_target_db_config(root)
        connection = connect_mysql(target_config)
        results = move_rows_to_mydata(connection, updated_to_time)
        append_stream_log(
            root,
            (
                "Scheduled target database update completed successfully. "
                f"rows_to_update={results['rows_to_update']}, "
                f"rows_inserted={results['rows_inserted']}, "
                f"rows_deleted={results['rows_deleted']}, "
                f"rows_missing_after_copy={results['rows_missing_after_copy']}, "
                f"futuredata_duplicates_removed={results['futuredata_duplicates_removed']}, "
                f"mydata_duplicates_removed={results['mydata_duplicates_removed']}."
            ),
        )
        return 0
    except Exception as error:
        if connection is not None:
            connection.rollback()
        append_stream_log(root, f"Scheduled target database update failed: {error}")
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    sys.exit(main())