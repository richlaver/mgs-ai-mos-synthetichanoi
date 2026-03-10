from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
import tomllib

import pymysql


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


def read_target_db_config(root: Path) -> dict[str, object]:
    with (root / ".streamlit" / "secrets.toml").open("rb") as file:
        project_data = tomllib.load(file)["project_data"]
    return dict(project_data["hanoi_synthetic"])


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


def move_rows_to_mydata(connection: pymysql.connections.Connection, updated_to_time: datetime) -> dict[str, int]:
    cutoff_timestamp = updated_to_time.strftime("%Y-%m-%d %H:%M:%S")

    with connection.cursor() as cursor:
        cursor.execute("CREATE TABLE IF NOT EXISTS `futuredata` LIKE `mydata`")

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
            }

        insert_sql = f"""
            INSERT INTO `mydata` (`instr_id`, `date1`, `data1`, `custom_fields`)
            SELECT f.`instr_id`, f.`date1`, f.`data1`, f.`custom_fields`
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
            FROM `futuredata` f
            LEFT JOIN `mydata` m
              ON {ROW_MATCH_CONDITION}
            WHERE f.`date1` <= %s
              AND m.`instr_id` IS NULL
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
                f"rows_missing_after_copy={results['rows_missing_after_copy']}."
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