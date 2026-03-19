from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
import calendar
import random
import threading
import tomllib
import json
import re
import math
from typing import Any, Callable

import pymysql
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(layout="wide")


@dataclass
class Instrument:
    instr_id: str
    db_type: str
    event_type: str
    easting: float
    northing: float
    latitude: float
    longitude: float
    timeseries: list[dict[str, Any]]
    children: list[tuple[str, float]] | None


@dataclass
class Location:
    easting: float
    northing: float
    latitude: float
    longitude: float


@dataclass
class Event:
    start_time: datetime
    epicentre: Location
    type: str
    magnitude_mm: float
    duration_days: float
    radius_m: float
    direction_deg: float | None


EVENT_TYPES = ["surface_settlement", "groundwater_level", "subsurface_lateral_displacement"]
EVENT_PROBABILITIES = {
    "surface_settlement": 0.3,
    "groundwater_level": 0.12,
    "subsurface_lateral_displacement": 0.1,
}
EVENT_TYPE_TO_DB_TYPES = {
    "surface_settlement": {"LP", "TLP"},
    "groundwater_level": {"CASA", "PZ"},
    "subsurface_lateral_displacement": {"INCL", "EI"},
}
EVENT_TYPE_BY_DB_TYPE = {
    "LP": "surface_settlement",
    "TLP": "surface_settlement",
    "CASA": "groundwater_level",
    "PZ": "groundwater_level",
    "INCL": "subsurface_lateral_displacement",
    "EI": "subsurface_lateral_displacement",
}
DB_TYPE_COLORS = {
    "CASA": "#4c6ef5",
    "PZ": "#15aabf",
    "LP": "#fa5252",
    "TLP": "#f76707",
    "INCL": "#40c057",
    "EI": "#82c91e",
}
TYPE_COLORS = {
    "surface_settlement": "#ff6b6b",
    "groundwater_level": "#4dabf7",
    "subsurface_lateral_displacement": "#51cf66",
}

RAW_INSTR_TYPE_FILTERS = [
    ("LP", "MOVEMENT"),
    ("TLP", "MOVEMENT"),
    ("CASA", "DEFAULT"),
    ("PZ", "DEFAULT"),
    ("INCL", "CHILD"),
    ("INCL", "MASTER"),
    ("EI", "CHILD"),
    ("EI", "MASTER"),
]

EVENT_DISTRIBUTION_PARAMS = {
    "surface_settlement": {
        "magnitude_negative_prob": 0.9,
        "magnitude_negative": {"alpha": 1.0, "beta": 2.0, "maximum": 20.0},
        "magnitude_positive": {"alpha": 1.0, "beta": 2.0, "maximum": 5.0},
        "duration_days": {"alpha": 0.5, "beta": 1.0, "maximum": 7.0},
        "radius_m": {"alpha": 2.0, "beta": 3.5, "maximum": 150.0},
    },
    "groundwater_level": {
        "magnitude_positive": {"alpha": 1.0, "beta": 2.0, "maximum": 2500.0},
        "duration_days": {"alpha": 2.0, "beta": 5.0, "maximum": 20.0},
        "radius_m": {"alpha": 2.0, "beta": 3.5, "maximum": 400.0},
    },
    "subsurface_lateral_displacement": {
        "magnitude_positive": {"alpha": 1.0, "beta": 2.0, "maximum": 70.0},
        "duration_days": {"alpha": 2.0, "beta": 5.0, "maximum": 10.0},
        "radius_m": {"alpha": 2.0, "beta": 3.5, "maximum": 200.0},
        "direction_deg": {"alpha": 1.0, "beta": 1.0, "maximum": 360.0},
    },
}

NOISE_EVENT_PEAK_PARAMS = {
    "surface_settlement": {"mean": 0.0, "stddev": 0.5},
    "groundwater_level": {"mean": 0.0, "stddev": 100.0},
    "subsurface_lateral_displacement": {"mean": 0.0, "stddev": 1.0},
}

NOISE_EVENT_TIME_SPREAD_PARAMS = {
    "surface_settlement": {"median_days": 1.5, "shape": 0.2},
    "groundwater_level": {"median_days": 2.5, "shape": 0.2},
    "subsurface_lateral_displacement": {"median_days": 1.5, "shape": 0.2},
}

NOISE_EVENT_DEPTH_SPREAD_PARAMS = {"median_m": 2.0, "shape": 0.1}
NOISE_EVENT_TIME_WINDOW_DAYS = 2

# Clear child tables before parent tables to satisfy FK delete constraints.
SUPPORTING_TABLES_CLEAR_ORDER = [
    "review_instruments_values",
    "aaa_color_info",
    "review_instruments",
    "hierarchy_members",
    "instr_cal_calibs",
    "hierarchies",
    "instrum",
    "raw_instr_typestbl",
    "location",
]

# Copy parent tables before child tables to satisfy FK insert constraints.
SUPPORTING_TABLES_COPY_ORDER = [
    "location",
    "raw_instr_typestbl",
    "instrum",
    "hierarchies",
    "hierarchy_members",
    "review_instruments",
    "review_instruments_values",
    "aaa_color_info",
    "instr_cal_calibs",
]

DB_WRITE_CHUNK_SIZE = 5000
SYNTHETIC_DATA_TABLES = ("mydata", "futuredata")
TIME_SERIES_START_MODE_ZERO = "Zero"
TIME_SERIES_START_MODE_SNAPSHOT = "Snapshot of existing synthetic data"


def create_db_write_runtime() -> dict[str, Any]:
    return {
        "thread": None,
        "cancel_event": threading.Event(),
        "lock": threading.Lock(),
        "source_connection": None,
        "target_connection": None,
        "state": {
            "status": "idle",
            "async_status": "idle",
            "rows_written": 0,
            "total_rows": 0,
            "percent_complete": 0.0,
            "message": "",
            "started": False,
            "last_update": "",
            "thread_alive": False,
            "thread_completed_at": "",
            "thread_terminal_message": "",
            "logs": [],
        },
    }


def get_db_write_runtime() -> dict[str, Any]:
    runtime = st.session_state.get("_db_write_runtime")
    if not isinstance(runtime, dict):
        runtime = create_db_write_runtime()
        st.session_state["_db_write_runtime"] = runtime
    return runtime


def update_db_write_state(*, runtime: dict[str, Any] | None = None, **updates: Any) -> None:
    runtime = runtime or get_db_write_runtime()
    lock = runtime["lock"]
    with lock:
        runtime["state"].update(updates)
        runtime["state"]["last_update"] = datetime.now().isoformat()


def get_db_write_state() -> dict[str, Any]:
    runtime = get_db_write_runtime()
    lock = runtime["lock"]
    with lock:
        return dict(runtime["state"])


def runtime_thread_is_alive(runtime: dict[str, Any]) -> bool:
    thread = runtime.get("thread")
    return isinstance(thread, threading.Thread) and thread.is_alive()


def append_stream_log(root: Path, message: str, runtime: dict[str, Any] | None = None) -> None:
    log_dir = root / "validation_data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "mydata_stream.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"{line}\n")

    # Mirror logs to terminal so operators can observe progress outside Streamlit.
    print(line, flush=True)

    runtime = runtime or get_db_write_runtime()
    lock = runtime["lock"]
    with lock:
        log_entries = list(runtime["state"].get("logs", []))
        log_entries.append(line)
        runtime["state"]["logs"] = log_entries[-300:]
        runtime["state"]["last_update"] = datetime.now().isoformat()


def create_form_log_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "message": "",
        "last_update": "",
        "completed_units": 0,
        "total_units": 0,
        "percent_complete": 0.0,
        "logs": [],
    }


def get_form_log_state(session_key: str) -> dict[str, Any]:
    state = st.session_state.get(session_key)
    if not isinstance(state, dict):
        state = create_form_log_state()
        st.session_state[session_key] = state
    return state


def update_form_log_state(session_key: str, **updates: Any) -> None:
    state = get_form_log_state(session_key)
    state.update(updates)
    state["last_update"] = datetime.now().isoformat()


def get_form_log_state_snapshot(session_key: str) -> dict[str, Any]:
    return dict(get_form_log_state(session_key))


def append_form_log(root: Path, log_filename: str, session_key: str, message: str) -> None:
    log_dir = root / "validation_data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_filename
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"{line}\n")

    print(line, flush=True)

    state = get_form_log_state(session_key)
    log_entries = list(state.get("logs", []))
    log_entries.append(line)
    state["logs"] = log_entries[-300:]
    state["last_update"] = datetime.now().isoformat()


def set_form_progress(
    session_key: str,
    completed_units: int,
    total_units: int,
    *,
    message: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    safe_total_units = max(0, int(total_units))
    safe_completed_units = max(0, min(int(completed_units), safe_total_units if safe_total_units else int(completed_units)))
    percent_complete = 100.0 if safe_total_units == 0 else (safe_completed_units / safe_total_units * 100.0)

    updates: dict[str, Any] = {
        "completed_units": safe_completed_units,
        "total_units": safe_total_units,
        "percent_complete": percent_complete,
    }
    if message is not None:
        updates["message"] = message
    if status is not None:
        updates["status"] = status

    update_form_log_state(session_key, **updates)
    return get_form_log_state_snapshot(session_key)


def render_form_progress_widgets(progress_bar, progress_caption, current_state: dict[str, Any]) -> None:
    percent_complete = float(current_state.get("percent_complete", 0.0) or 0.0)
    completed_units = int(current_state.get("completed_units", 0) or 0)
    total_units = int(current_state.get("total_units", 0) or 0)
    progress_value = max(0.0, min(1.0, percent_complete / 100.0))
    progress_bar.progress(progress_value)

    if total_units > 0:
        progress_caption.caption(f"Progress: {completed_units} / {total_units} ({percent_complete:.2f}%)")
    elif current_state.get("message"):
        progress_caption.caption(str(current_state.get("message")))
    else:
        progress_caption.caption("Progress will appear when the process starts.")


def form_status_badge_color(status: str) -> str:
    return {
        "idle": "gray",
        "running": "blue",
        "completed": "green",
        "error": "red",
    }.get(status, "gray")


def render_form_status_logs(title: str, current_state: dict[str, Any]) -> None:
    state = str(current_state.get("status", "idle"))
    status_state = "running"
    if state == "completed":
        status_state = "complete"
    elif state == "error":
        status_state = "error"

    st.caption(title)
    st.badge(state.replace("_", " ").title(), color=form_status_badge_color(state))
    if current_state.get("message"):
        st.caption(str(current_state.get("message")))

    logs = current_state.get("logs", [])
    if not isinstance(logs, list):
        logs = []

    latest_logs = [str(entry) for entry in logs[-7:]]
    with st.status(title, state=status_state, expanded=True):
        if not latest_logs:
            st.write("No log events yet.")
        else:
            for line in latest_logs:
                st.write(line)


def request_cancel(root: Path, runtime: dict[str, Any]) -> None:
    runtime["cancel_event"].set()
    append_stream_log(root, "Cancellation requested by user.", runtime=runtime)
    update_db_write_state(runtime=runtime, status="cancelling", async_status="cancelling", message="Cancellation requested.")

    # Force-close live DB connections to interrupt in-flight SQL statements.
    for connection_key in ("source_connection", "target_connection"):
        connection = runtime.get(connection_key)
        if connection is None:
            continue
        try:
            connection.close()
            append_stream_log(root, f"Force-closed {connection_key} for cancellation.", runtime=runtime)
        except Exception as error:
            append_stream_log(root, f"Failed to force-close {connection_key}: {error}", runtime=runtime)
        finally:
            runtime[connection_key] = None


def read_db_configs(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with (root / ".streamlit" / "secrets.toml").open("rb") as file:
        project_data = tomllib.load(file)["project_data"]
    source_config = project_data["hanoi_live"]
    target_config = project_data["hanoi_synthetic"]
    return source_config, target_config


def connect_mysql(config: dict[str, Any], cursorclass: type[pymysql.cursors.Cursor]) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=config["db_host"],
        user=config["db_user"],
        password=config["db_pass"],
        database=config["db_name"],
        port=int(config.get("port", 3306)),
        charset="utf8mb4",
        autocommit=False,
        cursorclass=cursorclass,
    )


def table_exists(cursor, table_name: str) -> bool:
    cursor.execute("SHOW TABLES LIKE %s", (table_name,))
    return cursor.fetchone() is not None


def get_existing_synthetic_data_tables(cursor) -> list[str]:
    return [table_name for table_name in SYNTHETIC_DATA_TABLES if table_exists(cursor, table_name)]


def build_synthetic_data_union_query(table_names: list[str]) -> str:
    return " UNION ALL ".join(
        f"SELECT `instr_id`, `date1`, `data1`, `custom_fields` FROM `{table_name}`"
        for table_name in table_names
    )


def fetch_synthetic_data_date_bounds(root: Path) -> tuple[date | None, date | None, str | None]:
    try:
        _, target_config = read_db_configs(root)
        with connect_mysql(target_config, pymysql.cursors.DictCursor) as connection:
            with connection.cursor() as cursor:
                table_names = get_existing_synthetic_data_tables(cursor)
                if not table_names:
                    return None, None, None

                union_query = build_synthetic_data_union_query(table_names)
                cursor.execute(
                    f"SELECT MIN(`date1`) AS min_date1, MAX(`date1`) AS max_date1 FROM ({union_query}) synthetic_data"
                )
                row = cursor.fetchone() or {}

        min_date1 = parse_iso_datetime(row.get("min_date1"))
        max_date1 = parse_iso_datetime(row.get("max_date1"))
        if min_date1 is None or max_date1 is None:
            return None, None, None
        return min_date1.date(), max_date1.date(), None
    except Exception as error:
        return None, None, str(error)


def parse_custom_fields_payload(raw_custom_fields: Any) -> dict[str, Any]:
    if isinstance(raw_custom_fields, dict):
        return raw_custom_fields
    if raw_custom_fields is None or raw_custom_fields == "":
        return {}
    try:
        parsed = json.loads(str(raw_custom_fields))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def snapshot_value_from_database_row(db_type: str, row: dict[str, Any]) -> float | None:
    data1_value = coerce_float(row.get("data1"))
    custom_fields = parse_custom_fields_payload(row.get("custom_fields"))

    if db_type in {"LP", "TLP"}:
        return data1_value
    if db_type in {"CASA", "PZ"}:
        if data1_value is not None:
            return -1000.0 * data1_value
        calculation1 = coerce_float(custom_fields.get("calculation1"))
        return None if calculation1 is None else 1000.0 * calculation1
    if db_type == "INCL":
        return coerce_float(custom_fields.get("calculation3"))
    if db_type == "EI":
        return coerce_float(custom_fields.get("calculation2"))
    return None


def fetch_snapshot_offsets(root: Path, instruments: list[Instrument], snapshot_date: date) -> dict[str, float]:
    instrument_type_by_id = {
        instrument.instr_id: instrument.db_type
        for instrument in instruments
    }
    for instrument in instruments:
        for child_id, _ in instrument.children or []:
            instrument_type_by_id[child_id] = instrument.db_type

    instrument_ids = sorted(instrument_type_by_id)
    if not instrument_ids:
        return {}

    _, target_config = read_db_configs(root)
    snapshot_cutoff = datetime.combine(snapshot_date, time.max).strftime("%Y-%m-%d %H:%M:%S")
    with connect_mysql(target_config, pymysql.cursors.DictCursor) as connection:
        with connection.cursor() as cursor:
            table_names = get_existing_synthetic_data_tables(cursor)
            if not table_names:
                return {}

            union_query = build_synthetic_data_union_query(table_names)
            latest_instr_clause, instr_params = build_in_clause("latest_source.instr_id", instrument_ids)
            combined_instr_clause, combined_instr_params = build_in_clause("combined.instr_id", instrument_ids)
            query = f"""
            SELECT combined.instr_id, combined.date1, combined.data1, combined.custom_fields
            FROM ({union_query}) combined
            JOIN (
                SELECT instr_id, MAX(date1) AS max_date1
                FROM ({union_query}) latest_source
                WHERE {latest_instr_clause} AND date1 <= %s
                GROUP BY instr_id
            ) latest
              ON combined.instr_id = latest.instr_id
             AND combined.date1 = latest.max_date1
            WHERE {combined_instr_clause} AND combined.date1 <= %s
            ORDER BY combined.instr_id, combined.date1 DESC
            """
            params = (*instr_params, snapshot_cutoff, *combined_instr_params, snapshot_cutoff)
            cursor.execute(query, params)
            rows = cursor.fetchall() or []

    offsets: dict[str, float] = {}
    for row in rows:
        instr_id = str(row.get("instr_id", ""))
        if instr_id in offsets:
            continue
        db_type = instrument_type_by_id.get(instr_id, "")
        offset = snapshot_value_from_database_row(db_type, row)
        if offset is None:
            continue
        offsets[instr_id] = float(offset)
    return offsets


def delete_target_rows_from_date(
    target_cursor,
    target_connection,
    table_name: str,
    start_date: date,
    root: Path,
    runtime: dict[str, Any] | None = None,
) -> None:
    start_timestamp = datetime.combine(start_date, time.min).strftime("%Y-%m-%d %H:%M:%S")
    target_cursor.execute(f"DELETE FROM `{table_name}` WHERE `date1` >= %s", (start_timestamp,))
    deleted_rows = int(getattr(target_cursor, "rowcount", 0) or 0)
    target_connection.commit()
    append_stream_log(
        root,
        (
            f"Deleted {deleted_rows} rows from {table_name} with date1 >= "
            f"{start_date.isoformat()} while preserving earlier rows in place."
        ),
        runtime=runtime,
    )


def selected_start_mode() -> str:
    mode = str(st.session_state.get("time_series_start_mode", TIME_SERIES_START_MODE_ZERO))
    if mode not in {TIME_SERIES_START_MODE_ZERO, TIME_SERIES_START_MODE_SNAPSHOT}:
        return TIME_SERIES_START_MODE_ZERO
    return mode


def selected_start_date(default_start_date: date) -> date:
    if selected_start_mode() == TIME_SERIES_START_MODE_SNAPSHOT:
        snapshot_date = st.session_state.get("snapshot_start_date")
        if isinstance(snapshot_date, date):
            return snapshot_date
    zero_date = st.session_state.get("zero_start_date")
    if isinstance(zero_date, date):
        return zero_date
    return default_start_date


def chunked_rows(rows: list[tuple[Any, ...]], chunk_size: int = DB_WRITE_CHUNK_SIZE) -> list[list[tuple[Any, ...]]]:
    return [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]


def normalize_mydata_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame(columns=["instr_id", "date1", "data1", "custom_fields", "date1_dt"])

    required_cols = ["instr_id", "date1", "data1", "custom_fields"]
    for column in required_cols:
        if column not in dataframe.columns:
            dataframe[column] = ""

    normalized = dataframe[required_cols].copy()
    normalized["instr_id"] = normalized["instr_id"].fillna("").astype(str)
    normalized["data1"] = normalized["data1"].fillna("").astype(str)
    normalized["custom_fields"] = normalized["custom_fields"].fillna("").astype(str)
    normalized["date1_dt"] = pd.to_datetime(normalized["date1"], errors="coerce")

    if not normalized.empty and getattr(normalized["date1_dt"].dt, "tz", None) is not None:
        normalized["date1_dt"] = normalized["date1_dt"].dt.tz_convert(None)

    normalized = normalized[normalized["date1_dt"].notna()].copy()
    normalized["date1"] = normalized["date1_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    normalized.sort_values(by="date1_dt", inplace=True)
    normalized.reset_index(drop=True, inplace=True)
    return normalized


def load_mydata_json_dataframe(root: Path) -> pd.DataFrame:
    json_path = root / "validation_data" / "mydata.json"
    if not json_path.exists():
        return pd.DataFrame(columns=["instr_id", "date1", "data1", "custom_fields"])
    try:
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (json.JSONDecodeError, OSError):
        return pd.DataFrame(columns=["instr_id", "date1", "data1", "custom_fields"])

    if not isinstance(payload, list):
        return pd.DataFrame(columns=["instr_id", "date1", "data1", "custom_fields"])
    return pd.DataFrame(payload)


def prepare_mydata_for_write(
    root: Path,
    start_date: date,
    end_date: date,
    events: list[Event],
    start_mode: str = TIME_SERIES_START_MODE_ZERO,
    runtime: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, bool, str, list[Instrument], list[Event]]:
    append_stream_log(root, "Preparing data source for write request.", runtime=runtime)
    from_session = st.session_state.get("mydata_df_records", [])
    if isinstance(from_session, list) and from_session:
        dataframe = normalize_mydata_dataframe(pd.DataFrame(from_session))
        if not dataframe.empty:
            append_stream_log(root, f"Using in-memory mydata_df with {len(dataframe)} rows.", runtime=runtime)
            return dataframe, False, "Using in-memory mydata_df from current session.", [], events

    from_json = normalize_mydata_dataframe(load_mydata_json_dataframe(root))
    if not from_json.empty:
        append_stream_log(root, f"Using validation_data/mydata.json with {len(from_json)} rows.", runtime=runtime)
        return from_json, False, "Using validation_data/mydata.json.", [], events

    append_stream_log(root, "No existing mydata source found; starting re-assimilation.", runtime=runtime)
    instruments, refreshed_events = load_or_synthesise_valid_instruments(
        root,
        start_date,
        end_date,
        events,
        start_mode=start_mode,
    )
    rebuilt = normalize_mydata_dataframe(build_assimilation_dataframe(instruments))
    if not rebuilt.empty:
        save_mydata_json(root, rebuilt[["instr_id", "date1", "data1", "custom_fields"]])
        append_stream_log(root, f"Re-assimilation produced {len(rebuilt)} rows.", runtime=runtime)
        return (
            rebuilt,
            True,
            "Re-assimilated data because mydata_df and mydata.json were unavailable or empty.",
            instruments,
            refreshed_events,
        )

    append_stream_log(root, "Data preparation failed: no rows available from any source.", runtime=runtime)
    return rebuilt, True, "Could not source data from mydata_df, mydata.json, or re-assimilation.", instruments, refreshed_events


def write_timeseries_batch(target_cursor, table_name: str, rows: list[tuple[str, str, str, str]]) -> None:
    insert_sql = (
        f"INSERT INTO `{table_name}` (`instr_id`, `date1`, `data1`, `custom_fields`) "
        "VALUES (%s, %s, %s, %s)"
    )
    target_cursor.executemany(insert_sql, rows)


def quote_sql_identifier(identifier: str) -> str:
    parts = [part.strip() for part in str(identifier).split(".") if part.strip()]
    if not parts:
        raise ValueError("SQL identifier cannot be empty")
    return ".".join(f"`{part}`" for part in parts)


def build_exact_match_filter_clause(
    column_names: tuple[str, ...],
    values: list[tuple[Any, ...]],
) -> tuple[str, tuple[Any, ...]]:
    if not values:
        return "1 = 0", ()

    row_clauses: list[str] = []
    params: list[Any] = []
    for row_values in values:
        row_clause = " AND ".join(f"{quote_sql_identifier(column_name)} = %s" for column_name in column_names)
        row_clauses.append(f"({row_clause})")
        params.extend(row_values)

    return " OR ".join(row_clauses), tuple(params)


def build_in_clause(column_name: str, values: list[Any]) -> tuple[str, tuple[Any, ...]]:
    placeholders = ", ".join(["%s"] * len(values))
    return f"{quote_sql_identifier(column_name)} IN ({placeholders})", tuple(values)


def fetch_source_rows(
    source_cursor,
    table_name: str,
    where_clause: str | None = None,
    params: tuple[Any, ...] = (),
    order_by: str | None = None,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    query = f"SELECT * FROM `{table_name}`"
    if where_clause:
        query = f"{query} WHERE {where_clause}"
    if order_by:
        query = f"{query} ORDER BY {order_by}"
    source_cursor.execute(query, params)
    description = source_cursor.description or []
    columns = [column_info[0] for column_info in description]
    rows = list(source_cursor.fetchall()) if columns else []
    return columns, rows


def add_table_copy_plan_entry(
    copy_plan: dict[str, dict[str, Any]],
    source_cursor,
    table_name: str,
    where_clause: str | None = None,
    params: tuple[Any, ...] = (),
    order_by: str | None = None,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    columns, rows = fetch_source_rows(
        source_cursor,
        table_name,
        where_clause=where_clause,
        params=params,
        order_by=order_by,
    )
    copy_plan[table_name] = {
        "columns": columns,
        "rows": rows,
    }
    return columns, rows


def collect_distinct_column_values(
    columns: list[str],
    rows: list[tuple[Any, ...]],
    column_name: str,
    *,
    cast: type | None = None,
    exclude: set[Any] | None = None,
) -> list[Any]:
    if not columns or not rows:
        return []

    column_index = columns.index(column_name)
    excluded_values = exclude or set()
    values: set[Any] = set()
    for row in rows:
        value = row[column_index]
        if value in excluded_values:
            continue
        values.add(cast(value) if cast is not None else value)
    return sorted(values)


def build_supporting_table_copy_plan(source_cursor) -> dict[str, dict[str, Any]]:
    copy_plan: dict[str, dict[str, Any]] = {}

    raw_type_where, raw_type_params = build_exact_match_filter_clause(("type", "subtype"), RAW_INSTR_TYPE_FILTERS)
    add_table_copy_plan_entry(
        copy_plan,
        source_cursor,
        "raw_instr_typestbl",
        where_clause=raw_type_where,
        params=raw_type_params,
        order_by="`type`, `subtype`",
    )

    instrum_where, instrum_params = build_exact_match_filter_clause(("type1", "subtype1"), RAW_INSTR_TYPE_FILTERS)
    instrum_columns, instrum_rows = add_table_copy_plan_entry(
        copy_plan,
        source_cursor,
        "instrum",
        where_clause=instrum_where,
        params=instrum_params,
        order_by="`id`",
    )

    instr_ids = collect_distinct_column_values(instrum_columns, instrum_rows, "instr_id", cast=str, exclude={"", None})
    location_ids = collect_distinct_column_values(instrum_columns, instrum_rows, "location_id", exclude={None})

    if location_ids:
        location_where, location_params = build_in_clause("id", location_ids)
        add_table_copy_plan_entry(
            copy_plan,
            source_cursor,
            "location",
            where_clause=location_where,
            params=location_params,
            order_by="`id`",
        )
    else:
        copy_plan["location"] = {"columns": [], "rows": []}

    if instr_ids:
        hierarchy_member_where, hierarchy_member_params = build_in_clause("instr_id", instr_ids)
        hierarchy_member_columns, hierarchy_member_rows = add_table_copy_plan_entry(
            copy_plan,
            source_cursor,
            "hierarchy_members",
            where_clause=hierarchy_member_where,
            params=hierarchy_member_params,
            order_by="`id`",
        )
        review_instrument_columns, review_instrument_rows = add_table_copy_plan_entry(
            copy_plan,
            source_cursor,
            "review_instruments",
            where_clause=hierarchy_member_where,
            params=hierarchy_member_params,
            order_by="`id`",
        )
        add_table_copy_plan_entry(
            copy_plan,
            source_cursor,
            "instr_cal_calibs",
            where_clause=hierarchy_member_where,
            params=hierarchy_member_params,
            order_by="`id`",
        )
    else:
        hierarchy_member_columns, hierarchy_member_rows = [], []
        review_instrument_columns, review_instrument_rows = [], []
        copy_plan["hierarchy_members"] = {"columns": [], "rows": []}
        copy_plan["review_instruments"] = {"columns": [], "rows": []}
        copy_plan["instr_cal_calibs"] = {"columns": [], "rows": []}

    hierarchy_ids = collect_distinct_column_values(
        hierarchy_member_columns,
        hierarchy_member_rows,
        "hierarchy_id",
        exclude={None},
    )

    if hierarchy_ids:
        hierarchies_where, hierarchies_params = build_in_clause("id", hierarchy_ids)
        add_table_copy_plan_entry(
            copy_plan,
            source_cursor,
            "hierarchies",
            where_clause=hierarchies_where,
            params=hierarchies_params,
            order_by="`id`",
        )
    else:
        copy_plan["hierarchies"] = {"columns": [], "rows": []}

    review_instr_ids = collect_distinct_column_values(
        review_instrument_columns,
        review_instrument_rows,
        "id",
        exclude={None},
    )

    if review_instr_ids:
        review_values_where, review_values_params = build_in_clause("review_instr_id", review_instr_ids)
        add_table_copy_plan_entry(
            copy_plan,
            source_cursor,
            "review_instruments_values",
            where_clause=review_values_where,
            params=review_values_params,
            order_by="`id`",
        )
    else:
        copy_plan["review_instruments_values"] = {"columns": [], "rows": []}

    add_table_copy_plan_entry(
        copy_plan,
        source_cursor,
        "aaa_color_info",
        order_by="`id`",
    )

    return copy_plan


def write_table_rows_to_target(
    target_cursor,
    target_connection,
    table_name: str,
    columns: list[str],
    rows: list[tuple[Any, ...]],
    cancel_event: threading.Event,
    root: Path,
    runtime: dict[str, Any] | None = None,
) -> bool:
    if cancel_event.is_set():
        return False

    if not columns:
        append_stream_log(root, f"Skipped table {table_name}: no columns returned.", runtime=runtime)
        return True

    if not rows:
        append_stream_log(root, f"Skipped table {table_name}: no matching filtered rows in source.", runtime=runtime)
        return True

    column_sql = ", ".join(quote_sql_identifier(column) for column in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = f"INSERT INTO `{table_name}` ({column_sql}) VALUES ({placeholders})"

    written_rows = 0
    for chunk in chunked_rows(rows, DB_WRITE_CHUNK_SIZE):
        if cancel_event.is_set():
            return False
        target_cursor.executemany(insert_sql, chunk)
        target_connection.commit()
        written_rows += len(chunk)

    append_stream_log(root, f"Copied filtered rows into target table {table_name} (rows_written={written_rows}).", runtime=runtime)
    return True


def clear_target_table(
    target_cursor,
    target_connection,
    table_name: str,
    cancel_event: threading.Event,
    root: Path,
    runtime: dict[str, Any] | None = None,
) -> bool:
    if cancel_event.is_set():
        return False
    target_cursor.execute(f"DELETE FROM `{table_name}`")
    target_connection.commit()
    append_stream_log(root, f"Cleared target table {table_name}.", runtime=runtime)
    return True


def run_database_write_job(
    root: Path,
    normalized_rows: list[dict[str, Any]],
    runtime: dict[str, Any],
    preserve_before_date: date | None = None,
) -> None:
    cancel_event: threading.Event = runtime["cancel_event"]
    source_connection = None
    target_connection = None

    try:
        append_stream_log(root, "Write job started.", runtime=runtime)
        update_db_write_state(
            runtime=runtime,
            status="running",
            async_status="writing_past_rows",
            rows_written=0,
            total_rows=len(normalized_rows),
            percent_complete=0.0,
            message="Connected and preparing writes.",
            started=True,
            thread_alive=True,
            thread_completed_at="",
            thread_terminal_message="",
        )

        source_config, target_config = read_db_configs(root)
        append_stream_log(root, "Read source and target database configuration.", runtime=runtime)
        source_connection = connect_mysql(source_config, pymysql.cursors.Cursor)
        runtime["source_connection"] = source_connection
        append_stream_log(root, "Connected to source database in read-only workflow mode.", runtime=runtime)
        target_connection = connect_mysql(target_config, pymysql.cursors.Cursor)
        runtime["target_connection"] = target_connection
        append_stream_log(root, "Connected to target database in write mode.", runtime=runtime)

        with source_connection.cursor() as source_cursor, target_connection.cursor() as target_cursor:
            if cancel_event.is_set():
                update_db_write_state(runtime=runtime, status="cancelled", async_status="cancelled", message="Write cancelled before start.")
                append_stream_log(root, "Write cancelled before start.", runtime=runtime)
                return

            target_cursor.execute("CREATE TABLE IF NOT EXISTS `futuredata` LIKE `mydata`")
            target_connection.commit()
            append_stream_log(root, "Ensured target table futuredata exists with the mydata schema.", runtime=runtime)
            if preserve_before_date is not None:
                delete_target_rows_from_date(
                    target_cursor,
                    target_connection,
                    "mydata",
                    preserve_before_date,
                    root,
                    runtime,
                )
                delete_target_rows_from_date(
                    target_cursor,
                    target_connection,
                    "futuredata",
                    preserve_before_date,
                    root,
                    runtime,
                )
            else:
                target_cursor.execute("DELETE FROM `mydata`")
                target_cursor.execute("DELETE FROM `futuredata`")
                target_connection.commit()
                append_stream_log(root, "Cleared target tables mydata and futuredata.", runtime=runtime)

            updated_to_time = datetime.now() - timedelta(hours=24)
            past_rows: list[tuple[str, str, str, str]] = []
            future_rows: list[tuple[str, str, str, str]] = []
            for row in normalized_rows:
                date1_dt = parse_iso_datetime(row.get("date1_dt"))
                if date1_dt is None:
                    continue
                payload = (
                    str(row.get("instr_id", "")),
                    date1_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    str(row.get("data1", "")),
                    str(row.get("custom_fields", "")),
                )
                if date1_dt < updated_to_time:
                    past_rows.append(payload)
                else:
                    future_rows.append(payload)

            total_rows_to_write = len(past_rows) + len(future_rows)
            update_db_write_state(runtime=runtime, total_rows=total_rows_to_write)

            append_stream_log(
                root,
                (
                    f"Prepared row groups: past_rows={len(past_rows)}, "
                    f"future_rows={len(future_rows)}, total_rows={total_rows_to_write}, "
                    f"preserved_rows={(0 if preserve_before_date is None else 'unchanged_in_place')}, "
                    f"updated_to_time={updated_to_time.strftime('%Y-%m-%d %H:%M:%S')}."
                ),
                runtime=runtime,
            )

            written_rows = 0
            past_batches = chunked_rows(past_rows, DB_WRITE_CHUNK_SIZE)
            for batch_index, batch in enumerate(past_batches, start=1):
                if cancel_event.is_set():
                    update_db_write_state(
                        runtime=runtime,
                        status="cancelled",
                        async_status="cancelled",
                        rows_written=written_rows,
                        percent_complete=(written_rows / total_rows_to_write * 100.0) if total_rows_to_write else 100.0,
                        message="Write cancelled while processing past rows.",
                    )
                    append_stream_log(root, "Write cancelled while processing past rows.", runtime=runtime)
                    return

                write_timeseries_batch(target_cursor, "mydata", batch)
                target_connection.commit()
                written_rows += len(batch)
                append_stream_log(
                    root,
                    (
                        f"Inserted batch {batch_index}/{max(len(past_batches), 1)} "
                        f"into mydata with {len(batch)} rows."
                    ),
                    runtime=runtime,
                )
                percent = (written_rows / total_rows_to_write * 100.0) if total_rows_to_write else 100.0
                update_db_write_state(
                    runtime=runtime,
                    status="running",
                    async_status="writing_past_rows",
                    rows_written=written_rows,
                    percent_complete=percent,
                    message=f"Written {written_rows} of {total_rows_to_write} rows to mydata/futuredata.",
                )

            future_batches = chunked_rows(future_rows, DB_WRITE_CHUNK_SIZE)
            for batch_index, batch in enumerate(future_batches, start=1):
                if cancel_event.is_set():
                    update_db_write_state(
                        runtime=runtime,
                        status="cancelled",
                        async_status="cancelled",
                        rows_written=written_rows,
                        percent_complete=(written_rows / total_rows_to_write * 100.0) if total_rows_to_write else 100.0,
                        message="Write cancelled while processing future rows.",
                    )
                    append_stream_log(root, "Write cancelled while processing future rows.", runtime=runtime)
                    return

                write_timeseries_batch(target_cursor, "futuredata", batch)
                target_connection.commit()
                written_rows += len(batch)
                append_stream_log(
                    root,
                    (
                        f"Inserted batch {batch_index}/{max(len(future_batches), 1)} "
                        f"into futuredata with {len(batch)} rows."
                    ),
                    runtime=runtime,
                )

                percent = (written_rows / total_rows_to_write * 100.0) if total_rows_to_write else 100.0
                update_db_write_state(
                    runtime=runtime,
                    status="running",
                    async_status="writing_future_rows",
                    rows_written=written_rows,
                    percent_complete=percent,
                    message=f"Written {written_rows} of {total_rows_to_write} rows across mydata and futuredata.",
                )

            if cancel_event.is_set():
                update_db_write_state(
                    runtime=runtime,
                    status="cancelled",
                    async_status="cancelled",
                    rows_written=written_rows,
                    percent_complete=(written_rows / total_rows_to_write * 100.0) if total_rows_to_write else 100.0,
                    message="Write cancelled before copying supporting tables.",
                )
                append_stream_log(root, "Write cancelled before copying supporting tables.", runtime=runtime)
                return

            append_stream_log(root, "Preparing filtered supporting-table copy plan.", runtime=runtime)
            supporting_table_copy_plan = build_supporting_table_copy_plan(source_cursor)

            for table_name in SUPPORTING_TABLES_CLEAR_ORDER:
                append_stream_log(root, f"Clearing supporting table {table_name}.", runtime=runtime)
                update_db_write_state(
                    runtime=runtime,
                    status="running",
                    async_status="copying_supporting_tables",
                    rows_written=written_rows,
                    percent_complete=(written_rows / total_rows_to_write * 100.0) if total_rows_to_write else 100.0,
                    message=f"Clearing supporting table {table_name}.",
                )
                cleared = clear_target_table(
                    target_cursor,
                    target_connection,
                    table_name,
                    cancel_event,
                    root,
                    runtime,
                )
                if not cleared:
                    update_db_write_state(
                        runtime=runtime,
                        status="cancelled",
                        async_status="cancelled",
                        rows_written=written_rows,
                        percent_complete=(written_rows / total_rows_to_write * 100.0) if total_rows_to_write else 100.0,
                        message=f"Write cancelled while clearing table {table_name}.",
                    )
                    append_stream_log(root, f"Write cancelled while clearing table {table_name}.", runtime=runtime)
                    return

            append_stream_log(root, "Supporting tables will be copied from filtered source rows after clearing target tables.", runtime=runtime)

            for table_name in SUPPORTING_TABLES_COPY_ORDER:
                append_stream_log(root, f"Starting copy for supporting table {table_name}.", runtime=runtime)
                update_db_write_state(
                    runtime=runtime,
                    status="running",
                    async_status="copying_supporting_tables",
                    rows_written=written_rows,
                    percent_complete=(written_rows / total_rows_to_write * 100.0) if total_rows_to_write else 100.0,
                    message=f"Copying supporting table {table_name}.",
                )
                table_copy_data = supporting_table_copy_plan.get(table_name, {"columns": [], "rows": []})
                copied = write_table_rows_to_target(
                    target_cursor,
                    target_connection,
                    table_name,
                    table_copy_data.get("columns", []),
                    table_copy_data.get("rows", []),
                    cancel_event,
                    root,
                    runtime,
                )
                if not copied:
                    update_db_write_state(
                        runtime=runtime,
                        status="cancelled",
                        async_status="cancelled",
                        rows_written=written_rows,
                        percent_complete=(written_rows / total_rows_to_write * 100.0) if total_rows_to_write else 100.0,
                        message=f"Write cancelled while copying table {table_name}.",
                    )
                    append_stream_log(root, f"Write cancelled while copying table {table_name}.", runtime=runtime)
                    return

            update_db_write_state(
                runtime=runtime,
                status="completed",
                async_status="completed",
                rows_written=written_rows,
                percent_complete=100.0,
                message="Write completed successfully for mydata, futuredata, and supporting tables.",
            )
            append_stream_log(root, "Write job completed successfully.", runtime=runtime)
    except Exception as error:
        if cancel_event.is_set():
            update_db_write_state(
                runtime=runtime,
                status="cancelled",
                async_status="cancelled",
                message="Write cancelled.",
            )
            append_stream_log(root, f"Write stopped due to cancellation: {error}", runtime=runtime)
        else:
            update_db_write_state(
                runtime=runtime,
                status="error",
                async_status="error",
                message=f"Write failed: {error}",
            )
            append_stream_log(root, f"Write failed: {error}", runtime=runtime)
    finally:
        if source_connection is not None:
            source_connection.close()
            append_stream_log(root, "Closed source database connection.", runtime=runtime)
            runtime["source_connection"] = None
        if target_connection is not None:
            target_connection.close()
            append_stream_log(root, "Closed target database connection.", runtime=runtime)
            runtime["target_connection"] = None

        lock = runtime["lock"]
        with lock:
            final_status = str(runtime["state"].get("status", ""))
        terminal_message = "Write thread ended."
        if final_status == "cancelled":
            terminal_message = "Cancellation completed. Write thread terminated."
        elif final_status == "completed":
            terminal_message = "Write completed. Thread terminated."
        elif final_status == "error":
            terminal_message = "Write ended with error. Thread terminated."

        update_db_write_state(
            runtime=runtime,
            thread_alive=False,
            thread_completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            thread_terminal_message=terminal_message,
        )
        runtime["thread"] = None
        append_stream_log(root, terminal_message, runtime=runtime)


def render_async_status_pill(status: str) -> None:
    label = status.replace("_", " ").title()
    color_map = {
        "idle": "gray",
        "starting": "blue",
        "writing_past_rows": "blue",
        "writing_future_rows": "blue",
        "copying_supporting_tables": "violet",
        "running": "green",
        "cancelling": "orange",
        "cancelled": "red",
        "completed": "green",
        "error": "red",
    }
    st.caption("Asynchronous writer status")
    st.badge(label, color=color_map.get(status, "gray"))


def render_write_status_logs(current_state: dict[str, Any]) -> None:
    state = str(current_state.get("status", "idle"))
    status_state = "running"
    if state == "completed":
        status_state = "complete"
    elif state in {"error", "cancelled"}:
        status_state = "error"

    logs = current_state.get("logs", [])
    if not isinstance(logs, list):
        logs = []

    latest_logs = [str(entry) for entry in logs[-7:]]
    with st.status("Database write log stream", state=status_state, expanded=True):
        if not latest_logs:
            st.write("No log events yet.")
        else:
            for line in latest_logs:
                st.write(line)


def get_projection_from_db(cursor) -> str:
    query = """
    SELECT projection_definition
    FROM crs_configuration
    WHERE is_deleted = 0
      AND projection_definition IS NOT NULL
      AND projection_definition != ''
    LIMIT 1
    """
    cursor.execute(query)
    row = cursor.fetchone()
    if not row or not row.get("projection_definition"):
        raise ValueError("No valid projection_definition found in crs_configuration")
    projection_definition = str(row["projection_definition"]).strip()
    if not projection_definition:
        raise ValueError("projection_definition is empty in crs_configuration")
    return projection_definition


def create_converter(projection_definition: str):
    from pyproj import Transformer

    return Transformer.from_proj(
        projection_definition,
        "+proj=longlat +datum=WGS84 +no_defs",
        always_xy=True,
    )


def create_validation_plot(converted_rows: list[dict], root: Path) -> Path | None:
    if not converted_rows:
        return None

    plot_dir = root / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = plot_dir / f"instr_locations_original_{timestamp}.html"

    figure = go.Figure()
    ordered_types = ["CASA", "PZ", "LP", "TLP", "INCL", "EI"]

    for instrument_type in ordered_types:
        series = [row for row in converted_rows if row.get("db_type") == instrument_type]
        if not series:
            continue

        figure.add_trace(
            go.Scattermapbox(
                lat=[row["latitude"] for row in series],
                lon=[row["longitude"] for row in series],
                mode="markers",
                marker={"size": 8, "color": DB_TYPE_COLORS.get(instrument_type, "#ffffff")},
                name=instrument_type,
                text=[row["instr_id"] for row in series],
                hovertemplate="instr_id=%{text}<br>db_type="
                + instrument_type
                + "<br>lat=%{lat}<br>lon=%{lon}<extra></extra>",
            )
        )

    mean_lat = sum(row["latitude"] for row in converted_rows) / len(converted_rows)
    mean_lon = sum(row["longitude"] for row in converted_rows) / len(converted_rows)

    figure.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_zoom=15,
        mapbox_center={"lat": mean_lat, "lon": mean_lon},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title="Instrument locations by type",
        legend_title_text="Type",
    )

    figure.write_html(str(plot_path), include_plotlyjs="cdn")
    return plot_path


def extract_instruments(root: Path) -> tuple[list[Instrument], Path, Path | None]:
    secrets_path = root / ".streamlit" / "secrets.toml"
    validation_data_dir = root / "validation_data"
    validation_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = validation_data_dir / "instruments.json"

    instrument_types = ["CASA", "PZ", "LP", "TLP", "INCL", "EI"]

    with secrets_path.open("rb") as file:
        secrets = tomllib.load(file)

    config = secrets["project_data"]["hanoi_live"]

    connection = pymysql.connect(
        host=config["db_host"],
        user=config["db_user"],
        password=config["db_pass"],
        database=config["db_name"],
        port=int(config.get("port", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

    query = """
    SELECT i.instr_id, i.type1 AS db_type, l.easting, l.northing
    FROM instrum i
    JOIN location l ON i.location_id = l.id
    WHERE i.type1 IN (%s, %s, %s, %s, %s, %s)
      AND l.project_id = 108
      AND l.contract_id = 104
      AND i.instr_id <> 'SH-SH-01-EI-02'
    ORDER BY i.instr_id
    """

    with connection:
        with connection.cursor() as cursor:
            projection_definition = get_projection_from_db(cursor)
            transformer = create_converter(projection_definition)
            cursor.execute(query, instrument_types)
            rows = cursor.fetchall()

            parent_ids = [
                str(row.get("instr_id", ""))
                for row in rows
                if EVENT_TYPE_BY_DB_TYPE.get(str(row.get("db_type", "")), "") == "subsurface_lateral_displacement"
            ]

            child_rows: list[dict] = []
            if parent_ids:
                placeholders = ", ".join(["%s"] * len(parent_ids))
                child_query = f"""
                SELECT h.master_instr AS parent_instr_id, hm.instr_id AS child_instr_id
                FROM hierarchies h
                JOIN hierarchy_members hm ON h.id = hm.hierarchy_id
                WHERE h.master_instr IN ({placeholders})
                """
                cursor.execute(child_query, parent_ids)
                child_rows = cursor.fetchall()

    valid_rows = []
    eastings = []
    northings = []
    for row in rows:
        try:
            easting = float(row.get("easting"))
            northing = float(row.get("northing"))
        except (TypeError, ValueError):
            continue
        valid_rows.append(row)
        eastings.append(easting)
        northings.append(northing)

    longitudes, latitudes = transformer.transform(eastings, northings) if valid_rows else ([], [])

    def parse_depth_from_child_instr_id(child_instr_id: str) -> float | None:
        match = re.search(r"Depth\s*-?\s*(-?\d+(?:\.\d+)?)\s*m?", child_instr_id, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return abs(float(match.group(1)))
        except (TypeError, ValueError):
            return None

    children_by_parent: dict[str, list[tuple[str, float]]] = {}
    for child_row in child_rows:
        parent_instr_id = str(child_row.get("parent_instr_id", ""))
        child_instr_id = str(child_row.get("child_instr_id", ""))
        depth_m = parse_depth_from_child_instr_id(child_instr_id)
        if depth_m is None:
            continue
        children_by_parent.setdefault(parent_instr_id, []).append((child_instr_id, depth_m))

    instruments: list[Instrument] = []
    converted_rows: list[dict] = []
    for row, latitude, longitude, easting, northing in zip(valid_rows, latitudes, longitudes, eastings, northings):
        db_type = str(row.get("db_type", ""))
        event_type = EVENT_TYPE_BY_DB_TYPE.get(db_type, "")
        children = children_by_parent.get(str(row.get("instr_id", "")), []) if event_type == "subsurface_lateral_displacement" else None
        instrument = Instrument(
            instr_id=str(row.get("instr_id", "")),
            db_type=db_type,
            event_type=event_type,
            easting=easting,
            northing=northing,
            latitude=float(latitude),
            longitude=float(longitude),
            timeseries=[],
            children=children,
        )
        instruments.append(instrument)
        converted_rows.append(
            {
                "instr_id": instrument.instr_id,
                "db_type": instrument.db_type,
                "easting": instrument.easting,
                "northing": instrument.northing,
                "latitude": instrument.latitude,
                "longitude": instrument.longitude,
            }
        )

    save_instruments_to_json(root, instruments)

    plot_path = create_validation_plot(converted_rows, root)
    return instruments, output_path, plot_path


def save_instruments_to_json(root: Path, instruments: list[Instrument]) -> Path:
    validation_data_dir = root / "validation_data"
    validation_data_dir.mkdir(parents=True, exist_ok=True)
    json_path = validation_data_dir / "instruments.json"

    payload = []
    for instrument in instruments:
        payload.append(
            {
                "instr_id": instrument.instr_id,
                "db_type": instrument.db_type,
                "event_type": instrument.event_type,
                "easting": instrument.easting,
                "northing": instrument.northing,
                "latitude": instrument.latitude,
                "longitude": instrument.longitude,
                "timeseries": instrument.timeseries,
                "children": (
                    [{"child_instr_id": child_id, "depth_m": depth_m} for child_id, depth_m in instrument.children]
                    if instrument.children is not None
                    else None
                ),
            }
        )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return json_path


def load_instruments_from_json(root: Path) -> list[Instrument]:
    validation_data_dir = root / "validation_data"
    json_path = validation_data_dir / "instruments.json"
    if not json_path.exists():
        return []

    try:
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    instruments: list[Instrument] = []
    for row in payload if isinstance(payload, list) else []:
        try:
            db_type = str(row.get("db_type", ""))
            event_type = str(row.get("event_type") or EVENT_TYPE_BY_DB_TYPE.get(db_type, ""))
            children_payload = row.get("children")
            children: list[tuple[str, float]] | None
            if children_payload is None:
                children = None
            else:
                parsed_children: list[tuple[str, float]] = []
                for child in children_payload:
                    child_id = str(child.get("child_instr_id", ""))
                    parsed_children.append((child_id, float(child.get("depth_m", 0.0))))
                children = parsed_children

            instruments.append(
                Instrument(
                    instr_id=str(row.get("instr_id", "")),
                    db_type=db_type,
                    event_type=event_type,
                    easting=float(row.get("easting", 0.0)),
                    northing=float(row.get("northing", 0.0)),
                    latitude=float(row.get("latitude", 0.0)),
                    longitude=float(row.get("longitude", 0.0)),
                    timeseries=list(row.get("timeseries", [])),
                    children=children,
                )
            )
        except (TypeError, ValueError, AttributeError):
            continue
    return instruments


def save_events_to_json(root: Path, events: list[Event]) -> Path:
    validation_data_dir = root / "validation_data"
    validation_data_dir.mkdir(parents=True, exist_ok=True)
    json_path = validation_data_dir / "events.json"

    payload = []
    for event in events:
        payload.append(
            {
                "start_time": event.start_time.isoformat(),
                "epicentre": {
                    "easting": event.epicentre.easting,
                    "northing": event.epicentre.northing,
                    "latitude": event.epicentre.latitude,
                    "longitude": event.epicentre.longitude,
                },
                "type": event.type,
                "magnitude_mm": event.magnitude_mm,
                "duration_days": event.duration_days,
                "radius_m": event.radius_m,
                "direction_deg": event.direction_deg,
            }
        )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return json_path


def load_events_from_json(root: Path) -> list[Event]:
    validation_data_dir = root / "validation_data"
    json_path = validation_data_dir / "events.json"
    if not json_path.exists():
        return []

    try:
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    events: list[Event] = []
    for row in payload if isinstance(payload, list) else []:
        try:
            epicentre_data = row.get("epicentre", {})
            events.append(
                Event(
                    start_time=datetime.fromisoformat(str(row.get("start_time"))),
                    epicentre=Location(
                        easting=float(epicentre_data.get("easting", 0.0)),
                        northing=float(epicentre_data.get("northing", 0.0)),
                        latitude=float(epicentre_data.get("latitude", 0.0)),
                        longitude=float(epicentre_data.get("longitude", 0.0)),
                    ),
                    type=str(row.get("type", "")),
                    magnitude_mm=float(row.get("magnitude_mm", 0.0)),
                    duration_days=float(row.get("duration_days", 0.0)),
                    radius_m=float(row.get("radius_m", 0.0)),
                    direction_deg=(
                        None
                        if row.get("direction_deg") is None
                        else float(row.get("direction_deg", 0.0))
                    ),
                )
            )
        except (TypeError, ValueError, AttributeError):
            continue

    events.sort(key=lambda event: event.start_time)
    return events


def six_months_before(input_date: date) -> date:
    year = input_date.year
    month = input_date.month - 6
    if month <= 0:
        month += 12
        year -= 1
    day = min(input_date.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def sample_beta_scaled(alpha: float, beta: float, maximum: float) -> float:
    return random.betavariate(alpha, beta) * maximum


def get_db_transformer(root: Path):
    with (root / ".streamlit" / "secrets.toml").open("rb") as file:
        config = tomllib.load(file)["project_data"]["hanoi_live"]

    connection = pymysql.connect(
        host=config["db_host"],
        user=config["db_user"],
        password=config["db_pass"],
        database=config["db_name"],
        port=int(config.get("port", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

    with connection:
        with connection.cursor() as cursor:
            projection_definition = get_projection_from_db(cursor)

    return create_converter(projection_definition)


def build_epicentre(event_type: str, instruments: list[Instrument], transformer) -> Location:
    matching = [instrument for instrument in instruments if instrument.event_type == event_type]
    if not matching:
        raise ValueError(f"No instruments found for event_type={event_type}")

    anchor = random.choice(matching)
    easting = anchor.easting + sample_beta_scaled(1, 2, 10)
    northing = anchor.northing + sample_beta_scaled(1, 2, 10)
    longitude, latitude = transformer.transform(easting, northing)
    return Location(easting=easting, northing=northing, latitude=float(latitude), longitude=float(longitude))


def sample_event_attributes(event_type: str) -> tuple[float, float, float, float | None]:
    params = EVENT_DISTRIBUTION_PARAMS[event_type]
    if event_type == "surface_settlement":
        is_negative = random.random() < params["magnitude_negative_prob"]
        if is_negative:
            negative = params["magnitude_negative"]
            magnitude_mm = -sample_beta_scaled(negative["alpha"], negative["beta"], negative["maximum"])
        else:
            positive = params["magnitude_positive"]
            magnitude_mm = sample_beta_scaled(positive["alpha"], positive["beta"], positive["maximum"])
        duration = params["duration_days"]
        radius = params["radius_m"]
        duration_days = sample_beta_scaled(duration["alpha"], duration["beta"], duration["maximum"])
        radius_m = sample_beta_scaled(radius["alpha"], radius["beta"], radius["maximum"])
        direction_deg = None
    elif event_type == "groundwater_level":
        magnitude = params["magnitude_positive"]
        duration = params["duration_days"]
        radius = params["radius_m"]
        magnitude_mm = sample_beta_scaled(magnitude["alpha"], magnitude["beta"], magnitude["maximum"])
        duration_days = sample_beta_scaled(duration["alpha"], duration["beta"], duration["maximum"])
        radius_m = sample_beta_scaled(radius["alpha"], radius["beta"], radius["maximum"])
        direction_deg = None
    else:
        magnitude = params["magnitude_positive"]
        duration = params["duration_days"]
        radius = params["radius_m"]
        magnitude_mm = sample_beta_scaled(magnitude["alpha"], magnitude["beta"], magnitude["maximum"])
        duration_days = sample_beta_scaled(duration["alpha"], duration["beta"], duration["maximum"])
        radius_m = sample_beta_scaled(radius["alpha"], radius["beta"], radius["maximum"])
        direction_deg = random.uniform(0, 360)
    return magnitude_mm, duration_days, radius_m, direction_deg


def generate_events(
    instruments: list[Instrument],
    start_date: date,
    end_date: date,
    transformer,
    logger: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[Event]:
    events: list[Event] = []
    current = start_date
    total_days = max(0, (end_date - start_date).days + 1)
    if logger is not None:
        logger(
            f"Generating event history for {len(instruments)} instruments from {start_date.isoformat()} to {end_date.isoformat()}."
        )
    day_index = 0
    while current <= end_date:
        day_index += 1
        daily_type_counts = {event_type: 0 for event_type in EVENT_TYPES}
        for event_type in EVENT_TYPES:
            if random.random() >= EVENT_PROBABILITIES[event_type]:
                continue

            magnitude_mm, duration_days, radius_m, direction_deg = sample_event_attributes(event_type)
            epicentre = build_epicentre(event_type, instruments, transformer)
            start_time = datetime.combine(current, time.min) + timedelta(seconds=random.randint(0, 86399))
            events.append(
                Event(
                    start_time=start_time,
                    epicentre=epicentre,
                    type=event_type,
                    magnitude_mm=float(magnitude_mm),
                    duration_days=float(duration_days),
                    radius_m=float(radius_m),
                    direction_deg=None if direction_deg is None else float(direction_deg),
                )
            )
            daily_type_counts[event_type] += 1
        if logger is not None:
            total_generated = sum(daily_type_counts.values())
            if total_generated == 0:
                logger(f"{current.isoformat()}: no events generated.")
            else:
                summary = ", ".join(
                    f"{event_type}={count}"
                    for event_type, count in daily_type_counts.items()
                    if count > 0
                )
                logger(f"{current.isoformat()}: generated {total_generated} events ({summary}).")
        if progress_callback is not None:
            progress_callback(day_index, total_days, f"Generated events for {day_index} of {total_days} days.")
        current += timedelta(days=1)
    events.sort(key=lambda event: event.start_time)
    if logger is not None:
        logger(f"Finished event generation with {len(events)} total events.")
    return events


def save_beta_distribution_plot(root: Path, timestamp: str) -> Path:
    def beta_pdf(x_value: float, alpha: float, beta: float) -> float:
        if x_value <= 0 or x_value >= 1:
            return 0.0
        numerator = (x_value ** (alpha - 1)) * ((1 - x_value) ** (beta - 1))
        denominator = (
            __import__("math").gamma(alpha) * __import__("math").gamma(beta) / __import__("math").gamma(alpha + beta)
        )
        return numerator / denominator

    grid = [i / 250 for i in range(1, 250)]

    attribute_groups = {
        "magnitude_mm": [
            (
                "surface_settlement (negative)",
                "surface_settlement",
                EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["magnitude_negative"]["alpha"],
                EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["magnitude_negative"]["beta"],
                -EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["magnitude_negative"]["maximum"],
                "dash",
            ),
            (
                "surface_settlement (positive)",
                "surface_settlement",
                EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["magnitude_positive"]["alpha"],
                EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["magnitude_positive"]["beta"],
                EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["magnitude_positive"]["maximum"],
                "solid",
            ),
            (
                "groundwater_level",
                "groundwater_level",
                EVENT_DISTRIBUTION_PARAMS["groundwater_level"]["magnitude_positive"]["alpha"],
                EVENT_DISTRIBUTION_PARAMS["groundwater_level"]["magnitude_positive"]["beta"],
                EVENT_DISTRIBUTION_PARAMS["groundwater_level"]["magnitude_positive"]["maximum"],
                "solid",
            ),
            (
                "subsurface_lateral_displacement",
                "subsurface_lateral_displacement",
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["magnitude_positive"]["alpha"],
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["magnitude_positive"]["beta"],
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["magnitude_positive"]["maximum"],
                "solid",
            ),
        ],
        "duration_days": [
            (
                "surface_settlement",
                "surface_settlement",
                EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["duration_days"]["alpha"],
                EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["duration_days"]["beta"],
                EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["duration_days"]["maximum"],
                "solid",
            ),
            (
                "groundwater_level",
                "groundwater_level",
                EVENT_DISTRIBUTION_PARAMS["groundwater_level"]["duration_days"]["alpha"],
                EVENT_DISTRIBUTION_PARAMS["groundwater_level"]["duration_days"]["beta"],
                EVENT_DISTRIBUTION_PARAMS["groundwater_level"]["duration_days"]["maximum"],
                "solid",
            ),
            (
                "subsurface_lateral_displacement",
                "subsurface_lateral_displacement",
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["duration_days"]["alpha"],
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["duration_days"]["beta"],
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["duration_days"]["maximum"],
                "solid",
            ),
        ],
        "radius_m": [
            (
                "surface_settlement",
                "surface_settlement",
                EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["radius_m"]["alpha"],
                EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["radius_m"]["beta"],
                EVENT_DISTRIBUTION_PARAMS["surface_settlement"]["radius_m"]["maximum"],
                "solid",
            ),
            (
                "groundwater_level",
                "groundwater_level",
                EVENT_DISTRIBUTION_PARAMS["groundwater_level"]["radius_m"]["alpha"],
                EVENT_DISTRIBUTION_PARAMS["groundwater_level"]["radius_m"]["beta"],
                EVENT_DISTRIBUTION_PARAMS["groundwater_level"]["radius_m"]["maximum"],
                "solid",
            ),
            (
                "subsurface_lateral_displacement",
                "subsurface_lateral_displacement",
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["radius_m"]["alpha"],
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["radius_m"]["beta"],
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["radius_m"]["maximum"],
                "solid",
            ),
        ],
        "direction_deg": [
            (
                "subsurface_lateral_displacement",
                "subsurface_lateral_displacement",
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["direction_deg"]["alpha"],
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["direction_deg"]["beta"],
                EVENT_DISTRIBUTION_PARAMS["subsurface_lateral_displacement"]["direction_deg"]["maximum"],
                "solid",
            ),
        ],
    }

    subplot_titles = [
        "magnitude_mm",
        "duration_days",
        "radius_m",
        "direction_deg",
    ]
    figure = make_subplots(rows=4, cols=1, subplot_titles=subplot_titles, vertical_spacing=0.06)

    for row_index, attribute in enumerate(subplot_titles, start=1):
        for label, event_type, alpha, beta, maximum, dash in attribute_groups[attribute]:
            if attribute == "magnitude_mm" and maximum < 0:
                x_values = [-(x_value * abs(maximum)) for x_value in grid]
            else:
                x_values = [x_value * maximum for x_value in grid]
            y_values = [beta_pdf(x_value, alpha, beta) for x_value in grid]
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=label,
                    legendgroup=event_type,
                    line={"color": TYPE_COLORS[event_type], "dash": dash},
                ),
                row=row_index,
                col=1,
            )

    figure.update_layout(height=1100, title="Beta distributions grouped by attribute")
    plot_dir = root / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"event_beta_distributions_{timestamp}.html"
    figure.write_html(str(path), include_plotlyjs="cdn")
    return path


def event_hover_text(event: Event) -> str:
    return (
        f"start_time={event.start_time.isoformat()}<br>"
        f"type={event.type}<br>"
        f"magnitude_mm={event.magnitude_mm:.4f}<br>"
        f"duration_days={event.duration_days:.4f}<br>"
        f"radius_m={event.radius_m:.4f}<br>"
        f"direction_deg={event.direction_deg if event.direction_deg is not None else 'None'}<br>"
        f"easting={event.epicentre.easting:.4f}<br>"
        f"northing={event.epicentre.northing:.4f}<br>"
        f"latitude={event.epicentre.latitude:.8f}<br>"
        f"longitude={event.epicentre.longitude:.8f}"
    )


def max_possible_magnitude(event_type: str) -> float:
    params = EVENT_DISTRIBUTION_PARAMS[event_type]
    positive = float(params["magnitude_positive"]["maximum"])
    negative = float(params.get("magnitude_negative", {}).get("maximum", 0.0))
    return max(positive, negative)


def kumaraswamy_cdf(x_value: float, alpha: float, beta: float) -> float:
    clipped = max(0.0, min(1.0, x_value))
    return 1.0 - ((1.0 - (clipped**alpha)) ** beta)


def kumaraswamy_decay(x_value: float, alpha: float, beta: float) -> float:
    return 1.0 - kumaraswamy_cdf(x_value, alpha, beta)


def instrument_distance_m(instrument: Instrument, event: Event) -> float:
    return math.hypot(instrument.easting - event.epicentre.easting, instrument.northing - event.epicentre.northing)


def build_measurement_days(start_date: date, end_date: date, include_start_on_sunday: bool = True) -> list[date]:
    days: list[date] = []
    current = start_date
    while current <= end_date:
        if current.weekday() != 6 or (include_start_on_sunday and current == start_date):
            days.append(current)
        current += timedelta(days=1)
    return days


def build_noise_event_days(start_date: date, end_date: date) -> list[date]:
    event_days: list[date] = []
    current = start_date - timedelta(days=NOISE_EVENT_TIME_WINDOW_DAYS)
    final_day = end_date + timedelta(days=NOISE_EVENT_TIME_WINDOW_DAYS)
    while current <= final_day:
        event_days.append(current)
        current += timedelta(days=1)
    return event_days


def gaussian_weight(offset: float, sigma: float) -> float:
    if sigma <= 0.0:
        return 1.0 if abs(offset) < 1e-12 else 0.0
    return math.exp(-0.5 * ((offset / sigma) ** 2))


def sample_lognormal_median_shape(median: float, shape: float) -> float:
    if median <= 0.0:
        return 0.0
    return random.lognormvariate(math.log(median), shape)


def sample_noise_event_peak(event_type: str) -> float:
    params = NOISE_EVENT_PEAK_PARAMS[event_type]
    return random.gauss(params["mean"], params["stddev"])


def sample_noise_time_spread_days(event_type: str) -> float:
    params = NOISE_EVENT_TIME_SPREAD_PARAMS[event_type]
    return sample_lognormal_median_shape(params["median_days"], params["shape"])


def sample_noise_depth_spread_m() -> float:
    return sample_lognormal_median_shape(
        NOISE_EVENT_DEPTH_SPREAD_PARAMS["median_m"],
        NOISE_EVENT_DEPTH_SPREAD_PARAMS["shape"],
    )


def build_scalar_noise_events(event_type: str, noise_event_days: list[date]) -> dict[date, tuple[float, float]]:
    return {
        event_day: (
            sample_noise_event_peak(event_type),
            sample_noise_time_spread_days(event_type),
        )
        for event_day in noise_event_days
    }


def scalar_noise_at_time(
    measurement_time: datetime,
    noise_events: dict[date, tuple[float, float]],
) -> float:
    measurement_day = measurement_time.date()
    total_noise = 0.0
    for offset_days in range(-NOISE_EVENT_TIME_WINDOW_DAYS, NOISE_EVENT_TIME_WINDOW_DAYS + 1):
        event_day = measurement_day + timedelta(days=offset_days)
        event_data = noise_events.get(event_day)
        if event_data is None:
            continue
        peak_value, sigma_days = event_data
        total_noise += peak_value * gaussian_weight(-float(offset_days), sigma_days)
    return total_noise


def build_profile_noise_events(
    noise_event_days: list[date],
    child_depths: list[float],
) -> dict[date, dict[float, tuple[float, float, float, float]]]:
    profile_noise_events: dict[date, dict[float, tuple[float, float, float, float]]] = {}
    for event_day in noise_event_days:
        depth_events: dict[float, tuple[float, float, float, float]] = {}
        for depth_m in child_depths:
            depth_events[float(depth_m)] = (
                sample_noise_event_peak("subsurface_lateral_displacement"),
                sample_noise_time_spread_days("subsurface_lateral_displacement"),
                sample_noise_depth_spread_m(),
                random.uniform(0.0, 2.0 * math.pi),
            )
        profile_noise_events[event_day] = depth_events
    return profile_noise_events


def profile_noise_at_time_and_depth(
    measurement_time: datetime,
    depth_m: float,
    noise_events: dict[date, dict[float, tuple[float, float, float, float]]],
) -> tuple[float, float]:
    measurement_day = measurement_time.date()
    noise_e = 0.0
    noise_n = 0.0

    for offset_days in range(-NOISE_EVENT_TIME_WINDOW_DAYS, NOISE_EVENT_TIME_WINDOW_DAYS + 1):
        event_day = measurement_day + timedelta(days=offset_days)
        day_events = noise_events.get(event_day)
        if not day_events:
            continue

        time_weight_offset = -float(offset_days)
        for event_depth, (peak_value, sigma_days, sigma_depth_m, angle_rad) in day_events.items():
            time_weight = gaussian_weight(time_weight_offset, sigma_days)
            depth_weight = gaussian_weight(float(depth_m) - float(event_depth), sigma_depth_m)
            contribution = peak_value * time_weight * depth_weight
            noise_e += contribution * math.cos(angle_rad)
            noise_n += contribution * math.sin(angle_rad)

    return noise_e, noise_n


def event_is_active(event: Event, measurement_time: datetime) -> bool:
    if event.duration_days <= 0:
        return False
    event_end = event.start_time + timedelta(days=event.duration_days)
    return event.start_time <= measurement_time <= event_end


def scalar_event_contribution(event: Event, instrument: Instrument, measurement_time: datetime) -> float:
    if not event_is_active(event, measurement_time):
        return 0.0
    distance = instrument_distance_m(instrument, event)
    if event.radius_m <= 0 or distance > event.radius_m:
        return 0.0

    if event.type == "surface_settlement":
        magnitude = event.magnitude_mm
    elif event.type == "groundwater_level":
        magnitude = -event.magnitude_mm
    else:
        return 0.0

    time_fraction = (measurement_time - event.start_time).total_seconds() / (event.duration_days * 24 * 3600)
    distance_fraction = distance / event.radius_m

    distance_decay = kumaraswamy_decay(distance_fraction, 2, 5)
    delta_v_max = magnitude * distance_decay
    time_approach = kumaraswamy_cdf(time_fraction, 1, 3)
    return delta_v_max * time_approach


def profile_event_contribution(event: Event, instrument: Instrument, measurement_time: datetime, depth_m: float) -> tuple[float, float]:
    if not event_is_active(event, measurement_time):
        return 0.0, 0.0
    distance = instrument_distance_m(instrument, event)
    if event.radius_m <= 0 or distance > event.radius_m:
        return 0.0, 0.0

    theta_rad = math.radians(event.direction_deg or 0.0)
    depth_max = 30.0
    depth_fraction = max(0.0, min(1.0, depth_m / depth_max))
    x_value = -depth_fraction
    alpha_shape = 2.0
    beta_shape = 5.0
    depth_scale = alpha_shape * beta_shape * (x_value ** (alpha_shape - 1)) * ((1 - (x_value**alpha_shape)) ** (beta_shape - 1))

    magnitude_e = event.magnitude_mm * math.sin(theta_rad) * depth_scale
    magnitude_n = event.magnitude_mm * math.cos(theta_rad) * depth_scale

    time_fraction = (measurement_time - event.start_time).total_seconds() / (event.duration_days * 24 * 3600)
    distance_fraction = distance / event.radius_m
    decay_factor = kumaraswamy_decay(distance_fraction, 2, 5) * kumaraswamy_cdf(time_fraction, 1, 3)

    return magnitude_e * decay_factor, magnitude_n * decay_factor


def synthesise_timeseries_for_instruments(
    instruments: list[Instrument],
    events: list[Event],
    start_date: date,
    end_date: date,
    baseline_offsets: dict[str, float] | None = None,
    include_start_on_sunday: bool = True,
    logger: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    baseline_offsets = baseline_offsets or {}
    measurement_days = build_measurement_days(start_date, end_date, include_start_on_sunday=include_start_on_sunday)
    noise_event_days = build_noise_event_days(start_date, end_date)
    events_by_type = {
        event_type: [event for event in events if event.type == event_type]
        for event_type in EVENT_TYPES
    }

    if logger is not None:
        logger(
            "Preparing time-series synthesis "
            f"for {len(instruments)} instruments across {len(measurement_days)} measurement days "
            f"with {len(events)} events."
        )

    total_instruments = len(instruments)
    for instrument_index, instrument in enumerate(instruments, start=1):
        instrument.timeseries = []
        instrument_events = events_by_type.get(instrument.event_type, [])
        scalar_offset = float(baseline_offsets.get(instrument.instr_id, 0.0))

        if instrument.event_type in {"surface_settlement", "groundwater_level"}:
            scalar_noise_events = build_scalar_noise_events(instrument.event_type, noise_event_days)
            previous_value = 0.0
            for day in measurement_days:
                measurement_time = datetime.combine(day, time.min)
                contributions = [
                    scalar_event_contribution(event, instrument, measurement_time)
                    for event in instrument_events
                ]
                active_contributions = [value for value in contributions if abs(value) > 0]

                if active_contributions:
                    signal_value = float(sum(active_contributions))
                    previous_value = signal_value
                else:
                    if instrument.event_type == "groundwater_level":
                        previous_value = float(previous_value * 0.8)
                        signal_value = previous_value
                    else:
                        signal_value = previous_value

                if day == start_date:
                    signal_value = 0.0
                    previous_value = 0.0

                measurement_value = signal_value + scalar_noise_at_time(measurement_time, scalar_noise_events) + scalar_offset

                instrument.timeseries.append(
                    {
                        "time": measurement_time.isoformat(),
                        "value": measurement_value,
                    }
                )
        else:
            child_profiles = instrument.children or []
            child_depths = [float(depth_m) for _, depth_m in child_profiles]
            profile_noise_events = build_profile_noise_events(noise_event_days, child_depths)
            previous_profile_values: dict[float, tuple[float, float]] = {
                float(depth_m): (0.0, 0.0)
                for _, depth_m in child_profiles
            }
            profile_offset_by_depth = {
                float(depth_m): float(baseline_offsets.get(child_id, 0.0))
                for child_id, depth_m in child_profiles
            }
            for day in measurement_days:
                measurement_time = datetime.combine(day, time.min)
                profile: list[dict[str, float]] = []

                for child_id, depth_m in child_profiles:
                    depth_key = float(depth_m)
                    value_e = 0.0
                    value_n = 0.0
                    for event in instrument_events:
                        delta_e, delta_n = profile_event_contribution(event, instrument, measurement_time, depth_m)
                        value_e += delta_e
                        value_n += delta_n

                    if abs(value_e) > 0 or abs(value_n) > 0:
                        previous_profile_values[depth_key] = (float(value_e), float(value_n))
                    else:
                        value_e, value_n = previous_profile_values.get(depth_key, (0.0, 0.0))

                    if day == start_date:
                        value_e = 0.0
                        value_n = 0.0
                        previous_profile_values[depth_key] = (0.0, 0.0)

                    noise_e, noise_n = profile_noise_at_time_and_depth(
                        measurement_time,
                        depth_m,
                        profile_noise_events,
                    )
                    value_e += noise_e
                    value_n += noise_n
                    value_e, value_n = apply_profile_scalar_offset(
                        value_e,
                        value_n,
                        profile_offset_by_depth.get(depth_key, 0.0),
                    )

                    profile.append(
                        {
                            "depth": depth_key,
                            "value_e": float(value_e),
                            "value_n": float(value_n),
                        }
                    )

                instrument.timeseries.append(
                    {
                        "time": measurement_time.isoformat(),
                        "profile": profile,
                    }
                )

        if logger is not None:
            logger(
                f"Synthesised instrument {instrument_index}/{total_instruments}: "
                f"{instrument.instr_id} ({instrument.db_type}, {instrument.event_type}) with {len(instrument.timeseries)} entries."
            )
        if progress_callback is not None:
            progress_callback(
                instrument_index,
                total_instruments,
                f"Synthesised {instrument_index} of {total_instruments} instruments.",
            )

    if logger is not None:
        logger(f"Finished time-series synthesis for {total_instruments} instruments.")


def save_event_timeline_plot(events: list[Event], root: Path, timestamp: str, start_date: date, end_date: date) -> Path:
    figure = go.Figure()
    for event_type in EVENT_TYPES:
        subset = [event for event in events if event.type == event_type]
        if not subset:
            continue
        magnitude_scale = max_possible_magnitude(event_type)
        figure.add_trace(
            go.Bar(
                x=[event.start_time for event in subset],
                y=[event.magnitude_mm / magnitude_scale for event in subset],
                width=[event.duration_days * 24 * 60 * 60 * 1000 for event in subset],
                name=event_type,
                marker_color=TYPE_COLORS[event_type],
                hovertemplate="%{customdata}<extra></extra>",
                customdata=[event_hover_text(event) for event in subset],
            )
        )
    figure.update_layout(
        barmode="overlay",
        title="Event timeline",
        xaxis_title="Start time",
        yaxis_title="Normalized magnitude",
        xaxis={
            "range": [
                datetime.combine(start_date, time.min).isoformat(),
                datetime.combine(end_date + timedelta(days=1), time.min).isoformat(),
            ]
        },
    )
    plot_dir = root / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"event_timeline_{timestamp}.html"
    figure.write_html(str(path), include_plotlyjs="cdn")
    return path


def circle_geojson(latitude: float, longitude: float, radius_m: float, points: int = 48) -> dict:
    earth_radius = 6378137.0
    lat_rad = __import__("math").radians(latitude)
    coords = []
    for idx in range(points + 1):
        theta = 2 * __import__("math").pi * idx / points
        delta_lat = (radius_m / earth_radius) * __import__("math").sin(theta)
        delta_lon = (radius_m / (earth_radius * __import__("math").cos(lat_rad))) * __import__("math").cos(theta)
        coords.append(
            [longitude + __import__("math").degrees(delta_lon), latitude + __import__("math").degrees(delta_lat)]
        )
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {},
            }
        ],
    }


def save_event_spatial_plot(
    events: list[Event],
    root: Path,
    timestamp: str,
    highlighted_instruments: list[Instrument] | None = None,
    show_event_epicentres: bool = True,
    highlighted_marker_size: int = 14,
) -> Path:
    figure = go.Figure()

    if show_event_epicentres:
        for event_type in EVENT_TYPES:
            subset = [event for event in events if event.type == event_type]
            if not subset:
                continue
            figure.add_trace(
                go.Scattermapbox(
                    lat=[event.epicentre.latitude for event in subset],
                    lon=[event.epicentre.longitude for event in subset],
                    mode="markers",
                    marker={"size": 10, "color": TYPE_COLORS[event_type]},
                    name=event_type,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=[event_hover_text(event) for event in subset],
                )
            )

    layers = []
    for event in events:
        layers.append(
            {
                "sourcetype": "geojson",
                "source": circle_geojson(event.epicentre.latitude, event.epicentre.longitude, event.radius_m),
                "type": "fill",
                "color": TYPE_COLORS[event.type],
                "opacity": 0.35,
            }
        )

    quiver_events = [event for event in events if event.type == "subsurface_lateral_displacement"]
    for event in quiver_events:
        if event.direction_deg is None:
            continue
        angle = __import__("math").radians(event.direction_deg)
        length_m = max(event.magnitude_mm, 1.0)
        lat0 = event.epicentre.latitude
        lon0 = event.epicentre.longitude
        earth_radius = 6378137.0
        dlat = (length_m / earth_radius) * __import__("math").sin(angle)
        dlon = (length_m / (earth_radius * __import__("math").cos(__import__("math").radians(lat0)))) * __import__("math").cos(angle)
        lat1 = lat0 + __import__("math").degrees(dlat)
        lon1 = lon0 + __import__("math").degrees(dlon)
        figure.add_trace(
            go.Scattermapbox(
                lat=[lat0, lat1],
                lon=[lon0, lon1],
                mode="lines",
                line={"width": 3, "color": TYPE_COLORS[event.type]},
                name="subsurface_lateral_displacement_vector",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=[event_hover_text(event), event_hover_text(event)],
                showlegend=False,
            )
        )

    if highlighted_instruments:
        for event_type in EVENT_TYPES:
            subset = [
                instrument
                for instrument in highlighted_instruments
                if instrument.event_type == event_type
                and math.isfinite(float(instrument.latitude))
                and math.isfinite(float(instrument.longitude))
            ]
            if not subset:
                continue
            figure.add_trace(
                go.Scattermapbox(
                    lat=[instrument.latitude for instrument in subset],
                    lon=[instrument.longitude for instrument in subset],
                    mode="markers",
                    marker={
                        "size": highlighted_marker_size,
                        "color": TYPE_COLORS[event_type],
                        "opacity": 1.0,
                    },
                    name=f"top_{event_type}_instruments",
                    text=[instrument.instr_id for instrument in subset],
                    hovertemplate="instr_id=%{text}<br>event_type=" + event_type + "<extra></extra>",
                )
            )

    center_lat = sum(event.epicentre.latitude for event in events) / len(events) if events else 21.0
    center_lon = sum(event.epicentre.longitude for event in events) / len(events) if events else 105.8
    figure.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_zoom=15,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_layers=layers,
        title="Event spatial distribution",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    plot_dir = root / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"event_spatial_distribution_{timestamp}.html"
    figure.write_html(str(path), include_plotlyjs="cdn")
    return path


def instrument_peak_magnitude(instrument: Instrument) -> float:
    if not instrument.timeseries:
        return 0.0
    if instrument.event_type in {"surface_settlement", "groundwater_level"}:
        return max((abs(float(point.get("value", 0.0))) for point in instrument.timeseries), default=0.0)

    peak = 0.0
    for measurement in instrument.timeseries:
        for profile_point in measurement.get("profile", []):
            value_e = float(profile_point.get("value_e", 0.0))
            value_n = float(profile_point.get("value_n", 0.0))
            peak = max(peak, math.hypot(value_e, value_n))
    return peak


def subsurface_peak_context(instrument: Instrument) -> tuple[str, float, float] | None:
    best_time = ""
    best_depth = 0.0
    best_value = -1.0
    for measurement in instrument.timeseries:
        measurement_time = str(measurement.get("time", ""))
        for profile_point in measurement.get("profile", []):
            depth = float(profile_point.get("depth", 0.0))
            value_e = float(profile_point.get("value_e", 0.0))
            value_n = float(profile_point.get("value_n", 0.0))
            magnitude = math.hypot(value_e, value_n)
            if magnitude > best_value:
                best_value = magnitude
                best_time = measurement_time
                best_depth = depth
    if best_value < 0:
        return None
    return best_time, best_depth, best_value


def select_top_instruments(instruments: list[Instrument], top_n: int = 3) -> dict[str, list[Instrument]]:
    selected: dict[str, list[Instrument]] = {}
    for event_type in EVENT_TYPES:
        candidates = [instrument for instrument in instruments if instrument.event_type == event_type]
        selected[event_type] = sorted(candidates, key=instrument_peak_magnitude, reverse=True)[:top_n]
    return selected


def save_top_timeseries_plot(root: Path, timestamp: str, top_by_type: dict[str, list[Instrument]]) -> Path:
    figure = make_subplots(rows=3, cols=1, subplot_titles=EVENT_TYPES, vertical_spacing=0.08)
    line_dashes = ["solid", "dash", "dot"]

    for row_index, event_type in enumerate(EVENT_TYPES, start=1):
        for instrument_index, instrument in enumerate(top_by_type.get(event_type, [])):
            dash = line_dashes[instrument_index % len(line_dashes)]
            if event_type in {"surface_settlement", "groundwater_level"}:
                x_values = [point.get("time") for point in instrument.timeseries]
                y_values = [float(point.get("value", 0.0)) for point in instrument.timeseries]
            else:
                context = subsurface_peak_context(instrument)
                if context is None:
                    continue
                _, peak_depth, _ = context
                x_values = []
                y_values = []
                for measurement in instrument.timeseries:
                    x_values.append(measurement.get("time"))
                    profile = measurement.get("profile", [])
                    value = 0.0
                    for profile_point in profile:
                        if abs(float(profile_point.get("depth", 0.0)) - peak_depth) < 1e-9:
                            value = math.hypot(
                                float(profile_point.get("value_e", 0.0)),
                                float(profile_point.get("value_n", 0.0)),
                            )
                            break
                    y_values.append(value)

            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=f"{event_type}: {instrument.instr_id}",
                    line={"color": TYPE_COLORS[event_type], "dash": dash},
                ),
                row=row_index,
                col=1,
            )

    figure.update_layout(height=1100, title="Top 3 instruments by event_type: time series")
    plot_dir = root / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"timeseries_top3_by_type_{timestamp}.html"
    figure.write_html(str(path), include_plotlyjs="cdn")
    return path


def save_subsurface_depth_profile_plot(root: Path, timestamp: str, instruments: list[Instrument]) -> Path:
    figure = go.Figure()
    line_dashes = ["solid", "dot", "dash", "dashdot"]

    for index, instrument in enumerate(instruments):
        context = subsurface_peak_context(instrument)
        if context is None:
            continue
        peak_time, _, _ = context
        measurement = next((point for point in instrument.timeseries if point.get("time") == peak_time), None)
        if measurement is None:
            continue
        profile = sorted(measurement.get("profile", []), key=lambda item: float(item.get("depth", 0.0)))
        depths = [float(point.get("depth", 0.0)) for point in profile]
        values_e = [float(point.get("value_e", 0.0)) for point in profile]
        values_n = [float(point.get("value_n", 0.0)) for point in profile]

        dash = line_dashes[index % len(line_dashes)]
        figure.add_trace(
            go.Scatter(
                x=values_e,
                y=depths,
                mode="lines",
                name=f"{instrument.instr_id} value_e",
                line={"color": TYPE_COLORS["subsurface_lateral_displacement"], "dash": dash, "width": 2},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=values_n,
                y=depths,
                mode="lines",
                name=f"{instrument.instr_id} value_n",
                line={"color": TYPE_COLORS["subsurface_lateral_displacement"], "dash": dash, "width": 4},
            )
        )

    figure.update_layout(
        title="Subsurface displacement depth profiles at peak measurement time",
        xaxis_title="Value",
        yaxis_title="Depth (m)",
        yaxis={"autorange": "reversed"},
        height=700,
    )
    plot_dir = root / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"subsurface_depth_profiles_{timestamp}.html"
    figure.write_html(str(path), include_plotlyjs="cdn")
    return path


def collect_measurement_times(instruments: list[Instrument]) -> list[datetime]:
    times: set[datetime] = set()
    for instrument in instruments:
        for point in instrument.timeseries:
            raw_time = point.get("time")
            if not raw_time:
                continue
            try:
                times.add(datetime.fromisoformat(str(raw_time)))
            except ValueError:
                continue
    return sorted(times)


def select_time_slices(measurement_times: list[datetime]) -> list[tuple[str, datetime]]:
    if not measurement_times:
        return []

    start_time = measurement_times[0]
    end_time = measurement_times[-1]
    total_seconds = max((end_time - start_time).total_seconds(), 0.0)
    fractions = [("25pct", 0.25), ("50pct", 0.5), ("75pct", 0.75), ("100pct", 1.0)]

    slices: list[tuple[str, datetime]] = []
    for label, fraction in fractions:
        target = start_time + timedelta(seconds=total_seconds * fraction)
        nearest = min(measurement_times, key=lambda value: abs((value - target).total_seconds()))
        slices.append((label, nearest))

    return slices


def instrument_scalar_value_at_time(instrument: Instrument, measurement_time: datetime) -> float:
    for point in instrument.timeseries:
        try:
            point_time = datetime.fromisoformat(str(point.get("time", "")))
        except ValueError:
            continue
        if point_time == measurement_time:
            return float(point.get("value", 0.0))
    return 0.0


def instrument_peak_profile_vector_at_time(instrument: Instrument, measurement_time: datetime) -> tuple[float, float, float, float] | None:
    for point in instrument.timeseries:
        try:
            point_time = datetime.fromisoformat(str(point.get("time", "")))
        except ValueError:
            continue
        if point_time != measurement_time:
            continue

        best_depth = 0.0
        best_e = 0.0
        best_n = 0.0
        best_mag = -1.0

        for profile_point in point.get("profile", []):
            value_e = float(profile_point.get("value_e", 0.0))
            value_n = float(profile_point.get("value_n", 0.0))
            magnitude = math.hypot(value_e, value_n)
            if magnitude > best_mag:
                best_mag = magnitude
                best_depth = float(profile_point.get("depth", 0.0))
                best_e = value_e
                best_n = value_n

        if best_mag < 0:
            return None
        return best_depth, best_e, best_n, best_mag

    return None


def save_scalar_timeslice_map_plot(
    root: Path,
    timestamp: str,
    event_type: str,
    slice_label: str,
    measurement_time: datetime,
    events: list[Event],
    instruments: list[Instrument],
) -> Path:
    subset = [instrument for instrument in instruments if instrument.event_type == event_type]
    plot_dir = root / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure = go.Figure()

    rows = [
        (
            instrument.latitude,
            instrument.longitude,
            instrument.instr_id,
            instrument_scalar_value_at_time(instrument, measurement_time),
        )
        for instrument in subset
    ]

    if event_type == "groundwater_level":
        rows = [(lat, lon, instr_id, min(value, 0.0)) for lat, lon, instr_id, value in rows]
        rows.sort(key=lambda item: item[3], reverse=True)
        values = [row[3] for row in rows]
        latitudes = [row[0] for row in rows]
        longitudes = [row[1] for row in rows]
        instr_ids = [row[2] for row in rows]
        cmin = min(values) if values else -1.0
        if cmin == 0.0:
            cmin = -1.0
        cmax = 0.0
        colorscale = [[0.0, "#1c7ed6"], [1.0, "#ffffff"]]
        colorbar_title = "Groundwater (mm)"
    else:
        rows.sort(key=lambda item: item[3], reverse=True)
        values = [row[3] for row in rows]
        latitudes = [row[0] for row in rows]
        longitudes = [row[1] for row in rows]
        instr_ids = [row[2] for row in rows]
        cmin = min(values) if values else -1.0
        cmax = max(values) if values else 1.0
        if cmin == cmax:
            if cmin == 0.0:
                cmin, cmax = -1.0, 1.0
            elif cmin < 0:
                cmax = 0.0
            else:
                cmin = 0.0

        if cmin < 0 < cmax:
            zero_pos = (0.0 - cmin) / (cmax - cmin)
            colorscale = [[0.0, "#ff0000"], [zero_pos, "#ffffff"], [1.0, "#8000ff"]]
        elif cmax <= 0:
            colorscale = [[0.0, "#ff0000"], [1.0, "#ffffff"]]
        else:
            colorscale = [[0.0, "#ffffff"], [1.0, "#8000ff"]]
        colorbar_title = "Settlement (mm)"

    layers = []
    if event_type == "groundwater_level":
        relevant_events = [
            event
            for event in events
            if event.type == event_type and event_is_active(event, measurement_time)
        ]
    else:
        relevant_events = [
            event
            for event in events
            if event.type == event_type and event.start_time <= measurement_time
        ]
    for event in relevant_events:
        layers.append(
            {
                "sourcetype": "geojson",
                "source": circle_geojson(event.epicentre.latitude, event.epicentre.longitude, event.radius_m),
                "type": "fill",
                "color": TYPE_COLORS[event.type],
                "opacity": 0.35,
            }
        )

    figure.add_trace(
        go.Scattermapbox(
            lat=latitudes,
            lon=longitudes,
            mode="markers",
            marker={
                "size": 7,
                "color": values,
                "colorscale": colorscale,
                "cmin": cmin,
                "cmax": cmax,
                "showscale": True,
                "colorbar": {"title": colorbar_title},
            },
            name=event_type,
            text=instr_ids,
            customdata=[[value] for value in values],
            hovertemplate="instr_id=%{text}<br>value_mm=%{customdata[0]:.4f}<extra></extra>",
        )
    )

    center_lat = sum(latitudes) / len(latitudes) if latitudes else 21.0
    center_lon = sum(longitudes) / len(longitudes) if longitudes else 105.8
    figure.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_zoom=15,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_layers=layers,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title=f"{event_type} at {slice_label} ({measurement_time.isoformat()})",
        legend_title_text="Series",
    )

    path = plot_dir / f"timeslice_{event_type}_{slice_label}_{timestamp}.html"
    figure.write_html(str(path), include_plotlyjs="cdn")
    return path


def save_subsurface_timeslice_vector_map_plot(
    root: Path,
    timestamp: str,
    slice_label: str,
    measurement_time: datetime,
    events: list[Event],
    instruments: list[Instrument],
) -> Path:
    subset = [instrument for instrument in instruments if instrument.event_type == "subsurface_lateral_displacement"]
    plot_dir = root / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure = go.Figure()
    layers = []
    relevant_events = [
        event
        for event in events
        if event.type == "subsurface_lateral_displacement" and event.start_time <= measurement_time
    ]
    for event in relevant_events:
        layers.append(
            {
                "sourcetype": "geojson",
                "source": circle_geojson(event.epicentre.latitude, event.epicentre.longitude, event.radius_m),
                "type": "fill",
                "color": TYPE_COLORS[event.type],
                "opacity": 0.35,
            }
        )

    figure.add_trace(
        go.Scattermapbox(
            lat=[instrument.latitude for instrument in subset],
            lon=[instrument.longitude for instrument in subset],
            mode="markers",
            marker={"size": 7, "color": "#bbbbbb"},
            name="instrument_location",
            text=[instrument.instr_id for instrument in subset],
            hovertemplate="instr_id=%{text}<extra></extra>",
        )
    )

    vectors: list[tuple[Instrument, float, float, float, float]] = []
    for instrument in subset:
        peak = instrument_peak_profile_vector_at_time(instrument, measurement_time)
        if peak is None:
            continue
        depth, value_e, value_n, magnitude = peak
        vectors.append((instrument, depth, value_e, value_n, magnitude))

    max_magnitude = max((vector[4] for vector in vectors), default=0.0)
    earth_radius = 6378137.0
    max_arrow_length_m = 50.0

    for index, (instrument, depth, value_e, value_n, magnitude) in enumerate(vectors):
        if magnitude <= 0 or max_magnitude <= 0:
            continue

        length_m = (magnitude / max_magnitude) * max_arrow_length_m
        unit_e = value_e / magnitude
        unit_n = value_n / magnitude

        lat0 = instrument.latitude
        lon0 = instrument.longitude
        lat0_rad = math.radians(lat0)
        dlat = (length_m / earth_radius) * unit_n
        dlon = (length_m / (earth_radius * max(math.cos(lat0_rad), 1e-9))) * unit_e

        lat1 = lat0 + math.degrees(dlat)
        lon1 = lon0 + math.degrees(dlon)

        figure.add_trace(
            go.Scattermapbox(
                lat=[lat0, lat1],
                lon=[lon0, lon1],
                mode="lines+markers",
                marker={"size": 7, "color": "#37b24d"},
                line={"width": 3, "color": "#37b24d"},
                name="peak_vector",
                showlegend=index == 0,
                text=[instrument.instr_id, instrument.instr_id],
                customdata=[
                    [depth, value_e, value_n, magnitude],
                    [depth, value_e, value_n, magnitude],
                ],
                hovertemplate=(
                    "instr_id=%{text}<br>depth_m=%{customdata[0]:.2f}<br>"
                    "value_e=%{customdata[1]:.4f}<br>value_n=%{customdata[2]:.4f}<br>"
                    "magnitude=%{customdata[3]:.4f}<extra></extra>"
                ),
            )
        )

    latitudes = [instrument.latitude for instrument in subset]
    longitudes = [instrument.longitude for instrument in subset]
    center_lat = sum(latitudes) / len(latitudes) if latitudes else 21.0
    center_lon = sum(longitudes) / len(longitudes) if longitudes else 105.8
    figure.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_zoom=15,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_layers=layers,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title=f"subsurface_lateral_displacement at {slice_label} ({measurement_time.isoformat()})",
        legend_title_text="Series",
    )

    path = plot_dir / f"timeslice_subsurface_lateral_displacement_{slice_label}_{timestamp}.html"
    figure.write_html(str(path), include_plotlyjs="cdn")
    return path


def save_timeslice_validation_plots(root: Path, timestamp: str, instruments: list[Instrument], events: list[Event]) -> list[Path]:
    measurement_times = collect_measurement_times(instruments)
    time_slices = select_time_slices(measurement_times)
    if not time_slices:
        return []

    paths: list[Path] = []
    for slice_label, measurement_time in time_slices:
        paths.append(
            save_scalar_timeslice_map_plot(
                root,
                timestamp,
                "surface_settlement",
                slice_label,
                measurement_time,
                events,
                instruments,
            )
        )
        paths.append(
            save_scalar_timeslice_map_plot(
                root,
                timestamp,
                "groundwater_level",
                slice_label,
                measurement_time,
                events,
                instruments,
            )
        )
        paths.append(
            save_subsurface_timeslice_vector_map_plot(
                root,
                timestamp,
                slice_label,
                measurement_time,
                events,
                instruments,
            )
        )

    return paths


def render_html_plot(path: Path, height: int = 720) -> None:
    if path.exists():
        components.html(path.read_text(encoding="utf-8"), height=height, scrolling=True)


def parse_iso_datetime(raw_time: Any) -> datetime | None:
    if raw_time is None:
        return None
    if isinstance(raw_time, datetime):
        return raw_time
    try:
        return datetime.fromisoformat(str(raw_time))
    except ValueError:
        return None


def format_db_datetime(raw_time: Any) -> str:
    parsed = parse_iso_datetime(raw_time)
    if parsed is None:
        return ""
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_field_value(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def apply_profile_scalar_offset(value_e: float, value_n: float, offset: float) -> tuple[float, float]:
    if abs(offset) <= 1e-12:
        return value_e, value_n

    magnitude = math.hypot(value_e, value_n)
    if magnitude <= 1e-12:
        return 0.0, float(offset)

    scaled_magnitude = max(magnitude + offset, 0.0)
    scale = scaled_magnitude / magnitude
    return value_e * scale, value_n * scale


def has_valid_timeseries(instruments: list[Instrument]) -> bool:
    for instrument in instruments:
        if not instrument.timeseries:
            continue
        for point in instrument.timeseries:
            if not isinstance(point, dict):
                continue
            if parse_iso_datetime(point.get("time")) is None:
                continue

            if instrument.db_type in {"LP", "TLP", "CASA", "PZ"}:
                if "value" in point and coerce_float(point.get("value")) is not None:
                    return True
            else:
                profile = point.get("profile")
                if not isinstance(profile, list):
                    continue
                for profile_item in profile:
                    if not isinstance(profile_item, dict):
                        continue
                    if coerce_float(profile_item.get("depth")) is None:
                        continue
                    if coerce_float(profile_item.get("value_e")) is None and coerce_float(profile_item.get("value_n")) is None:
                        continue
                    return True
    return False


def get_profile_item_for_depth(profile: list[dict[str, Any]], depth_m: float, tolerance: float = 1e-9) -> dict[str, Any] | None:
    for item in profile:
        if not isinstance(item, dict):
            continue
        depth = coerce_float(item.get("depth"))
        if depth is None:
            continue
        if abs(depth - float(depth_m)) <= tolerance:
            return item
    return None


def next_greater_child_depth(children: list[tuple[str, float]], depth_m: float) -> float | None:
    sorted_depths = sorted(float(child_depth) for _, child_depth in children)
    for candidate in sorted_depths:
        if candidate > float(depth_m):
            return candidate
    return None


def build_assimilation_dataframe(
    instruments: list[Instrument],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    total_instruments = len(instruments)

    for instrument_index, instrument in enumerate(instruments, start=1):
        db_type = instrument.db_type
        if db_type in {"LP", "TLP", "CASA", "PZ"}:
            for point in instrument.timeseries:
                timestamp = format_db_datetime(point.get("time"))
                value = coerce_float(point.get("value"))

                data1_value: float | None = None
                calc_values: dict[str, float | None] = {}

                if db_type == "LP":
                    data1_value = value
                    calc_values = {
                        "calculation1": value,
                        "calculation4": value,
                        "calculation6": value,
                    }
                elif db_type == "TLP":
                    data1_value = value
                    calc_values = {
                        "calculation1": value,
                        "calculation2": value,
                        "calculation3": value,
                    }
                elif db_type in {"CASA", "PZ"}:
                    if value is None:
                        data1_value = None
                        calc_values = {"calculation1": None}
                    else:
                        data1_value = -0.001 * value
                        calc_values = {"calculation1": 0.001 * value}

                custom_fields = {
                    field_name: format_field_value(field_value)
                    for field_name, field_value in calc_values.items()
                }

                rows.append(
                    {
                        "instr_id": instrument.instr_id,
                        "date1": timestamp,
                        "data1": format_field_value(data1_value),
                        "custom_fields": json.dumps(custom_fields, ensure_ascii=False),
                    }
                )
        elif db_type in {"INCL", "EI"}:
            children = instrument.children or []
            child_depth_map = {child_id: float(depth_m) for child_id, depth_m in children}

            for point in instrument.timeseries:
                timestamp = format_db_datetime(point.get("time"))
                profile = point.get("profile", [])
                if not isinstance(profile, list):
                    profile = []

                child_calc3_by_depth: dict[float, float | None] = {}
                for _, depth_m in children:
                    profile_item = get_profile_item_for_depth(profile, float(depth_m))
                    if profile_item is None:
                        child_calc3_by_depth[float(depth_m)] = None
                        continue
                    value_e = coerce_float(profile_item.get("value_e"))
                    value_n = coerce_float(profile_item.get("value_n"))
                    if value_e is None or value_n is None:
                        child_calc3_by_depth[float(depth_m)] = None
                        continue
                    child_calc3_by_depth[float(depth_m)] = math.hypot(value_n, value_e)

                for child_id, depth_m in children:
                    depth_value = child_depth_map.get(child_id)
                    if depth_value is None:
                        continue
                    calc3_value = child_calc3_by_depth.get(depth_value)

                    data1_value: float | None = None
                    calc_values: dict[str, float | None] = {}

                    if db_type == "INCL":
                        next_depth = next_greater_child_depth(children, depth_value)
                        next_calc3 = child_calc3_by_depth.get(next_depth) if next_depth is not None else None
                        if calc3_value is not None and next_calc3 is not None:
                            data1_value = calc3_value - next_calc3
                        else:
                            data1_value = None
                        calc_values = {
                            "calculation1": data1_value,
                            "calculation3": calc3_value,
                        }
                    else:
                        data1_value = None
                        calc_values = {
                            "calculation2": calc3_value,
                        }

                    custom_fields = {
                        field_name: format_field_value(field_value)
                        for field_name, field_value in calc_values.items()
                    }

                    rows.append(
                        {
                            "instr_id": child_id,
                            "date1": timestamp,
                            "data1": format_field_value(data1_value),
                            "custom_fields": json.dumps(custom_fields, ensure_ascii=False),
                        }
                    )

                zero_depth_child_calc3: float | None = None
                for child_id, depth_m in children:
                    if abs(float(depth_m)) > 1e-9:
                        continue
                    zero_depth_child_calc3 = child_calc3_by_depth.get(float(depth_m))
                    if zero_depth_child_calc3 is not None:
                        break

                master_custom_fields = {
                    "calculation5": format_field_value(zero_depth_child_calc3),
                }
                rows.append(
                    {
                        "instr_id": instrument.instr_id,
                        "date1": timestamp,
                        "data1": "",
                        "custom_fields": json.dumps(master_custom_fields, ensure_ascii=False),
                    }
                )

        if progress_callback is not None:
            progress_callback(
                instrument_index,
                total_instruments,
                f"Assimilated {instrument_index} of {total_instruments} instruments into mydata rows.",
            )

    return pd.DataFrame(rows, columns=["instr_id", "date1", "data1", "custom_fields"])


def save_mydata_json(root: Path, dataframe: pd.DataFrame) -> Path:
    validation_data_dir = root / "validation_data"
    validation_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = validation_data_dir / "mydata.json"

    records = dataframe.to_dict(orient="records")
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    return output_path


def load_or_synthesise_valid_instruments(
    root: Path,
    start_date: date,
    end_date: date,
    events: list[Event],
    start_mode: str = TIME_SERIES_START_MODE_ZERO,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[Instrument], list[Event]]:
    in_memory_instruments = st.session_state.get("instruments", [])
    instruments: list[Instrument] = in_memory_instruments if has_valid_timeseries(in_memory_instruments) else []

    if progress_callback is not None:
        progress_callback(0.05, "Checking for existing valid time-series data.")

    if not instruments:
        from_json = load_instruments_from_json(root)
        if has_valid_timeseries(from_json):
            instruments = from_json

    if instruments:
        if progress_callback is not None:
            progress_callback(1.0, f"Using {len(instruments)} instruments with existing valid time-series data.")
        return instruments, events

    instruments = load_instruments_from_json(root)
    if not instruments:
        instruments, _, _ = extract_instruments(root)
        instruments = load_instruments_from_json(root)
    if progress_callback is not None:
        progress_callback(0.2, f"Loaded {len(instruments)} instruments for assimilation.")

    if not events:
        events = load_events_from_json(root)
    if events and progress_callback is not None:
        progress_callback(0.35, f"Loaded {len(events)} events for assimilation.")

    if not events:
        transformer = get_db_transformer(root)
        if progress_callback is not None:
            progress_callback(0.3, "Generating event history required for assimilation.")
        events = generate_events(
            instruments,
            start_date,
            end_date,
            transformer,
            progress_callback=(
                None
                if progress_callback is None
                else lambda day_index, total_days, message: progress_callback(
                    0.3 + (0.2 * (day_index / total_days if total_days else 1.0)),
                    message,
                )
            ),
        )
        save_events_to_json(root, events)
        if progress_callback is not None:
            progress_callback(0.5, f"Generated and saved {len(events)} events for assimilation.")

    baseline_offsets: dict[str, float] = {}
    if start_mode == TIME_SERIES_START_MODE_SNAPSHOT:
        try:
            baseline_offsets = fetch_snapshot_offsets(root, instruments, start_date)
        except Exception:
            baseline_offsets = {}
    if progress_callback is not None:
        progress_callback(
            0.55,
            (
                f"Prepared {len(baseline_offsets)} snapshot offsets for assimilation."
                if start_mode == TIME_SERIES_START_MODE_SNAPSHOT
                else "Prepared zero-baseline mode for assimilation."
            ),
        )

    synthesise_timeseries_for_instruments(
        instruments,
        events,
        start_date,
        end_date,
        baseline_offsets=baseline_offsets,
        include_start_on_sunday=(start_mode == TIME_SERIES_START_MODE_ZERO),
        progress_callback=(
            None
            if progress_callback is None
            else lambda instrument_index, total_instruments, message: progress_callback(
                0.55 + (0.35 * (instrument_index / total_instruments if total_instruments else 1.0)),
                message,
            )
        ),
    )
    save_instruments_to_json(root, instruments)
    if progress_callback is not None:
        progress_callback(1.0, f"Saved {len(instruments)} instruments with valid time-series data.")

    return instruments, events


st.title("Hanoi AI Demo Data Synthesis")

if "instruments" not in st.session_state:
    st.session_state["instruments"] = []
if "events" not in st.session_state:
    st.session_state["events"] = []
if "instrument_plot_path" not in st.session_state:
    st.session_state["instrument_plot_path"] = ""
if "synthesis_plot_paths" not in st.session_state:
    st.session_state["synthesis_plot_paths"] = []
if "generate_plot_paths" not in st.session_state:
    st.session_state["generate_plot_paths"] = []
if "assimilated_preview" not in st.session_state:
    st.session_state["assimilated_preview"] = []
if "assimilated_total_rows" not in st.session_state:
    st.session_state["assimilated_total_rows"] = 0
if "assimilated_output_path" not in st.session_state:
    st.session_state["assimilated_output_path"] = ""
if "mydata_df_records" not in st.session_state:
    st.session_state["mydata_df_records"] = []
if "generate_log_state" not in st.session_state:
    st.session_state["generate_log_state"] = create_form_log_state()
if "synthesise_log_state" not in st.session_state:
    st.session_state["synthesise_log_state"] = create_form_log_state()
if "assimilate_log_state" not in st.session_state:
    st.session_state["assimilate_log_state"] = create_form_log_state()

root = Path(__file__).resolve().parent
snapshot_min_date, snapshot_max_date, snapshot_bounds_error = fetch_synthetic_data_date_bounds(root)
snapshot_available = snapshot_min_date is not None and snapshot_max_date is not None
default_start_date = six_months_before(date.today())

with st.form("extract_form"):
    st.subheader("Extract instrument locations")
    extract_clicked = st.form_submit_button("Extract")
    if extract_clicked:
        instruments, _, plot_path = extract_instruments(root)
        st.session_state["instruments"] = instruments
        if plot_path is not None:
            st.session_state["instrument_plot_path"] = str(plot_path)

    instrument_plot_path = st.session_state.get("instrument_plot_path", "")
    if instrument_plot_path:
        render_html_plot(Path(instrument_plot_path))

# Keep interdependent controls out of st.form so Streamlit can rerun immediately
# when users toggle the start mode or Today checkbox.
if not snapshot_available:
    st.session_state["time_series_start_mode"] = TIME_SERIES_START_MODE_ZERO

start_mode_options = [TIME_SERIES_START_MODE_ZERO, TIME_SERIES_START_MODE_SNAPSHOT]

current_start_mode = st.session_state.get("time_series_start_mode", TIME_SERIES_START_MODE_ZERO)
if current_start_mode not in start_mode_options or (not snapshot_available and current_start_mode == TIME_SERIES_START_MODE_SNAPSHOT):
    current_start_mode = TIME_SERIES_START_MODE_ZERO

start_mode = st.radio(
    "Time series start from:",
    options=start_mode_options,
    index=start_mode_options.index(current_start_mode),
    key="time_series_start_mode",
    disabled=not snapshot_available,
)
if snapshot_bounds_error:
    st.caption(f"Snapshot start is unavailable because the target database could not be queried: {snapshot_bounds_error}")
elif not snapshot_available:
    st.caption("Snapshot of existing synthetic data is unavailable because mydata and futuredata do not contain any rows.")

if start_mode == TIME_SERIES_START_MODE_SNAPSHOT and snapshot_available:
    snapshot_default_date = st.session_state.get("snapshot_start_date", snapshot_min_date)
    if not isinstance(snapshot_default_date, date):
        snapshot_default_date = snapshot_min_date
    snapshot_default_date = min(max(snapshot_default_date, snapshot_min_date), snapshot_max_date)
    start_date = st.date_input(
        "Start date",
        value=snapshot_default_date,
        min_value=snapshot_min_date,
        max_value=snapshot_max_date,
        key="snapshot_start_date",
    )
else:
    zero_default_date = st.session_state.get("zero_start_date", default_start_date)
    if not isinstance(zero_default_date, date):
        zero_default_date = default_start_date
    start_date = st.date_input(
        "Start date",
        value=zero_default_date,
        key="zero_start_date",
    )
st.markdown("End date")
use_today = st.checkbox("Today", value=True, key="use_today")
end_date = st.date_input(
    "End date",
    value=st.session_state.get("selected_end_date", date.today()),
    disabled=use_today,
    label_visibility="collapsed",
    key="selected_end_date",
)
with st.form("generate_event_history_form"):
    st.subheader("Generate event history")
    generate_current_state = get_form_log_state_snapshot("generate_log_state")
    generate_progress_bar = st.progress(max(0.0, min(1.0, float(generate_current_state.get("percent_complete", 0.0) or 0.0) / 100.0)))
    generate_progress_caption = st.empty()
    render_form_progress_widgets(generate_progress_bar, generate_progress_caption, generate_current_state)
    generate_clicked = st.form_submit_button("Generate")

    if generate_clicked:
        final_end_date = date.today() if use_today else end_date
        total_generation_days = max(0, (final_end_date - start_date).days + 1)
        total_progress_units = total_generation_days + 6
        completed_units = 0
        current_state = set_form_progress(
            "generate_log_state",
            completed_units,
            total_progress_units,
            message="Generating event history.",
            status="running",
        )
        render_form_progress_widgets(generate_progress_bar, generate_progress_caption, current_state)
        append_form_log(root, "generate_stream.log", "generate_log_state", "Generate button clicked by user.")
        try:
            instruments = load_instruments_from_json(root)
            if instruments:
                append_form_log(
                    root,
                    "generate_stream.log",
                    "generate_log_state",
                    f"Loaded {len(instruments)} instruments from validation_data/instruments.json.",
                )
            else:
                append_form_log(
                    root,
                    "generate_stream.log",
                    "generate_log_state",
                    "No saved instruments found; extracting instruments from the database.",
                )
                instruments, _, _ = extract_instruments(root)
                instruments = load_instruments_from_json(root)
                append_form_log(
                    root,
                    "generate_stream.log",
                    "generate_log_state",
                    f"Extraction completed and loaded {len(instruments)} instruments.",
                )
            st.session_state["instruments"] = instruments
            completed_units += 1
            current_state = set_form_progress(
                "generate_log_state",
                completed_units,
                total_progress_units,
                message=f"Loaded {len(instruments)} instruments.",
            )
            render_form_progress_widgets(generate_progress_bar, generate_progress_caption, current_state)

            append_form_log(
                root,
                "generate_stream.log",
                "generate_log_state",
                f"Resolved generation window: start_date={start_date.isoformat()}, end_date={final_end_date.isoformat()}, start_mode={start_mode}.",
            )
            transformer = get_db_transformer(root)
            append_form_log(root, "generate_stream.log", "generate_log_state", "Loaded coordinate transformer from the database projection configuration.")
            completed_units += 1
            current_state = set_form_progress(
                "generate_log_state",
                completed_units,
                total_progress_units,
                message="Loaded coordinate transformer.",
            )
            render_form_progress_widgets(generate_progress_bar, generate_progress_caption, current_state)
            events = generate_events(
                instruments,
                start_date,
                final_end_date,
                transformer,
                logger=lambda message: append_form_log(root, "generate_stream.log", "generate_log_state", message),
                progress_callback=lambda day_index, total_days, message: render_form_progress_widgets(
                    generate_progress_bar,
                    generate_progress_caption,
                    set_form_progress(
                        "generate_log_state",
                        2 + day_index,
                        total_progress_units,
                        message=message,
                    ),
                ),
            )
            completed_units = 2 + total_generation_days
            save_events_to_json(root, events)
            append_form_log(root, "generate_stream.log", "generate_log_state", f"Saved {len(events)} events to validation_data/events.json.")
            st.session_state["events"] = events
            completed_units += 1
            current_state = set_form_progress(
                "generate_log_state",
                completed_units,
                total_progress_units,
                message=f"Saved {len(events)} events.",
            )
            render_form_progress_widgets(generate_progress_bar, generate_progress_caption, current_state)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            append_form_log(root, "generate_stream.log", "generate_log_state", f"Rendering validation plots for timestamp {timestamp}.")
            beta_path = save_beta_distribution_plot(root, timestamp)
            append_form_log(root, "generate_stream.log", "generate_log_state", f"Saved beta distribution plot to {beta_path.name}.")
            completed_units += 1
            current_state = set_form_progress(
                "generate_log_state",
                completed_units,
                total_progress_units,
                message="Rendered beta distribution plot.",
            )
            render_form_progress_widgets(generate_progress_bar, generate_progress_caption, current_state)
            timeline_path = save_event_timeline_plot(events, root, timestamp, start_date, final_end_date)
            append_form_log(root, "generate_stream.log", "generate_log_state", f"Saved event timeline plot to {timeline_path.name}.")
            completed_units += 1
            current_state = set_form_progress(
                "generate_log_state",
                completed_units,
                total_progress_units,
                message="Rendered event timeline plot.",
            )
            render_form_progress_widgets(generate_progress_bar, generate_progress_caption, current_state)
            spatial_path = save_event_spatial_plot(events, root, timestamp)
            append_form_log(root, "generate_stream.log", "generate_log_state", f"Saved spatial distribution plot to {spatial_path.name}.")
            st.session_state["generate_plot_paths"] = [
                str(beta_path),
                str(timeline_path),
                str(spatial_path),
            ]
            completed_units += 1
            current_state = set_form_progress(
                "generate_log_state",
                completed_units,
                total_progress_units,
                status="completed",
                message=f"Generated {len(events)} events and 3 validation plots.",
            )
            render_form_progress_widgets(generate_progress_bar, generate_progress_caption, current_state)
            append_form_log(root, "generate_stream.log", "generate_log_state", "Generate workflow completed successfully.")
        except Exception as error:
            current_state = set_form_progress(
                "generate_log_state",
                completed_units,
                total_progress_units,
                status="error",
                message=f"Generate failed: {error}",
            )
            render_form_progress_widgets(generate_progress_bar, generate_progress_caption, current_state)
            append_form_log(root, "generate_stream.log", "generate_log_state", f"Generate failed: {error}")
            st.error(f"Generate failed: {error}")

    render_form_status_logs("Event generation log stream", get_form_log_state_snapshot("generate_log_state"))

    for plot_path in st.session_state.get("generate_plot_paths", []):
        render_html_plot(Path(plot_path), height=760)

with st.form("synthesise_time_series_form"):
    st.subheader("Synthesise time series data")
    synthesise_current_state = get_form_log_state_snapshot("synthesise_log_state")
    synthesise_progress_bar = st.progress(max(0.0, min(1.0, float(synthesise_current_state.get("percent_complete", 0.0) or 0.0) / 100.0)))
    synthesise_progress_caption = st.empty()
    render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, synthesise_current_state)
    synthesise_clicked = st.form_submit_button("Synthesise")

    if synthesise_clicked:
        completed_units = 0
        total_progress_units = 1
        current_state = set_form_progress(
            "synthesise_log_state",
            completed_units,
            total_progress_units,
            status="running",
            message="Loading instruments for synthesis.",
        )
        render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)
        append_form_log(root, "synthesise_stream.log", "synthesise_log_state", "Synthesise button clicked by user.")
        try:
            active_start_mode = selected_start_mode()
            active_start_date = selected_start_date(default_start_date)
            instruments = load_instruments_from_json(root)
            if instruments:
                append_form_log(
                    root,
                    "synthesise_stream.log",
                    "synthesise_log_state",
                    f"Loaded {len(instruments)} instruments from validation_data/instruments.json.",
                )
            else:
                append_form_log(
                    root,
                    "synthesise_stream.log",
                    "synthesise_log_state",
                    "No saved instruments found; extracting instruments from the database.",
                )
                instruments, _, _ = extract_instruments(root)
                instruments = load_instruments_from_json(root)
                append_form_log(
                    root,
                    "synthesise_stream.log",
                    "synthesise_log_state",
                    f"Extraction completed and loaded {len(instruments)} instruments.",
                )
            final_end_date = date.today() if use_today else end_date
            total_progress_units = len(instruments) + 8
            completed_units = 1
            current_state = set_form_progress(
                "synthesise_log_state",
                completed_units,
                total_progress_units,
                message=f"Loaded {len(instruments)} instruments.",
            )
            render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)

            events = st.session_state.get("events", [])
            append_form_log(
                root,
                "synthesise_stream.log",
                "synthesise_log_state",
                f"Resolved synthesis window: start_date={active_start_date.isoformat()}, end_date={final_end_date.isoformat()}, start_mode={active_start_mode}.",
            )
            if not events:
                events = load_events_from_json(root)
                if events:
                    append_form_log(
                        root,
                        "synthesise_stream.log",
                        "synthesise_log_state",
                        f"Loaded {len(events)} events from validation_data/events.json.",
                    )
                    completed_units += 1
                    current_state = set_form_progress(
                        "synthesise_log_state",
                        completed_units,
                        total_progress_units,
                        message=f"Loaded {len(events)} events.",
                    )
                    render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)
                else:
                    total_generation_days = max(0, (final_end_date - active_start_date).days + 1)
                    total_progress_units += total_generation_days + 1
                    current_state = set_form_progress(
                        "synthesise_log_state",
                        completed_units,
                        total_progress_units,
                        message="Generating event history required for synthesis.",
                    )
                    render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)
                    append_form_log(
                        root,
                        "synthesise_stream.log",
                        "synthesise_log_state",
                        "No saved events found; generating a fresh event history for synthesis.",
                    )
                    transformer = get_db_transformer(root)
                    append_form_log(root, "synthesise_stream.log", "synthesise_log_state", "Loaded coordinate transformer from the database projection configuration.")
                    completed_units += 1
                    current_state = set_form_progress(
                        "synthesise_log_state",
                        completed_units,
                        total_progress_units,
                        message="Loaded coordinate transformer for event generation.",
                    )
                    render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)
                    events = generate_events(
                        instruments,
                        active_start_date,
                        final_end_date,
                        transformer,
                        logger=lambda message: append_form_log(root, "synthesise_stream.log", "synthesise_log_state", message),
                        progress_callback=lambda day_index, total_days, message: render_form_progress_widgets(
                            synthesise_progress_bar,
                            synthesise_progress_caption,
                            set_form_progress(
                                "synthesise_log_state",
                                completed_units + day_index,
                                total_progress_units,
                                message=message,
                            ),
                        ),
                    )
                    completed_units += total_generation_days
                    save_events_to_json(root, events)
                    append_form_log(root, "synthesise_stream.log", "synthesise_log_state", f"Saved {len(events)} generated events to validation_data/events.json.")
                    completed_units += 1
                    current_state = set_form_progress(
                        "synthesise_log_state",
                        completed_units,
                        total_progress_units,
                        message=f"Saved {len(events)} generated events.",
                    )
                    render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)
                st.session_state["events"] = events
            else:
                append_form_log(
                    root,
                    "synthesise_stream.log",
                    "synthesise_log_state",
                    f"Using {len(events)} events already loaded in session state.",
                )
                completed_units += 1
                current_state = set_form_progress(
                    "synthesise_log_state",
                    completed_units,
                    total_progress_units,
                    message=f"Using {len(events)} session events.",
                )
                render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)

            baseline_offsets: dict[str, float] = {}
            if active_start_mode == TIME_SERIES_START_MODE_SNAPSHOT:
                append_form_log(root, "synthesise_stream.log", "synthesise_log_state", "Fetching snapshot baseline offsets from the target database.")
                try:
                    baseline_offsets = fetch_snapshot_offsets(root, instruments, active_start_date)
                    append_form_log(
                        root,
                        "synthesise_stream.log",
                        "synthesise_log_state",
                        f"Loaded {len(baseline_offsets)} snapshot baseline offsets.",
                    )
                except Exception as error:
                    append_form_log(
                        root,
                        "synthesise_stream.log",
                        "synthesise_log_state",
                        f"Snapshot offsets could not be loaded. Continuing from zero. Details: {error}",
                    )
                    st.warning(f"Snapshot offsets could not be loaded from the target database. Continuing from zero. Details: {error}")
                    baseline_offsets = {}
            else:
                append_form_log(root, "synthesise_stream.log", "synthesise_log_state", "Using zero baseline start mode; snapshot offsets are not required.")
            completed_units += 1
            current_state = set_form_progress(
                "synthesise_log_state",
                completed_units,
                total_progress_units,
                message=(
                    f"Prepared {len(baseline_offsets)} snapshot baseline offsets."
                    if active_start_mode == TIME_SERIES_START_MODE_SNAPSHOT
                    else "Prepared zero-baseline synthesis mode."
                ),
            )
            render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)

            synthesise_timeseries_for_instruments(
                instruments,
                events,
                active_start_date,
                final_end_date,
                baseline_offsets=baseline_offsets,
                include_start_on_sunday=(active_start_mode == TIME_SERIES_START_MODE_ZERO),
                logger=lambda message: append_form_log(root, "synthesise_stream.log", "synthesise_log_state", message),
                progress_callback=lambda instrument_index, total_instruments, message: render_form_progress_widgets(
                    synthesise_progress_bar,
                    synthesise_progress_caption,
                    set_form_progress(
                        "synthesise_log_state",
                        completed_units + instrument_index,
                        total_progress_units,
                        message=message,
                    ),
                ),
            )
            completed_units += len(instruments)
            save_instruments_to_json(root, instruments)
            append_form_log(root, "synthesise_stream.log", "synthesise_log_state", f"Saved {len(instruments)} instruments with synthesised time series to validation_data/instruments.json.")
            st.session_state["instruments"] = instruments
            completed_units += 1
            current_state = set_form_progress(
                "synthesise_log_state",
                completed_units,
                total_progress_units,
                message=f"Saved {len(instruments)} synthesised instruments.",
            )
            render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)

            top_by_type = select_top_instruments(instruments, top_n=3)
            highlighted_by_id: dict[str, Instrument] = {}
            for items in top_by_type.values():
                for instrument in items:
                    if not math.isfinite(float(instrument.latitude)) or not math.isfinite(float(instrument.longitude)):
                        continue
                    highlighted_by_id[instrument.instr_id] = instrument
            highlighted_instruments = list(highlighted_by_id.values())

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            append_form_log(root, "synthesise_stream.log", "synthesise_log_state", f"Rendering synthesis validation plots for timestamp {timestamp}.")
            top_timeseries_path = save_top_timeseries_plot(root, timestamp, top_by_type)
            append_form_log(root, "synthesise_stream.log", "synthesise_log_state", f"Saved top time-series plot to {top_timeseries_path.name}.")
            completed_units += 1
            current_state = set_form_progress(
                "synthesise_log_state",
                completed_units,
                total_progress_units,
                message="Rendered top time-series plot.",
            )
            render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)
            subsurface_depth_path = save_subsurface_depth_profile_plot(
                root,
                timestamp,
                top_by_type.get("subsurface_lateral_displacement", []),
            )
            append_form_log(root, "synthesise_stream.log", "synthesise_log_state", f"Saved subsurface depth-profile plot to {subsurface_depth_path.name}.")
            completed_units += 1
            current_state = set_form_progress(
                "synthesise_log_state",
                completed_units,
                total_progress_units,
                message="Rendered subsurface depth-profile plot.",
            )
            render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)
            spatial_overlay_path = save_event_spatial_plot(
                events,
                root,
                timestamp,
                highlighted_instruments=highlighted_instruments,
                show_event_epicentres=False,
                highlighted_marker_size=8,
            )
            append_form_log(root, "synthesise_stream.log", "synthesise_log_state", f"Saved synthesis spatial overlay plot to {spatial_overlay_path.name}.")
            completed_units += 1
            current_state = set_form_progress(
                "synthesise_log_state",
                completed_units,
                total_progress_units,
                message="Rendered synthesis spatial overlay plot.",
            )
            render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)
            timeslice_paths = save_timeslice_validation_plots(root, timestamp, instruments, events)
            append_form_log(
                root,
                "synthesise_stream.log",
                "synthesise_log_state",
                f"Saved {len(timeslice_paths)} timeslice validation plots.",
            )
            st.session_state["synthesis_plot_paths"] = [
                str(top_timeseries_path),
                str(subsurface_depth_path),
                str(spatial_overlay_path),
                *[str(path) for path in timeslice_paths],
            ]
            completed_units += 1
            current_state = set_form_progress(
                "synthesise_log_state",
                completed_units,
                total_progress_units,
                status="completed",
                message=f"Synthesised {len(instruments)} instruments and rendered {3 + len(timeslice_paths)} validation plots.",
            )
            render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)
            append_form_log(root, "synthesise_stream.log", "synthesise_log_state", "Synthesise workflow completed successfully.")
        except Exception as error:
            current_state = set_form_progress(
                "synthesise_log_state",
                completed_units,
                total_progress_units,
                status="error",
                message=f"Synthesise failed: {error}",
            )
            render_form_progress_widgets(synthesise_progress_bar, synthesise_progress_caption, current_state)
            append_form_log(root, "synthesise_stream.log", "synthesise_log_state", f"Synthesise failed: {error}")
            st.error(f"Synthesise failed: {error}")

    render_form_status_logs("Time-series synthesis log stream", get_form_log_state_snapshot("synthesise_log_state"))

    for plot_path in st.session_state.get("synthesis_plot_paths", []):
        render_html_plot(Path(plot_path), height=760)

with st.form("assimilate_database_fields_form"):
    st.subheader("Assimilate database fields")
    assimilate_current_state = get_form_log_state_snapshot("assimilate_log_state")
    st.badge(
        str(assimilate_current_state.get("status", "idle")).replace("_", " ").title(),
        color=form_status_badge_color(str(assimilate_current_state.get("status", "idle"))),
    )
    assimilate_progress_bar = st.progress(
        max(0.0, min(1.0, float(assimilate_current_state.get("percent_complete", 0.0) or 0.0) / 100.0))
    )
    assimilate_progress_caption = st.empty()
    render_form_progress_widgets(assimilate_progress_bar, assimilate_progress_caption, assimilate_current_state)
    assimilate_clicked = st.form_submit_button("Assimilate")

    if assimilate_clicked:
        active_start_mode = selected_start_mode()
        active_start_date = selected_start_date(default_start_date)
        events: list[Event] = st.session_state.get("events", [])
        final_end_date = date.today() if use_today else end_date

        current_state = set_form_progress(
            "assimilate_log_state",
            0,
            100,
            status="running",
            message="Preparing valid instrument time series for assimilation.",
        )
        render_form_progress_widgets(assimilate_progress_bar, assimilate_progress_caption, current_state)

        try:
            instruments, events = load_or_synthesise_valid_instruments(
                root,
                active_start_date,
                final_end_date,
                events,
                start_mode=active_start_mode,
                progress_callback=lambda fraction_complete, message: render_form_progress_widgets(
                    assimilate_progress_bar,
                    assimilate_progress_caption,
                    set_form_progress(
                        "assimilate_log_state",
                        int(max(0.0, min(1.0, fraction_complete)) * 35),
                        100,
                        message=message,
                    ),
                ),
            )
            st.session_state["instruments"] = instruments
            st.session_state["events"] = events

            current_state = set_form_progress(
                "assimilate_log_state",
                35,
                100,
                message=f"Building assimilation dataframe from {len(instruments)} instruments.",
            )
            render_form_progress_widgets(assimilate_progress_bar, assimilate_progress_caption, current_state)

            mydata_df = build_assimilation_dataframe(
                instruments,
                progress_callback=lambda instrument_index, total_instruments, message: render_form_progress_widgets(
                    assimilate_progress_bar,
                    assimilate_progress_caption,
                    set_form_progress(
                        "assimilate_log_state",
                        35 + int((instrument_index / total_instruments if total_instruments else 1.0) * 55),
                        100,
                        message=message,
                    ),
                ),
            )

            current_state = set_form_progress(
                "assimilate_log_state",
                92,
                100,
                message=f"Saving {len(mydata_df)} assimilated rows to validation_data/mydata.json.",
            )
            render_form_progress_widgets(assimilate_progress_bar, assimilate_progress_caption, current_state)
            output_path = save_mydata_json(root, mydata_df)

            current_state = set_form_progress(
                "assimilate_log_state",
                97,
                100,
                message="Updating in-memory assimilation preview and session data.",
            )
            render_form_progress_widgets(assimilate_progress_bar, assimilate_progress_caption, current_state)

            st.session_state["assimilated_preview"] = mydata_df.head(200).to_dict(orient="records")
            st.session_state["assimilated_total_rows"] = int(len(mydata_df))
            st.session_state["assimilated_output_path"] = str(output_path)
            st.session_state["mydata_df_records"] = mydata_df.to_dict(orient="records")

            current_state = set_form_progress(
                "assimilate_log_state",
                100,
                100,
                status="completed",
                message=f"Assimilation completed with {len(mydata_df)} rows.",
            )
            render_form_progress_widgets(assimilate_progress_bar, assimilate_progress_caption, current_state)
        except Exception as error:
            current_state = set_form_progress(
                "assimilate_log_state",
                int(get_form_log_state_snapshot("assimilate_log_state").get("completed_units", 0) or 0),
                100,
                status="error",
                message=f"Assimilation failed: {error}",
            )
            render_form_progress_widgets(assimilate_progress_bar, assimilate_progress_caption, current_state)
            st.error(f"Assimilation failed: {error}")

    preview_records = st.session_state.get("assimilated_preview", [])
    preview_df = pd.DataFrame(preview_records, columns=["instr_id", "date1", "data1", "custom_fields"])
    st.dataframe(preview_df, use_container_width=True)
    st.caption(f"Total rows generated: {st.session_state.get('assimilated_total_rows', 0)}")

def render_write_to_database_form() -> None:
    runtime = get_db_write_runtime()
    st.subheader("Write to database")

    current_state = get_db_write_state()

    # Ensure UI reflects actual thread liveness even during teardown transitions.
    live_thread = runtime_thread_is_alive(runtime)
    if live_thread != bool(current_state.get("thread_alive")):
        update_db_write_state(runtime=runtime, thread_alive=live_thread)
        current_state = get_db_write_state()

    render_async_status_pill(str(current_state.get("async_status", "idle")))

    total_rows = int(current_state.get("total_rows", 0) or 0)
    rows_written = int(current_state.get("rows_written", 0) or 0)
    percent_complete = float(current_state.get("percent_complete", 0.0) or 0.0)
    progress_value = max(0.0, min(1.0, percent_complete / 100.0))
    st.progress(progress_value)
    st.caption(f"Rows written: {rows_written} / {total_rows} ({percent_complete:.2f}%)")
    if current_state.get("message"):
        st.caption(str(current_state.get("message")))

    thread_alive = bool(current_state.get("thread_alive"))
    st.badge("Thread Running" if thread_alive else "Thread Stopped", color=("green" if thread_alive else "gray"))
    completed_at = str(current_state.get("thread_completed_at", "") or "")
    terminal_message = str(current_state.get("thread_terminal_message", "") or "")
    if completed_at:
        st.caption(f"Thread completed at: {completed_at}")
    if terminal_message:
        st.caption(terminal_message)

    render_write_status_logs(current_state)

    write_column, cancel_column = st.columns(2)
    write_enabled = not runtime_thread_is_alive(runtime)
    with write_column:
        write_clicked = st.button("Write", disabled=not write_enabled, use_container_width=True)
    with cancel_column:
        # Keep cancel enabled until the worker thread is fully terminated.
        cancel_enabled = runtime_thread_is_alive(runtime)
        cancel_clicked = st.button("Cancel", disabled=not cancel_enabled, use_container_width=True)

    if cancel_clicked:
        append_stream_log(root, "Cancel button clicked by user.", runtime=runtime)
        request_cancel(root, runtime)
        st.warning("Cancellation requested. No further SQL write statements will be executed.")
        st.rerun()

    if write_clicked:
        runtime = get_db_write_runtime()
        existing_thread = runtime.get("thread")
        append_stream_log(root, "Write button clicked by user.", runtime=runtime)
        if isinstance(existing_thread, threading.Thread) and existing_thread.is_alive():
            append_stream_log(root, "Write request rejected because a job is already running.", runtime=runtime)
            st.warning("A write operation is already running.")
        else:
            active_start_mode = selected_start_mode()
            active_start_date = selected_start_date(default_start_date)
            final_end_date = date.today() if use_today else end_date
            events_for_write: list[Event] = st.session_state.get("events", [])

            append_stream_log(root, "Beginning pre-write data preparation.", runtime=runtime)
            dataframe, was_reassimilated, source_message, rebuilt_instruments, rebuilt_events = prepare_mydata_for_write(
                root,
                active_start_date,
                final_end_date,
                events_for_write,
                start_mode=active_start_mode,
                runtime=runtime,
            )

            if was_reassimilated and rebuilt_instruments:
                st.session_state["instruments"] = rebuilt_instruments
            if rebuilt_events:
                st.session_state["events"] = rebuilt_events

            if dataframe.empty:
                append_stream_log(root, "Write request aborted: prepared dataframe is empty.", runtime=runtime)
                update_db_write_state(
                    runtime=runtime,
                    status="error",
                    async_status="error",
                    rows_written=0,
                    total_rows=0,
                    percent_complete=0.0,
                    message=source_message,
                    started=True,
                )
                st.error(source_message)
            else:
                runtime["cancel_event"].clear()
                update_db_write_state(
                    runtime=runtime,
                    status="running",
                    async_status="starting",
                    rows_written=0,
                    total_rows=int(len(dataframe)),
                    percent_complete=0.0,
                    message=source_message,
                    started=True,
                    thread_alive=True,
                    thread_completed_at="",
                    thread_terminal_message="",
                )
                append_stream_log(root, f"Prepared {len(dataframe)} rows. Spawning background write thread.", runtime=runtime)

                normalized_records = dataframe.to_dict(orient="records")
                worker = threading.Thread(
                    target=run_database_write_job,
                    args=(
                        root,
                        normalized_records,
                        runtime,
                        active_start_date if active_start_mode == TIME_SERIES_START_MODE_SNAPSHOT else None,
                    ),
                    daemon=True,
                )
                runtime["thread"] = worker
                worker.start()
                st.success("Write process started in background.")
                st.rerun()


if hasattr(st, "fragment") and str(get_db_write_state().get("status")) in {"running", "cancelling"}:
    @st.fragment(run_every="1s")
    def write_form_fragment() -> None:
        render_write_to_database_form()

        current_state = get_db_write_state()
        if str(current_state.get("status")) not in {"running", "cancelling"} and not bool(current_state.get("thread_alive")):
            st.rerun()

    write_form_fragment()
else:
    render_write_to_database_form()
    if not hasattr(st, "fragment") and str(get_db_write_state().get("status")) in {"running", "cancelling"}:
        st.info("Automatic refresh is unavailable in this Streamlit version. Interact with the page to refresh status.")