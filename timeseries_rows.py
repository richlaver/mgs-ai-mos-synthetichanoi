from __future__ import annotations

from datetime import date, datetime, time
from typing import Any


def parse_iso_datetime(raw_time: Any) -> datetime | None:
    if raw_time is None:
        return None
    if isinstance(raw_time, datetime):
        return raw_time
    try:
        return datetime.fromisoformat(str(raw_time))
    except ValueError:
        return None


def split_timeseries_rows_for_write(
    normalized_rows: list[dict[str, Any]],
    updated_to_time: datetime,
    preserve_before_date: date | None = None,
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]], int]:
    """Split assimilated rows into mydata (past) and futuredata (future) payloads.

    When ``preserve_before_date`` is set (Snapshot writes), rows strictly before that
    date are skipped. Those rows were intentionally left in place by the partial
    delete and must not be re-inserted, otherwise exact duplicates are created in
    ``mydata`` and/or ``futuredata``.

    Exact duplicate payloads are also dropped as a safeguard.
    """
    preserve_before_dt = (
        datetime.combine(preserve_before_date, time.min) if preserve_before_date is not None else None
    )
    past_rows: list[tuple[str, str, str, str]] = []
    future_rows: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    duplicates_skipped = 0

    for row in normalized_rows:
        date1_dt = parse_iso_datetime(row.get("date1_dt"))
        if date1_dt is None:
            date1_dt = parse_iso_datetime(row.get("date1"))
        if date1_dt is None:
            continue
        if preserve_before_dt is not None and date1_dt < preserve_before_dt:
            continue

        payload = (
            str(row.get("instr_id", "")),
            date1_dt.strftime("%Y-%m-%d %H:%M:%S"),
            str(row.get("data1", "")),
            str(row.get("custom_fields", "")),
        )
        if payload in seen:
            duplicates_skipped += 1
            continue
        seen.add(payload)

        if date1_dt < updated_to_time:
            past_rows.append(payload)
        else:
            future_rows.append(payload)

    return past_rows, future_rows, duplicates_skipped
