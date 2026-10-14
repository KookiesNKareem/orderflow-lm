"""Normalize live Binance Futures recorder envelopes into event-model rows."""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .reconstruct import BATCH_ROWS, OFF_MAX, OFF_MIN, SCHEMA, Book, infer_tick


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("r")


def _records(paths: Iterable[Path]) -> Iterator[dict]:
    for path in sorted(paths):
        with _open_text(path) as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: malformed JSON") from error
                if record.get("schema_version") != 1:
                    raise ValueError(f"{path}:{line_number}: unsupported envelope schema")
                yield record


def _timestamp_us(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1e6)


def _stream_paths(root: Path, channel: str) -> list[Path]:
    paths = [
        *root.glob(f"**/binance_futures/{channel}.jsonl"),
        *root.glob(f"**/binance_futures/{channel}.jsonl.gz"),
    ]
    result = sorted(set(paths))
    if not result:
        raise FileNotFoundError(f"no binance_futures/{channel} JSONL under {root}")
    return result


def reconstruct_live_binance(root: Path, out_path: Path) -> dict:
    """Convert one recorder tree into the normalized Tardis-compatible schema."""

    root = Path(root)
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    depth_records = _records(_stream_paths(root, "book_delta"))
    trade_records = _records(_stream_paths(root, "trade"))
    next_trade = next(trade_records, None)
    bids, asks = Book(True), Book(False)
    tick: float | None = None
    previous_sequence: int | None = None
    expect_snapshot_bridge = False
    pending_reset = False
    seen_trade_ids: set[str] = set()
    counts = {
        "events": 0,
        "frames": 0,
        "trades": 0,
        "snapshots": 0,
        "resets": 0,
        "duplicate_trades_dropped": 0,
    }

    writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")
    columns: list[list] = [[] for _ in SCHEMA]

    def flush() -> None:
        nonlocal columns
        if not columns[0]:
            return
        writer.write_batch(
            pa.record_batch(
                [
                    pa.array(values, type=field.type)
                    for values, field in zip(columns, SCHEMA)
                ],
                schema=SCHEMA,
            )
        )
        columns = [[] for _ in SCHEMA]

    def emit(
        ts: int,
        kind: int,
        new_frame: bool,
        side: int,
        offset: int,
        size: float,
        best_bid: float,
        best_ask: float,
    ) -> None:
        nonlocal pending_reset
        reset = pending_reset and (kind == 1 or new_frame)
        if reset:
            pending_reset = False
            counts["resets"] += 1
        values = (
            ts,
            kind,
            new_frame,
            reset,
            side,
            offset,
            size,
            best_bid,
            best_ask,
        )
        for column, value in zip(columns, values):
            column.append(value)
        counts["events"] += 1
        if len(columns[0]) >= BATCH_ROWS:
            flush()

    def emit_trade_record(record: dict) -> None:
        payload = record.get("payload") or {}
        for trade in payload.get("trades") or []:
            trade_id = str(trade.get("trade_id") or "")
            if trade_id and trade_id in seen_trade_ids:
                counts["duplicate_trades_dropped"] += 1
                continue
            if trade_id:
                seen_trade_ids.add(trade_id)
            best_bid, best_ask = bids.best(), asks.best()
            if best_bid is None or best_ask is None or tick is None:
                continue
            side_name = trade.get("taker_side")
            if side_name not in {"buy", "sell"}:
                raise ValueError(f"invalid taker side: {side_name!r}")
            price = float(trade["price"])
            if side_name == "buy":
                offset = round((price - best_ask) / tick)
                side = 1
            else:
                offset = round((best_bid - price) / tick)
                side = 0
            timestamp = trade.get("exchange_at") or record.get("exchange_at")
            if timestamp is None:
                raise ValueError("trade has no exchange timestamp")
            emit(
                _timestamp_us(timestamp),
                1,
                False,
                side,
                max(-128, min(127, offset)),
                float(trade["quantity"]),
                best_bid,
                best_ask,
            )
            counts["trades"] += 1

    def emit_trades_until(timestamp_us: int, *, inclusive: bool) -> None:
        nonlocal next_trade
        while next_trade is not None:
            value = next_trade.get("exchange_at")
            if value is None:
                raise ValueError("trade envelope has no exchange timestamp")
            trade_us = _timestamp_us(value)
            if trade_us > timestamp_us or (trade_us == timestamp_us and not inclusive):
                return
            emit_trade_record(next_trade)
            next_trade = next(trade_records, None)

    try:
        for record in depth_records:
            if record.get("source") != "binance_futures":
                raise ValueError("book delta has the wrong source")
            payload = record.get("payload") or {}
            timestamp = record.get("exchange_at")
            if timestamp is None:
                raise ValueError("book delta has no exchange timestamp")
            timestamp_us = _timestamp_us(timestamp)
            action = payload.get("action")
            updates = payload.get("updates") or []
            sequence = int(payload["sequence"])

            if action == "snapshot":
                emit_trades_until(timestamp_us, inclusive=False)
                bids.clear()
                asks.clear()
                prices = []
                for update in updates:
                    price = float(update["price"])
                    quantity = float(update["quantity"])
                    side = update["side"]
                    if side not in {"bid", "ask"}:
                        raise ValueError(f"invalid book side: {side!r}")
                    (bids if side == "bid" else asks).set(price, quantity)
                    prices.append(price)
                if bids.best() is None or asks.best() is None:
                    raise ValueError("snapshot did not contain both book sides")
                inferred = infer_tick(prices)
                if tick is not None and abs(inferred - tick) > 1e-12:
                    raise ValueError(f"tick changed from {tick} to {inferred}")
                tick = inferred
                previous_sequence = sequence
                expect_snapshot_bridge = True
                pending_reset = True
                counts["snapshots"] += 1
                continue

            if action != "update":
                raise ValueError(f"unknown book action: {action!r}")
            if previous_sequence is None:
                raise ValueError("book update arrived before a snapshot")
            if expect_snapshot_bridge:
                first_update = payload.get("first_update_id")
                if (
                    first_update is None
                    or int(first_update) > previous_sequence + 1
                    or sequence < previous_sequence + 1
                ):
                    raise ValueError(
                        "book snapshot bridge gap: "
                        f"{first_update} <= {previous_sequence + 1} <= {sequence} failed"
                    )
                expect_snapshot_bridge = False
            else:
                previous = payload.get("previous_update_id")
                if previous is None or int(previous) != previous_sequence:
                    raise ValueError(
                        f"book sequence gap: expected previous {previous_sequence}, "
                        f"got {previous}"
                    )
            if sequence <= previous_sequence:
                raise ValueError(
                    f"book sequence did not advance: {previous_sequence} -> {sequence}"
                )

            emit_trades_until(timestamp_us, inclusive=True)
            pre_bid, pre_ask = bids.best(), asks.best()
            if pre_bid is None or pre_ask is None or tick is None:
                raise ValueError("book update has no valid pre-frame book")
            changed: dict[tuple[int, float], float] = {}
            for update in updates:
                side_name = update["side"]
                if side_name not in {"bid", "ask"}:
                    raise ValueError(f"invalid book side: {side_name!r}")
                side = 0 if side_name == "bid" else 1
                price = float(update["price"])
                quantity = float(update["quantity"])
                (bids if side == 0 else asks).set(price, quantity)
                changed[(side, price)] = quantity
            post_bid, post_ask = bids.best(), asks.best()
            if post_bid is None or post_ask is None or post_bid >= post_ask:
                raise ValueError(f"invalid post-frame book: {post_bid}/{post_ask}")
            output = []
            for (side, price), quantity in changed.items():
                best = pre_bid if side == 0 else pre_ask
                offset = (
                    round((best - price) / tick)
                    if side == 0
                    else round((price - best) / tick)
                )
                if offset <= OFF_MAX and (offset >= OFF_MIN or quantity > 0):
                    output.append((side, offset, quantity))
            output.sort(key=lambda item: (item[0], item[1]))
            for index, (side, offset, quantity) in enumerate(output):
                emit(
                    timestamp_us,
                    0,
                    index == 0,
                    side,
                    offset,
                    quantity,
                    post_bid,
                    post_ask,
                )
            if output:
                counts["frames"] += 1
            previous_sequence = sequence

        while next_trade is not None:
            emit_trade_record(next_trade)
            next_trade = next(trade_records, None)
        flush()
    finally:
        writer.close()

    if counts["events"] == 0:
        out_path.unlink(missing_ok=True)
        raise ValueError("no normalized events emitted")
    metadata = {
        **counts,
        "tick": tick,
        "source": str(root.resolve()),
        "schema": [field.name for field in SCHEMA],
        "alignment": (
            "Binance depth messages are atomic frames; trades at the same exchange "
            "timestamp precede the frame; snapshots reset state without becoming events"
        ),
    }
    Path(f"{out_path}.meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
