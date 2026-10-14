import json
from pathlib import Path

import pyarrow.parquet as pq

from oflm.live_reconstruct import reconstruct_live_binance


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _envelope(channel: str, timestamp: str, payload: dict) -> dict:
    return {
        "schema_version": 1,
        "source": "binance_futures",
        "channel": channel,
        "received_at_ns": 1,
        "exchange_at": timestamp,
        "payload": payload,
    }


def test_reconstruct_live_binance_preserves_frame_and_reset_semantics(tmp_path):
    raw = tmp_path / "raw"
    source = raw / "2026-01-01" / "00" / "binance_futures"
    _write(
        source / "book_delta.jsonl",
        [
            _envelope(
                "book_delta",
                "2026-01-01T00:00:00.000Z",
                {
                    "action": "snapshot",
                    "sequence": 10,
                    "previous_update_id": None,
                    "updates": [
                        {"side": "bid", "price": "100", "quantity": "2"},
                        {"side": "ask", "price": "101", "quantity": "3"},
                    ],
                },
            ),
            _envelope(
                "book_delta",
                "2026-01-01T00:00:02.000Z",
                {
                    "action": "update",
                    "first_update_id": 11,
                    "sequence": 12,
                    "previous_update_id": 10,
                    "updates": [
                        {"side": "bid", "price": "100", "quantity": "0"},
                        {"side": "bid", "price": "99", "quantity": "4"},
                        {"side": "ask", "price": "101", "quantity": "5"},
                    ],
                },
            ),
        ],
    )
    _write(
        source / "trade.jsonl",
        [
            _envelope(
                "trade",
                "2026-01-01T00:00:01.500Z",
                {
                    "trades": [
                        {
                            "trade_id": "a",
                            "exchange_at": "2026-01-01T00:00:01.500Z",
                            "taker_side": "buy",
                            "price": "101",
                            "quantity": "0.5",
                        }
                    ]
                },
            )
        ],
    )
    out = tmp_path / "normalized.parquet"

    metadata = reconstruct_live_binance(raw, out)
    frame = pq.read_table(out).to_pydict()

    assert metadata["tick"] == 1.0
    assert metadata["trades"] == 1
    assert metadata["frames"] == 1
    assert frame["kind"] == [1, 0, 0, 0]
    assert frame["reset"] == [True, False, False, False]
    assert frame["new_frame"] == [False, True, False, False]
    assert frame["side"] == [1, 0, 0, 1]
    assert frame["offset"] == [0, 0, 1, 0]
    assert frame["best_bid"] == [100.0, 99.0, 99.0, 99.0]
    assert frame["best_ask"] == [101.0, 101.0, 101.0, 101.0]


def test_reconstruct_live_binance_fails_closed_on_sequence_gap(tmp_path):
    raw = tmp_path / "raw"
    source = raw / "2026-01-01" / "00" / "binance_futures"
    _write(
        source / "book_delta.jsonl",
        [
            _envelope(
                "book_delta",
                "2026-01-01T00:00:00.000Z",
                {
                    "action": "snapshot",
                    "sequence": 10,
                    "previous_update_id": None,
                    "updates": [
                        {"side": "bid", "price": "100", "quantity": "2"},
                        {"side": "ask", "price": "101", "quantity": "3"},
                    ],
                },
            ),
            _envelope(
                "book_delta",
                "2026-01-01T00:00:02.000Z",
                {
                    "action": "update",
                    "first_update_id": 12,
                    "sequence": 14,
                    "previous_update_id": 11,
                    "updates": [],
                },
            ),
        ],
    )
    _write(
        source / "trade.jsonl",
        [
            _envelope(
                "trade",
                "2026-01-01T00:00:01.000Z",
                {"trades": []},
            )
        ],
    )

    try:
        reconstruct_live_binance(raw, tmp_path / "normalized.parquet")
    except ValueError as error:
        assert "bridge gap" in str(error)
    else:
        raise AssertionError("sequence gap passed reconstruction")
