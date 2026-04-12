from __future__ import annotations

from typing import Any


def ctc_token_segments(pred_ids: Any, tokenizer: Any, blank_id: int, time_offset: float) -> list[dict[str, Any]]:
    if hasattr(pred_ids, "tolist"):
        pred_ids = pred_ids.tolist()
    segments: list[dict[str, Any]] = []
    if not pred_ids:
        return segments

    current_id = pred_ids[0]
    start_index = 0
    for idx in range(1, len(pred_ids) + 1):
        next_id = pred_ids[idx] if idx < len(pred_ids) else None
        if next_id == current_id:
            continue
        if current_id != blank_id:
            token = tokenizer.convert_ids_to_tokens(int(current_id))
            if token not in {None, "", "[PAD]", "<pad>"}:
                segments.append(
                    {
                        "token_id": int(current_id),
                        "token": token,
                        "start_time": float(start_index * time_offset),
                        "end_time": float(idx * time_offset),
                    }
                )
        start_index = idx
        current_id = next_id
    return segments
