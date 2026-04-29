from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import ReceiptItem, ResolutionResult
from .normalize import parse_price


def load_receipt_items(dataset_path: str | Path, dataset_name: str) -> List[ReceiptItem]:
    path = Path(dataset_path)
    payload = json.loads(path.read_text())
    receipts = payload.get("receipts", [])
    items: List[ReceiptItem] = []

    for receipt_index, receipt in enumerate(receipts):
        merchant = str(receipt.get("merchant", "")).strip()
        receipt_urls = list(receipt.get("receipt_urls") or [])
        for item_index, item in enumerate(receipt.get("items", [])):
            raw_id = item.get("item_id")
            items.append(
                ReceiptItem(
                    dataset_name=dataset_name,
                    receipt_index=receipt_index,
                    item_index=item_index,
                    merchant=merchant,
                    item_name=str(item.get("item_name", "")).strip(),
                    item_id=str(raw_id).strip() if raw_id not in (None, "") else None,
                    item_price=parse_price(item.get("item_price")),
                    receipt_urls=receipt_urls,
                    reference_photo_urls=list(item.get("reference_photo_urls") or []),
                )
            )

    return items


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_results(path: str | Path, results: Iterable[ResolutionResult]) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    serialized = [result.to_dict() for result in results]
    output_path.write_text(json.dumps(serialized, indent=2))
    return output_path


def write_text(path: str | Path, content: str) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(content)
    return output_path
