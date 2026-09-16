"""Receipt barcode to CartGate SKU mapping."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


class CatalogError(ValueError):
    """A Spring receipt cannot be represented by the deployed model catalog."""


@dataclass(frozen=True)
class ReceiptItem:
    barcode: str
    qty: int


@dataclass(frozen=True)
class ProductCatalog:
    _sku_by_barcode: dict[str, str]

    @classmethod
    def from_csv(cls, path: str | Path) -> "ProductCatalog":
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            required = {"sku_id", "barcode"}
            if not rows.fieldnames or not required.issubset(rows.fieldnames):
                raise CatalogError("catalog must contain sku_id and barcode columns")
            mapping: dict[str, str] = {}
            for row in rows:
                barcode = (row.get("barcode") or "").strip()
                sku_id = (row.get("sku_id") or "").strip()
                if not barcode or not sku_id:
                    raise CatalogError("catalog contains an empty barcode or sku_id")
                if barcode in mapping:
                    raise CatalogError(f"duplicate barcode: {barcode}")
                mapping[barcode] = sku_id
        if not mapping:
            raise CatalogError("catalog contains no products")
        return cls(mapping)

    def receipt_to_skus(self, items: list[ReceiptItem]) -> dict[str, int]:
        receipt: dict[str, int] = {}
        for item in items:
            barcode = str(item.barcode).strip()
            if not barcode or barcode not in self._sku_by_barcode:
                raise CatalogError(f"unknown barcode: {barcode}")
            if not isinstance(item.qty, int) or isinstance(item.qty, bool) or item.qty <= 0:
                raise CatalogError("receipt quantity must be positive")
            sku_id = self._sku_by_barcode[barcode]
            receipt[sku_id] = receipt.get(sku_id, 0) + item.qty
        if not receipt:
            raise CatalogError("receipt contains no items")
        return receipt
