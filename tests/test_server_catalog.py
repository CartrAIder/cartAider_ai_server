import csv

import pytest

from cartgate.server.catalog import CatalogError, ProductCatalog, ReceiptItem


def write_catalog(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sku_id", "name", "barcode"])
        writer.writeheader()
        writer.writerows(rows)


def test_receipt_mapping_preserves_leading_zero_barcode(tmp_path):
    """Catches parsing barcodes as numbers and losing the leading zero."""
    catalog_path = tmp_path / "products.csv"
    write_catalog(catalog_path, [{"sku_id": "S0001", "name": "aloe", "barcode": "0000289908820"}])

    receipt = ProductCatalog.from_csv(catalog_path).receipt_to_skus(
        [ReceiptItem(barcode="0000289908820", qty=2)]
    )

    assert receipt == {"S0001": 2}


def test_receipt_mapping_rejects_unknown_barcode(tmp_path):
    """Catches silently omitting a paid Spring item before vision inference."""
    catalog_path = tmp_path / "products.csv"
    write_catalog(catalog_path, [{"sku_id": "S0001", "name": "aloe", "barcode": "0000289908820"}])

    with pytest.raises(CatalogError, match="unknown barcode"):
        ProductCatalog.from_csv(catalog_path).receipt_to_skus(
            [ReceiptItem(barcode="0000000000000", qty=1)]
        )


@pytest.mark.parametrize("qty", [0, -1])
def test_receipt_mapping_rejects_non_positive_quantity(tmp_path, qty):
    """Catches a malformed Spring receipt becoming an empty or negative cart."""
    catalog_path = tmp_path / "products.csv"
    write_catalog(catalog_path, [{"sku_id": "S0001", "name": "aloe", "barcode": "0000289908820"}])

    with pytest.raises(CatalogError, match="positive"):
        ProductCatalog.from_csv(catalog_path).receipt_to_skus(
            [ReceiptItem(barcode="0000289908820", qty=qty)]
        )
