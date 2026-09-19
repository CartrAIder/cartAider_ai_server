#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 CARTGATE_MODEL_HOST_DIR" >&2
    exit 2
fi

bundle_root=$1
if [ ! -d "$bundle_root" ]; then
    echo "model bundle directory does not exist: $bundle_root" >&2
    exit 1
fi

missing=0
for relative_path in dino_arc.onnx runs/detector/best.pt products.csv; do
    if [ ! -f "$bundle_root/$relative_path" ] || [ ! -s "$bundle_root/$relative_path" ]; then
        echo "model bundle is missing or empty: $relative_path under: $bundle_root" >&2
        missing=1
    fi
done

if [ ! -f "$bundle_root/out/gallery.npz" ] || [ ! -s "$bundle_root/out/gallery.npz" ]; then
    echo "model bundle is missing or empty: out/gallery.npz under: $bundle_root" >&2
    missing=1
fi

if [ "$missing" -ne 0 ]; then
    echo "CARTGATE_MODEL_HOST_DIR must point directly to the CartGate_AI bundle root" >&2
    exit 1
fi

echo "validated CartGate model bundle: $bundle_root"
