#!/bin/bash
set -e
echo "🔬 Running SEAL Watermark Experiment..."
source venv/bin/activate
python scripts/01_generate_watermarked.py
python scripts/02_generate_negatives.py
python scripts/03_apply_transforms.py
python scripts/04_run_detection.py
python scripts/05_generate_roc.py
echo "✅ Experiment complete! Results in ./results/"
