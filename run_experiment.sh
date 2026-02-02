#!/bin/bash
# SEAL Watermark Experiment Runner
# Executes the complete experiment pipeline

set -e  # Exit on error

echo "=========================================="
echo "SEAL Watermark Experiment"
echo "Full Pipeline Execution"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating now..."
    source venv/bin/activate
fi

# Set environment variables
export HF_HOME="./models_cache"
export TRANSFORMERS_CACHE="./models_cache"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Check GPU
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "❌ Error: CUDA not available. This experiment requires GPU."
    echo "Please check your PyTorch installation."
    exit 1
fi

echo "✓ Environment ready"
echo "✓ GPU available: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# Get start time
start_time=$(date +%s)
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="results/seal_eval_${timestamp}"

echo "Output directory: $output_dir"
echo ""
echo "=========================================="
echo "Pipeline Steps:"
echo "=========================================="
echo "  [1/7] Setup models"
echo "  [2/7] Load prompts"
echo "  [3/7] Generate watermarked images (500)"
echo "  [4/7] Generate negative images (1000)"
echo "  [5/7] Apply transformations (7 types)"
echo "  [6/7] Run detection (4,500 images)"
echo "  [7/7] Generate ROC curves"
echo ""
echo "⏱️  Estimated time: 15-20 hours on RTX 4090"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# Step 1: Setup models (verify they're cached)
echo "=========================================="
echo "[1/7] Verifying Models"
echo "=========================================="
python3 scripts/01_setup_models.py --output-dir "$output_dir"
echo ""

# Step 2: Load prompts
echo "=========================================="
echo "[2/7] Loading Prompts"
echo "=========================================="
python3 scripts/02_load_prompts.py --output-dir "$output_dir"
echo ""

# Step 3: Generate watermarked images
echo "=========================================="
echo "[3/7] Generating Watermarked Images"
echo "=========================================="
echo "⏱️  This step takes ~5-7 hours"
python3 scripts/03_generate_watermarked.py --output-dir "$output_dir"
echo ""

# Step 4: Generate negative images
echo "=========================================="
echo "[4/7] Generating Negative Images"
echo "=========================================="
echo "⏱️  This step takes ~3-4 hours"
python3 scripts/04_generate_negatives.py --output-dir "$output_dir"
echo ""

# Step 5: Apply transformations
echo "=========================================="
echo "[5/7] Applying Transformations"
echo "=========================================="
echo "⏱️  This step takes ~30 minutes"
python3 scripts/05_apply_transforms.py --output-dir "$output_dir"
echo ""

# Step 6: Run detection
echo "=========================================="
echo "[6/7] Running Detection (Per-Patch Search)"
echo "=========================================="
echo "⏱️  This step takes ~6-8 hours"
python3 scripts/06_run_detection.py --output-dir "$output_dir"
echo ""

# Step 7: Generate ROC curves
echo "=========================================="
echo "[7/7] Generating ROC Curves"
echo "=========================================="
python3 scripts/07_generate_roc.py --output-dir "$output_dir"
echo ""

# Calculate elapsed time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))

echo ""
echo "=========================================="
echo "✅ EXPERIMENT COMPLETE!"
echo "=========================================="
echo ""
echo "⏱️  Total time: ${hours}h ${minutes}m"
echo ""
echo "📊 Results saved to: $output_dir"
echo ""
echo "Key files:"
echo "  ✓ $output_dir/results.csv"
echo "  ✓ $output_dir/metrics.csv"
echo "  ✓ $output_dir/roc_curves.png"
echo ""
echo "View results:"
echo "  - Open: $output_dir/roc_curves.png"
echo "  - Read: $output_dir/metrics.csv"
echo ""
echo "Next steps:"
echo "  1. Validate results: python scripts/validate_results.py --results-dir $output_dir"
echo "  2. Visualize: jupyter notebook notebooks/analysis.ipynb"
echo "  3. Archive: tar -czf seal_results_${timestamp}.tar.gz $output_dir"
echo ""
