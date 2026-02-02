#!/bin/bash

set -e

echo "Setting up SEAL experiment..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python - << 'EOF'
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer

print("Downloading BLIP-2...")
Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

print("Downloading sentence embedder...")
SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

print("Models cached.")
EOF

mkdir -p data results/wm results/neg results/metrics

echo "Done. Run ./run_experiment.sh"
