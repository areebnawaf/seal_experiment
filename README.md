SEAL Watermark Experiment Reproduction

This repository contains a reproduction of the SEAL watermarking experiments from the ICCV 2024 paper. The goal is to replicate the reported detection performance under common image transformations using a clean and reproducible pipeline.

Quick start:

git clone https://github.com/yourusername/seal-watermark-experiment.git
cd seal-watermark-experiment

chmod +x setup.sh run_experiment.sh
./setup.sh

./run_experiment.sh

Running the full experiment takes approximately 20 hours on an RTX 4000 GPU.

Expected results (approximate):

Transform      AUC     TPR @ 1% FPR
Original       0.999   99.0%
Brightness     0.961   83.0%
JPEG_25        0.870   52.2%

Minor variations are expected depending on hardware, random seeds, and library versions.

Repository structure:

seal-watermark-experiment/
├── config/        experiment configuration
├── src/           source code
├── scripts/       experiment pipeline
├── results/       generated outputs
└── notebooks/     analysis notebooks

Configuration:

Experiment settings can be modified in config/experiment_config.yaml. This includes the number of watermarked and negative images, watermark parameters such as number of patches and bits per patch, and the set of image transformations to evaluate.

Citation:

@inproceedings{arabi2024seal,
  title={SEAL: Semantic Aware Image Watermarking},
  author={Arabi, Kasra and Witter, R. Teal and Hegde, Chinmay and Cohen, Niv},
  booktitle={ICCV},
  year={2024}
}

Contact:

For questions or issues, open a GitHub issue or contact: your.email@nyu.edu
