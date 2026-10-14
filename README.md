# PRLF

Progressive Representation Learning Framework for multimodal sentiment analysis with incomplete modalities, implemented from *Progressive Representation Learning for Multimodal Sentiment Analysis with Incomplete Modalities*.

PRLF combines an Adaptive Modality Reliability Estimator (AMRE) with four progressive interaction steps, learned projection/residual decomposition, and residual denoising. It supports frame-level missingness and evaluation with any nonempty subset of language, audio, and vision.

This repository provides the author's PRLF implementation; components adapted from the DMD project are identified below and retain their original attribution.

## Installation

Python 3.10 is recommended. Install the appropriate CPU/CUDA version of PyTorch, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data and pretrained model

Obtain preprocessed MOSI/MOSEI datasets separately under their usage terms. Datasets, pretrained weights, experiment checkpoints, and manuscript files are not included.

The feature pickle must contain `train`, `valid`, and `test` dictionaries with:

- `text_bert`: (N, 3, T), containing token IDs, attention masks, and segment IDs.
- `audio`: (N, T, Da); `vision`: (N, T, Dv).
- `regression_labels`, `raw_text`, and `id`.

Audio/vision dimensions are inferred from training features. Negative sentiment scores become class 0; zero and positive scores become class 1.

Default data paths are `dataset/MOSI/Processed/aligned_50.pkl` and `dataset/MOSEI/Processed/aligned_50.pkl`. Override with `--data-path`. BERT defaults to the local `bert-base-uncased/` directory; use `--pretrained /path/to/bert-base-uncased` or `--pretrained bert-base-uncased` for a Hugging Face download.

## Usage

```bash
# Validate two tiny training epochs, checkpoint reload, and test on real data.
python run_prlf.py --smoke --data-path /path/to/aligned_50.pkl --pretrained /path/to/bert-base-uncased

# Train with BERT fine-tuning.
python run_prlf.py --dataset mosi --epochs 50 --pretrained bert-base-uncased

# Evaluate all modalities with the default 30% frame missingness.
python run_prlf.py --mode test --checkpoint pt/prlf.pth --pretrained bert-base-uncased

# Audio and vision only, without additional frame missingness.
python run_prlf.py --mode test --checkpoint pt/prlf.pth --missing-rate 0 --available av --pretrained bert-base-uncased

# Tests require no dataset, pretrained weights, or GPU.
python -m unittest discover -s tests -v
```

Use `--help` for device, seed, batch size, gradient accumulation, and loss settings. Checkpoints are selected using validation loss and written to `pt/` alongside JSON metrics. Default losses are `task + 0.5 * unimodal + 0.1 * phase`, with `gamma=0.8` and four interaction steps.

Exact parameter Fisher calculations are expensive with fine-tuned BERT. `--freeze-bert` reduces the cost and excludes frozen BERT parameters from Fisher. The smoke run freezes BERT for CPU validation. SIMS and full five-seed benchmark reproduction are not implemented/validated.

## Project structure

```text
PRLF/
├── run_prlf.py
├── prlf/
│   ├── model.py
│   ├── text_encoder.py
│   ├── data.py
│   ├── config.py
│   └── utils.py
├── configs/default.json
├── tests/test_prlf.py
├── IMPLEMENTATION.md
├── requirements.txt
└── LICENSE
```

## License and attribution

MIT license. The feature loader and BERT wrapper were adapted from the existing DMD project. Its copyright notice is preserved in [LICENSE](LICENSE). The PRLF model follows the supplied manuscript, with underspecified choices documented in [IMPLEMENTATION.md](IMPLEMENTATION.md).
