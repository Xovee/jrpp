# JRPP

A reference implementation for *Learning to Curate Context: Jointly Optimizing Retrieval and Prediction for Multimodal Social Media Popularity* (AAAI 2026).

## Installation

Use Python 3.12 and PyTorch 2.7.1. The following installs the CUDA 12.8 build:

```bash
conda create -n jrpp python=3.12
conda activate jrpp
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Data

Download `skapp-icip.zip`, `skapp-smpd.zip`, or `skapp-instagram.zip` from
[Google Drive](https://drive.google.com/drive/folders/1VbRgLHxWpCj_JOKLxdrqd_ZeNzyCZheq).
These are the same data packages used by [SKAPP](https://github.com/Xovee/skapp).
Keep the ZIP intact and run from the repository root:

```bash
python src/prepare_data.py --dataset icip --archive /path/to/skapp-icip.zip --output data/icip
```

Replace `icip` with `smpd` or `instagram` for the other datasets. See
[data/README.md](data/README.md) for the input fields and fixed split sizes.

## Implementation

The model jointly learns retrieval, context filtering and popularity prediction.
The default configuration uses the following implementation choices:

- **Retrieval:** shortlist 100 training samples using mean component similarity,
  then rerank them with the full Mixture-of-Logits (MoL) score to retain 50 candidates.
- **Selection:** score candidates using their representations, the query and the
  retrieval scores. During training, Gumbel noise produces a hard top-40 selection
  without replacement; a straight-through soft surrogate supplies gradients.
  Evaluation uses deterministic top-40 selection.
- **Information bottleneck:** compress each selected candidate with a
  query-conditioned Gaussian encoder. KL terms are summed over latent dimensions
  and averaged over selected candidates and the batch before applying `beta`.
  Training samples the latent representations; evaluation uses their means.
- **Prediction:** retain each compressed neighbor as an attention token. The query
  attends to these neighbors with retrieval scores as an attention prior, followed
  by a regression head.

Model settings are in [src/config/config.yaml](src/config/config.yaml).

## Training

```bash
python src/train.py --data-name icip --run-name jrpp
```

Each command trains once with seed 12. Use `--seed` to choose another seed;
for multiple runs, repeat the command with the desired seeds (e.g. 12, 22, 32).
Replace `icip` with `smpd` or `instagram` to train on either dataset.

Authors use a train-only vocabulary with a separate embedding row per author
and a zero row for unseen authors, supporting up to 100,000 training authors.
Checkpoints include the vocabulary for resuming and evaluation.

Outputs are saved under `results/<dataset>/<run-name>/`. Existing run directories
are preserved by appending a numeric suffix. Choose another output root with
`--output-dir runs/experiment-name`.

The best checkpoint is selected by validation MSE using plain prediction.
`JRPP_best.pt` contains weights and configuration for evaluation;
`checkpoints/JRPP_last.pt` contains the complete training state for resuming.
To resume at the next epoch, repeat the training arguments, omit `--run-name`,
and add `--resume-path results/icip/jrpp/checkpoints/JRPP_last.pt`.
`--epochs` specifies the total epoch count, including completed epochs.
Use `--keep-epoch-checkpoints` only if every epoch's checkpoint is needed.

## Evaluation

```bash
python src/test.py --model-path results/icip/jrpp/JRPP_best.pt --output-dir results/icip-evaluation
```

The checkpoint supplies the dataset name and model settings. Evaluation verifies
that the prepared data matches the training data. Use `--data-dir` if the datasets
are outside `data/`.

Testing defaults to four perturbations (two positive/negative pairs) of the image
and text features, with noise norm 0.02 times each feature's norm. The final prediction
combines the original prediction and the mean of the perturbed predictions with
equal weights. Use `--no-tta` for plain prediction:

```bash
python src/test.py --model-path results/icip/jrpp/JRPP_best.pt --no-tta --output-dir results/icip-evaluation
```

Metrics, per-sample predictions and labels, and evaluation settings are saved
separately for plain and TTA evaluation. Use a new `--output-dir` when comparing
different checkpoints. The retrieval bank is cached during evaluation; use
`--no-retrieval-cache` to disable caching.


## Configuration

Method settings are in `src/config/config.yaml`. Training and evaluation options
are available through `python src/train.py --help` and `python src/test.py --help`.

## Citation

```bibtex
@inproceedings{xu2026learning,
  title = {Learning to Curate Context: Jointly Optimizing Retrieval and Prediction for Multimodal Social Media Popularity},
  author = {Xovee Xu and Shuojun Lin and Fan Zhou and Jingkuan Song},
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
  year = {2026},
  volume = {40},
  number = {2},
  month = {jan},
  numpages = {9},
  pages = {1382--1390},
  publisher = {AAAI},
  doi = {10.1609/aaai.v40i2.37112}
}
```

## License

MIT

## Contact

`xovee at uestc.edu.cn`
