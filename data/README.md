# Datasets

Download a dataset ZIP from
[Google Drive](https://drive.google.com/drive/folders/1VbRgLHxWpCj_JOKLxdrqd_ZeNzyCZheq).
JRPP uses the same published packages as [SKAPP](https://github.com/Xovee/skapp):
`skapp-icip.zip`, `skapp-smpd.zip`, and `skapp-instagram.zip`.

From the repository root:

```bash
python src/prepare_data.py --dataset icip --archive /path/to/skapp-icip.zip --output data/icip
```

Replace `icip` with `smpd` or `instagram` for the other datasets. No separate JRPP
archive or pickle files are needed. Preparation creates:

```text
data/icip/
  dataset.json
  train.npz
  valid.npz
  test.npz
```

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| ICIP | 16,269 | 2,034 | 2,034 |
| SMPD | 244,490 | 30,561 | 30,562 |
| Instagram | 238,610 | 29,826 | 29,827 |

The original sample order and train/validation/test membership are retained;
training never resplits the data. Preparation verifies the archive's split
checksums, counts, and ordered IDs and rejects duplicates and overlapping splits.
Existing output directories are never overwritten, and failed preparation does
not publish partial data.

Each split contains string `image_id` and `user_id`, the published `label`, and
the 768-dimensional `cls_vec` and `merged_text_vec` features. ICIP additionally
contains raw `mean_views`; SMPD and Instagram have no extra numeric metadata in
JRPP. The shared feature arrays are copied unchanged. Labels and numeric metadata
are cast to float32 for training, without further label transformation or
standardization. No new feature extraction is required.

Full identifiers are preserved. Numeric IDs are used directly where possible;
other IDs are encoded with a deterministic hash, with collision checks. Only the
training split forms the retrieval bank; training queries exclude themselves.

`dataset.json` records the source archive hash, split counts, sample order and
file checksums. Loading verifies all splits before use. Checkpoints retain the
data fingerprint, so evaluation and resuming reject different prepared data.
