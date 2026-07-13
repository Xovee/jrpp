# Datasets

This directory contains the compact datasets used directly by JRPP:

```text
icip/{train,val,test}.pkl
smpd/{train,val,test}.pkl
instagram/{train,val,test}.pkl
```

Each split is a pandas DataFrame with only the fields needed by the model.

| Dataset | Columns |
| --- | --- |
| ICIP | `image_id`, `user_id`, `label`, `merged_text_vec`, `cls_vec`, `mean_views` |
| SMPD | `image_id`, `user_id`, `label`, `merged_text_vec`, `cls_vec` |
| Instagram | `image_id`, `user_id`, `label`, `merged_text_vec`, `cls_vec` |

`merged_text_vec` and `cls_vec` are 768-d `float32` NumPy arrays. `label` is the log-transformed popularity target.

The original full datasets include raw text, tags, timestamps, linguistic fields, and precomputed retrieval fields. They are not required for JRPP training or evaluation.

Full dataset download link: [Google Drive](https://drive.google.com/drive/folders/1PgR7c6-n6tAc-XQxsqf6SFX-wcipyy6N?usp=sharing)
