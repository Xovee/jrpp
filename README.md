# JRPP

A reference implementation for *Learning to Curate Context: Jointly Optimizing Retrieval and Prediction for Multimodal Social Media Popularity* (AAAI 2026).


## Repo Layout

```text
data/                 # ICIP, SMPD, Instagram
data/README.md
src/config/config.yaml
src/models/
src/utils/
src/train.py
src/test.py
requirements.txt
```

See [data/README.md](data/README.md) for more data descriptions. Datasets download link: [Google Drive](https://drive.google.com/drive/folders/1LQqt9WYpjh8RcrkUGWIw4cqpt17n_53z?usp=sharing).

## Environment

Use the project Conda environment:

```shell
conda create -n jrpp python=3.12
conda activate jrpp
```

Install PyTorch and required packages:

```shell
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

The PyTorch version, CUDA version, and random seed may affect model performance. Please use the recommended environment for reproducibility.
Verify the environment:

```shell
python -c "import torch, pandas; print(torch.__version__); print(torch.cuda.is_available())"
```

## Training


```shell
python src/train.py --data-name icip --run-name jrpp
python src/train.py --data-name smpd --run-name jrpp
python src/train.py --data-name instagram --run-name jrpp
```

## Evaluation

```shell
python src/test.py --data-name icip --model-path results/icip/jrpp/JRPP_best.pt
python src/test.py --data-name smpd --model-path results/smpd/jrpp/JRPP_best.pt
python src/test.py --data-name instagram --model-path results/instagram/jrpp/JRPP_best.pt
```


## Configuration

Most method parameters live in `src/config/config.yaml`. For training and evaluation parameters, run `python src/train.py --help` or `python src/test.py --help`.


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
