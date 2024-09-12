
# Beyond Uncertainty: Evidential Deep Learning for Robust Video Temporal Grounding

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
  <img src="https://img.shields.io/badge/python-3.10-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/conda-4.10.3-blue" alt="Conda Version">
  <a href="https://arxiv.org/abs/2408.16272"><img src="https://img.shields.io/badge/arXiv-2408.16272-red" alt="arXiv"></a>
<img src="https://komarev.com/ghpvc/?username=KaijingOfficial&repo=sram_vtg" alt="GitHub Views">
</p>

## 💡 News
- **[2024-08-30]:** Release codebase.  

## 🚀Dataset Preparation

Follow the instructions in the [Dataset section](https://github.com/showlab/UniVTG/blob/main/install.md#datasets) to set up the datasets.

## 🔨Environment Setup

```
conda create --name sram python=3.10
conda activate sram
pip install -r requirements.txt
```

## 📖Extract Nouns

```
python data_prepross/detect_words.py \
--input_json_path /path/to/meta_jsonl \
--output_json_path /path/to/meta_jsonl
```

## 🔥Training

### Train on QVHighlights

```
bash scripts/train/qvhl_ddp_2stage_train.sh
```

### Train on Charades

```
bash scripts/train/charades_ddp_2stage_train.sh
```

## 🤖Inference

### Inference on our checkpoint

```
bash scripts/inference/inf_sram_base_ddp.sh
```

## 🤝Citation

If you find our work useful, please cite:

```
@article{ma2024beyond,
  title={Beyond Uncertainty: Evidential Deep Learning for Robust Video Temporal Grounding},
  author={Ma, Kaijing and Huang, Haojian and Chen, Jin and Chen, Haodong and Ji, Pengliang and Zang, Xianghao and Fang, Han and Ban, Chao and Sun, Hao and Chen, Mulin and others},
  journal={arXiv preprint arXiv:2408.16272},
  year={2024}
}
```
