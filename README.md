
# Beyond Uncertainty: Evidential Deep Learning for Robust Video Temporal Grounding

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
  <img src="https://img.shields.io/badge/python-3.10-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/conda-4.10.3-blue" alt="Conda Version">
</p>

## Dataset Preparation

Follow the instructions in the [Dataset section](https://github.com/showlab/UniVTG/blob/main/install.md#datasets) to set up the datasets.

## Environment Setup

<pre style="background-color: rgb(43, 43, 43);margin-right: 15px;"><div class="pre-code-area"><code class="language-javascript" style="white-space: pre-wrap;">bash
conda create --name sram python=3.10
conda activate sram
pip install -r requirements.txt
</code></div></pre>

## Extract Nouns

<pre style="background-color: rgb(43, 43, 43);margin-right: 15px;"><div class="pre-code-area"><code class="language-javascript" style="white-space: pre-wrap;">bash
python data_prepross/detect_words.py \
--input_json_path /path/to/meta_jsonl \
--output_json_path /path/to/meta_jsonl
</code></div></pre>

## Training

### Train on QVHighlights

<pre style="background-color: rgb(43, 43, 43);margin-right: 15px;"><div class="pre-code-area"><code class="language-javascript" style="white-space: pre-wrap;">bash
bash scripts/train/qvhl_ddp_2stage_train.sh
</code></div></pre>

### Train on Charades

<pre style="background-color: rgb(43, 43, 43);margin-right: 15px;"><div class="pre-code-area"><code class="language-javascript" style="white-space: pre-wrap;">bash
bash scripts/train/charades_ddp_2stage_train.sh
</code></div></pre>

## Inference

### Inference on our checkpoint

<pre style="background-color: rgb(43, 43, 43);margin-right: 15px;"><div class="pre-code-area"><code class="language-javascript" style="white-space: pre-wrap;">bash
bash scripts/inference/inf_sram_base_ddp.sh
</code></div></pre>

## Reference

If you find our work useful, please cite:

<pre style="background-color: rgb(43, 43, 43);margin-right: 15px;"><div class="pre-code-area"><code class="language-javascript" style="white-space: pre-wrap;">bibtex
@article{ma2024beyond,
  title={Beyond Uncertainty: Evidential Deep Learning for Robust Video Temporal Grounding},
  author={Ma, Kaijing and Huang, Haojian and Chen, Jin and Chen, Haodong and Ji, Pengliang and Zang, Xianghao and Fang, Han and Ban, Chao and Sun, Hao and Chen, Mulin and others},
  journal={arXiv preprint arXiv:2408.16272},
  year={2024}
}
</code></div></pre>

---

————————————————————————————————————
