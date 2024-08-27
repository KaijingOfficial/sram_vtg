
# Beyond Uncertainty: Evidential Deep Learning for Robust Video Temporal Grounding


## Dataset Preparing

Please find instructions in [Dataset](https://github.com/showlab/UniVTG/blob/main/install.md#datasets) to setup datasets.


## Environment Setup

```
conda create --name sram python=3.10
conda activate sram
pip install -r requirements.txt
```

## Extract Nouns


```
python data_prepross/detect_words.py \
--input_json_path /path/to/meta_jsonl \
--output_json_path /path/to/meta_jsonl

```

## Train

### Train on QVHighlights
```
bash scripts/train/qvhl_ddp_2stage_train.sh
```

### Train on Charades

```
bash scripts/train/charades_ddp_2stage_train.sh
```


## Inference

### Inference on our checkpoint:
```
bash scripts/inference/inf_sram_base_ddp.sh
```


