# Reproducing SteelBench evaluation

This repository contains the reproduction-essential code for SteelBench,
a NeurIPS 2026 Datasets and Benchmarks track submission.

The dataset itself is hosted on Hugging Face:
  https://huggingface.co/datasets/steelbench/SteelBench

## Setup

```bash
git clone https://anonymous.4open.science/r/<anon-id>/  steelbench
cd steelbench

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull the dataset (or use the 50-clip sample inside the HF repo for a quick test)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='steelbench/SteelBench', repo_type='dataset',
                  local_dir='./data', allow_patterns=['sample/**', 'manifests/**'])
"
```

## Set API keys for evaluated VLMs

The eval pipeline calls remote VLM APIs. Add keys to `.env`:

```
DeepInfra_API_KEY_2=<your_key>
OPENAI_API_KEY=<your_key>
Claude_API_Key=<your_key>
AWS_ACCESS_KEY_ID=<your_key>
AWS_SECRET_ACCESS_KEY=<your_key>
AWS_DEFAULT_REGION=us-east-1
```

## Run a single VLM on the GT set

```bash
python eval_inference.py --model gemma4_31b --test    # 10-clip smoke test
python eval_inference.py --model gemma4_31b           # full 1,345 clips
```

Output: `eval_data/results/<model>.jsonl` (one record per clip).

## Compute metrics from inference results

```bash
python eval_metrics_compute.py
```

## Run the frame-density ablation (Section 7)

```bash
# Extract 15 frames at 1fps for the 150-clip ablation subset
python scripts/frame_density_run.py --extract-15 \
    --clips-dir data/clips \
    --output-dir eval_data/frames_15

# Run Gemma 4-31B at each density
for n in 1 2 4 15; do
  python scripts/frame_density_run.py --provider deepinfra --frames $n
done

# Submit GPT-4o batches (50% off via Batch API)
for n in 1 2 4 15; do
  python scripts/frame_density_run.py --provider openai --frames $n --batch-input
  python scripts/frame_density_run.py --provider openai --frames $n --batch-submit
done
```

## Run the audit protocol on your own annotations

See `audit/README.md`.

## Anonymize frames before public release

```bash
python scripts/anonymize_frames.py \
    --input data/frames \
    --output anonymized_frames \
    --ids-file manifests/gt_clips.json
```

## License

Code: Apache-2.0 (see `LICENSE`)
Data: CC-BY-NC 4.0 (see the dataset repo on Hugging Face)
