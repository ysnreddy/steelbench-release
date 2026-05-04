#!/usr/bin/env python3
"""AWS Bedrock Batch Inference runner for SteelBench evaluation.

Prepares JSONL with S3 image URLs, uploads to S3, submits batch job.
Bypasses real-time RPM throttle limits.

Usage:
    # Submit batch for Claude Sonnet
    python eval_runners/bedrock_batch.py --model claude_sonnet --submit

    # Submit all pending Bedrock models
    python eval_runners/bedrock_batch.py --model all --submit

    # Check status
    python eval_runners/bedrock_batch.py --status

    # Download results
    python eval_runners/bedrock_batch.py --download
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from annotation_tool.vlm_annotator import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    normalize_annotation,
    parse_json_response,
    validate_and_fix_response,
    validate_response_completeness,
)
from annotation_tool.safety_rules import (
    build_safety_prompt_block,
    load_camera_zones,
    load_safety_rules,
)

EVAL_DIR = Path(__file__).parent.parent / "eval_data"
FRAMES_DIR = EVAL_DIR / "frames"
RESULTS_DIR = EVAL_DIR / "results"
CONFIG_DIR = EVAL_DIR / "config"

S3_BUCKET = "steelbench-eval-frames"
S3_BATCH_PREFIX = "batch-jobs"

BEDROCK_MODELS = {
    "claude_sonnet": {
        "model_id": "us.anthropic.claude-sonnet-4-6",
        "content_type": "application/json",
    },
    "claude_opus": {
        "model_id": "us.anthropic.claude-opus-4-6-v1",
        "content_type": "application/json",
    },
    "pixtral_large": {
        "model_id": "us.mistral.pixtral-large-2502-v1:0",
        "content_type": "application/json",
    },
}


def get_session():
    """Create boto3 session."""
    import boto3
    load_dotenv()
    return boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def get_completed_clips(model_name):
    """Get clip IDs already successfully completed."""
    output_path = RESULTS_DIR / f"{model_name}.jsonl"
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("success"):
                        done.add(r.get("clip_id", ""))
                except:
                    pass
    return done


def prepare_batch(model_name):
    """Prepare batch JSONL for Bedrock batch inference."""
    if model_name not in BEDROCK_MODELS:
        print(f"Unknown model: {model_name}")
        return None

    config = BEDROCK_MODELS[model_name]
    model_id = config["model_id"]

    clips = json.load(open(EVAL_DIR / "gt_clips.json"))
    done = get_completed_clips(model_name)
    remaining = [c for c in clips if c not in done]

    print(f"  {model_name}: {len(done)} done, {len(remaining)} remaining")

    if not remaining:
        print("  All clips done!")
        return None

    # Load safety rules
    try:
        rules_config = load_safety_rules(str(CONFIG_DIR))
        camera_zones = load_camera_zones(str(CONFIG_DIR))
    except:
        rules_config = None
        camera_zones = None

    # Load manifest
    manifest = {}
    manifest_path = CONFIG_DIR / "batch_manifest.csv"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                manifest[row.get("clip_id", "")] = row

    # Build JSONL — Bedrock Converse API format
    batch_lines = []
    for clip_id in remaining:
        frame_dir = FRAMES_DIR / clip_id
        if not frame_dir.exists():
            continue
        frames = sorted(frame_dir.glob("frame_*.jpg"))
        if len(frames) != 8:
            continue

        # Safety block
        row = manifest.get(clip_id, {})
        safety_block = None
        if rules_config and row.get("site"):
            try:
                safety_block = build_safety_prompt_block(
                    row.get("camera_id", ""), row.get("site", ""),
                    row.get("work_area", ""),
                    rules_config=rules_config, camera_zones=camera_zones,
                )
            except:
                pass

        prompt_text = USER_PROMPT
        if safety_block:
            prompt_text = USER_PROMPT + "\n\n" + safety_block

        # Build content with S3 image URLs
        content = [{"text": prompt_text}]
        for fp in frames:
            s3_url = f"s3://{S3_BUCKET}/{clip_id}/{fp.name}"
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {"s3Location": {"uri": s3_url}},
                }
            })

        # Bedrock batch record format
        record = {
            "recordId": clip_id,
            "modelInput": {
                "messages": [{"role": "user", "content": content}],
                "system": [{"text": SYSTEM_PROMPT}],
                "inferenceConfig": {
                    "maxTokens": 6000,
                    "temperature": 0.1,
                },
            },
        }
        batch_lines.append(json.dumps(record))

    if not batch_lines:
        print("  No valid clips to process")
        return None

    # Write to local file
    batch_path = RESULTS_DIR / f"{model_name}_bedrock_batch.jsonl"
    with open(batch_path, "w") as f:
        f.write("\n".join(batch_lines))

    size_mb = batch_path.stat().st_size / 1024 / 1024
    print(f"  Batch file: {batch_path} ({size_mb:.1f} MB, {len(batch_lines)} records)")
    return batch_path


def submit_batch(model_name, batch_path):
    """Upload batch file to S3 and submit Bedrock batch job."""
    session = get_session()
    s3 = session.client("s3")
    bedrock = session.client("bedrock")

    config = BEDROCK_MODELS[model_name]
    model_id = config["model_id"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Upload input to S3
    s3_input_key = f"{S3_BATCH_PREFIX}/{model_name}/input_{timestamp}.jsonl"
    s3_output_prefix = f"{S3_BATCH_PREFIX}/{model_name}/output_{timestamp}/"

    print(f"  Uploading to s3://{S3_BUCKET}/{s3_input_key}...")
    s3.upload_file(str(batch_path), S3_BUCKET, s3_input_key)

    # Create IAM role ARN for Bedrock (use the account's default)
    # Bedrock needs permission to read from S3 and write results
    account_id = session.client("sts").get_caller_identity()["Account"]

    # Submit batch job
    print(f"  Submitting batch job for {model_id}...")
    try:
        response = bedrock.create_model_invocation_job(
            jobName=f"steelbench-{model_name}-{timestamp}",
            modelId=model_id,
            roleArn=f"arn:aws:iam::{account_id}:role/BedrockBatchRole",
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3Uri": f"s3://{S3_BUCKET}/{s3_input_key}",
                    "s3InputFormat": "JSONL",
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": f"s3://{S3_BUCKET}/{s3_output_prefix}",
                }
            },
        )
        job_arn = response["jobArn"]
        print(f"  Job ARN: {job_arn}")
        print(f"  Status: Submitted")

        # Save job info
        job_info = {
            "model_name": model_name,
            "model_id": model_id,
            "job_arn": job_arn,
            "s3_input": f"s3://{S3_BUCKET}/{s3_input_key}",
            "s3_output": f"s3://{S3_BUCKET}/{s3_output_prefix}",
            "timestamp": timestamp,
        }
        job_file = RESULTS_DIR / f"{model_name}_bedrock_job.json"
        with open(job_file, "w") as f:
            json.dump(job_info, f, indent=2)
        print(f"  Job info saved to {job_file}")
        return job_arn

    except Exception as e:
        print(f"  ERROR: {e}")
        # Common error: need to create BedrockBatchRole
        if "role" in str(e).lower():
            print(f"\n  You need to create an IAM role 'BedrockBatchRole' with:")
            print(f"    - Trust: bedrock.amazonaws.com")
            print(f"    - Policy: AmazonS3FullAccess + AmazonBedrockFullAccess")
            print(f"  Or use an existing role ARN.")
        return None


def check_status():
    """Check status of all Bedrock batch jobs."""
    session = get_session()
    bedrock = session.client("bedrock")

    print("=== Bedrock Batch Jobs ===")
    jobs = bedrock.list_model_invocation_jobs(maxResults=20)
    for job in jobs.get("invocationJobSummaries", []):
        name = job.get("jobName", "?")
        status = job.get("status", "?")
        model = job.get("modelId", "?")
        arn = job.get("jobArn", "")
        print(f"  {name}  {status}  {model}")

        if status == "Completed":
            output = job.get("outputDataConfig", {}).get("s3OutputDataConfig", {}).get("s3Uri", "")
            print(f"    Output: {output}")

    # Also check local job files
    print()
    for jf in RESULTS_DIR.glob("*_bedrock_job.json"):
        info = json.load(open(jf))
        job_arn = info.get("job_arn", "")
        if job_arn:
            try:
                job = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
                status = job.get("status", "?")
                print(f"  {info['model_name']}: {status} (ARN: {job_arn[:60]}...)")
            except:
                print(f"  {info['model_name']}: Could not retrieve status")


def download_results(model_name=None):
    """Download and parse batch results from S3."""
    session = get_session()
    s3 = session.client("s3")
    bedrock = session.client("bedrock")

    # Find completed jobs
    job_files = list(RESULTS_DIR.glob("*_bedrock_job.json"))
    if model_name:
        job_files = [f for f in job_files if model_name in f.name]

    for jf in job_files:
        info = json.load(open(jf))
        mname = info["model_name"]
        job_arn = info.get("job_arn", "")

        if not job_arn:
            continue

        # Check job status
        try:
            job = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
            status = job.get("status", "")
        except:
            print(f"  {mname}: Could not get job status")
            continue

        if status != "Completed":
            print(f"  {mname}: {status} (not ready)")
            continue

        # Get output location
        s3_output = info.get("s3_output", "")
        if not s3_output:
            continue

        print(f"  {mname}: Downloading results from {s3_output}...")

        # Parse S3 URI
        parts = s3_output.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        # List output files
        output_path = RESULTS_DIR / f"{mname}.jsonl"
        success = 0
        failed = 0

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".jsonl"):
                    continue

                response = s3.get_object(Bucket=bucket, Key=key)
                body = response["Body"].read().decode("utf-8")

                with open(output_path, "a") as out_f:
                    for line in body.strip().split("\n"):
                        try:
                            result = json.loads(line)
                            record_id = result.get("recordId", "")
                            model_output = result.get("modelOutput", {})

                            # Extract text from Converse response
                            raw_text = ""
                            output_msg = model_output.get("output", {}).get("message", {})
                            for block in output_msg.get("content", []):
                                if "text" in block:
                                    raw_text += block["text"]

                            if not raw_text:
                                # Try alternate format
                                raw_text = json.dumps(model_output)

                            parsed = parse_json_response(raw_text)
                            if parsed:
                                parsed = validate_and_fix_response(parsed) or parsed
                                parsed, _ = validate_response_completeness(parsed)
                                normalized = normalize_annotation(parsed)

                                record = {
                                    "clip_id": record_id,
                                    "timestamp": datetime.now().isoformat(),
                                    "model": info["model_id"],
                                    "success": True,
                                    "parsed": parsed,
                                    "normalized": normalized,
                                    "raw_response": raw_text[:5000],
                                }
                                out_f.write(json.dumps(record) + "\n")
                                success += 1
                            else:
                                record = {
                                    "clip_id": record_id,
                                    "timestamp": datetime.now().isoformat(),
                                    "model": info["model_id"],
                                    "success": False,
                                    "error": "parse_fail",
                                    "raw_response": raw_text[:5000],
                                }
                                out_f.write(json.dumps(record) + "\n")
                                failed += 1
                        except Exception as e:
                            failed += 1

        print(f"    {success} success, {failed} failed")
        print(f"    Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Bedrock Batch Inference")
    parser.add_argument("--model", choices=list(BEDROCK_MODELS.keys()) + ["all"])
    parser.add_argument("--submit", action="store_true", help="Prepare and submit batch")
    parser.add_argument("--status", action="store_true", help="Check job status")
    parser.add_argument("--download", action="store_true", help="Download completed results")
    args = parser.parse_args()

    load_dotenv()

    if args.status:
        check_status()
    elif args.download:
        download_results(args.model)
    elif args.submit:
        if not args.model:
            print("--model required with --submit")
            return

        models = list(BEDROCK_MODELS.keys()) if args.model == "all" else [args.model]
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"  {model_name}")
            print(f"{'='*60}")
            batch_path = prepare_batch(model_name)
            if batch_path:
                submit_batch(model_name, batch_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
