import os
from tqdm import tqdm
import logging
from datetime import datetime
import json
import argparse
import torch
from pathlib import Path
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Helper: Build Prompt
def build_grounding_prompt(query):
        return f"""
You are performing temporal grounding for a video.

Event: "{query}"

Please answer using EXACTLY the following format:
{query} from frame <start_frame> to <end_frame>

Only output the final answer in ONE line, no explanation.
"""

def get_scaled_video_info(video_path, max_long_side=420, min_short_side=240):
    cap = cv2.VideoCapture(video_path)
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 1️⃣ Scale based on the longest side
    scale = min(max_long_side / max(orig_W, orig_H), 1.0)
    scaled_W = int(orig_W * scale)
    scaled_H = int(orig_H * scale)

    # 2️⃣ Check if the shorter side is less than the threshold
    if min(scaled_W, scaled_H) < min_short_side:
        # Enlarged based on the shorter side
        scale = min_short_side / min(scaled_W, scaled_H)
        scaled_W = int(scaled_W * scale)
        scaled_H = int(scaled_H * scale)

    return {
        "orig_size": [orig_W, orig_H],
        "scaled_size": [scaled_W, scaled_H],
        "scale": scale
    }

def main():
    parser = argparse.ArgumentParser(description="STVG with Qwen2.5-VL")
    parser.add_argument("--dataset", type=str, default="ovis",
                        help="Which dataset to process: ovis / mot17 / mot20")
    args = parser.parse_args()

    # Dataset Config. Only for Inference
    if args.dataset.lower() == "ovis":
        VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/ovis_valid_videos_137_V1"
        QUERY_JSON = "data/queries_ovis.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/ovis_temporal"
    elif args.dataset.lower() == "mot17":
        VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/mot17_valid_V1"
        QUERY_JSON = "data/queries_mot17.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/mot17_temporal"
    elif args.dataset.lower() == "mot20":
        VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/mot20_valid_V1"
        QUERY_JSON = "data/queries_mot20.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/mot20_temporal"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Dataset: {args.dataset}")
    print(f"VIDEO_DIR: {VIDEO_DIR}")
    print(f"QUERY_JSON: {QUERY_JSON}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    log_filename = os.path.join(
        OUTPUT_DIR,
        f"process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info("===== Start Processing =====")

    # Load Model Default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
    )
    # Default Processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    with open(QUERY_JSON, "r") as f:
        query_dict = json.load(f)

    # for video_name, query_list in query_dict.items():
    for video_idx, (video_name, query_list) in enumerate(
            tqdm(query_dict.items(), desc="Processing Videos"), start=1):
        video_path = os.path.join(VIDEO_DIR, video_name)
        logging.info(f"=== Processing video {video_idx}/{len(query_dict)}: {video_name} ===")

        if not os.path.exists(video_path):
            logging.warning(f"[WARN] Video not found: {video_path}")
            continue

        output_json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_name)[0]}.json")
        if os.path.exists(output_json_path):
            with open(output_json_path, "r") as f:
                results = json.load(f)
            logging.info(f"[INFO] Loaded existing results for {video_name}")
        else:
            results = {}

        video_info = get_scaled_video_info(video_path, max_long_side=420, min_short_side=240)
        logging.info(f"[INFO] Video info: {video_info}")

        # for query in query_list:
        for q_idx, query in enumerate(
                tqdm(query_list, desc=f"Queries of {video_name}", leave=False), start=1):
            logging.info(f"--> Processing query {q_idx}/{len(query_list)}: {query}")

            if query in results:
                logging.info(f"[SKIP] Query '{query}' already processed.")
                continue

            # Construct messages for Qwen
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video",
                         "video": f"file://{video_path}",
                         "max_pixels": video_info["scaled_size"][0] * video_info["scaled_size"][1],
                         "fps": 1.0},
                        {"type": "text", "text": build_grounding_prompt(query)}
                    ],
                }
            ]

            # Prepare inputs
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs
            ).to("cuda")

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512
            )

            out_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
            output_text = processor.decode(out_tokens, skip_special_tokens=True)

            try:
                parsed = json.loads(output_text)
            except:
                logging.error("[ERROR] Model output invalid JSON, saving raw text.")
                parsed = output_text

            if isinstance(parsed, list):
                parsed = {
                    "orig_size": video_info["orig_size"],
                    "scaled_size": video_info["scaled_size"],
                    "instances": parsed
                }

            results[query] = parsed

            del inputs, image_inputs, video_inputs
            torch.cuda.empty_cache()

        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"[INFO] Saved results → {output_json_path}")

    logging.info("===== All Processing Finished =====")

if __name__ == "__main__":
    main()