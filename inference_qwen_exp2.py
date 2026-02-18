import os
from tqdm import tqdm
import logging
from datetime import datetime
import json
import argparse
import torch
from pathlib import Path
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# -----------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------
def build_multi_object_prompt(query):
    return f"""
You are checking whether an event occurs for multiple objects in the video.

Event: "{query}"

Bounding boxes for all objects are provided for each frame.

For each object, output EXACTLY one line:

If the event happens:
object <obj_id>: Yes, (start_frame, end_frame, confidence_score)
or if it does NOT happen:
object <obj_id>: No, it does not happen.

IMPORTANT RULES:
- Output results ONLY for the object IDs that appear in the bounding boxes.
- Do NOT create additional object IDs (do NOT invent object 0 or new objects).
- You MUST output one line for EVERY object with bounding boxes.
- Do NOT skip objects.
- Each object has AT MOST one (start,end,score).
- No explanation.
"""


# -----------------------------------------------------------
# Resize video preprocessing info
# -----------------------------------------------------------
def get_scaled_video_info(video_path, max_long_side=420, min_short_side=240):
    cap = cv2.VideoCapture(video_path)
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    scale = min(max_long_side / max(orig_W, orig_H), 1.0)
    scaled_W = int(orig_W * scale)
    scaled_H = int(orig_H * scale)

    if min(scaled_W, scaled_H) < min_short_side:
        scale = min_short_side / min(scaled_W, scaled_H)
        scaled_W = int(scaled_W * scale)
        scaled_H = int(scaled_H * scale)

    return {
        "orig_size": [orig_W, orig_H],
        "scaled_size": [scaled_W, scaled_H],
        "scale": scale
    }


# -----------------------------------------------------------
# Load GT boxes and scale them
# -----------------------------------------------------------
def load_gt_boxes(gt_path, scale):
    """
    Return:
    boxes_by_obj = {
        object_id: {
            frame: [x1, y1, x2, y2]
        }
    }
    """
    boxes_by_obj = {}

    with open(gt_path, "r") as f:
        for line in f:
            frame, obj_id, x, y, w, h, score, _, _ = line.strip().split(",")
            frame = int(frame)
            obj_id = int(obj_id)
            x, y, w, h = map(float, (x, y, w, h))

            # scale
            x *= scale
            y *= scale
            w *= scale
            h *= scale

            box = [x, y, x + w, y + h]

            if obj_id not in boxes_by_obj:
                boxes_by_obj[obj_id] = {}

            boxes_by_obj[obj_id][frame] = box

    return boxes_by_obj


# -----------------------------------------------------------
# Build messages for ALL objects together
# -----------------------------------------------------------
def build_multi_object_messages(video_path, query, boxes_by_obj, video_info):
    """
    boxes_by_obj: { obj_id: {frame: [x1,y1,x2,y2], ...}, ... }
    """
    content = [
        {
            "type": "video",
            "video": f"file://{video_path}",
            "max_pixels": video_info["scaled_size"][0] * video_info["scaled_size"][1],
            "fps": 1.0
        },
        {
            "type": "text",
            "text": build_multi_object_prompt(query)
        }
    ]

    # Add bounding boxes for each frame
    for obj_id, frames in boxes_by_obj.items():
        for frame_idx, box in frames.items():
            content.append({
                "type": "box",
                "box": box,
                "frame_index": frame_idx,
                "label": f"object {obj_id}"
            })

    return [{"role": "user", "content": content}]


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Object-level event grounding with Qwen2.5-VL")
    parser.add_argument("--dataset", type=str, default="ovis", help="ovis / mot17 / mot20")
    args = parser.parse_args()

    # =============== PATHS BY DATASET ==================
    if args.dataset.lower() == "ovis":
        VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/ovis_valid_videos_137_V1"
        QUERY_JSON = "data/queries_ovis.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/ovis_spatial_exp2"
        GT_DIR = "/nfs/data3/shuaicong/box_gt/OVIS_GTs/val"
    elif args.dataset.lower() == "mot17":
        VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/mot17_valid_V1"
        QUERY_JSON = "data/queries_mot17.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/mot17_spatial_exp2"
        GT_DIR = "/nfs/data3/shuaicong/box_gt/MOT/MOT17_GTs/val"
    elif args.dataset.lower() == "mot20":
        VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/mot20_valid_V1"
        QUERY_JSON = "data/queries_mot20.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/mot20_spatial_exp2"
        GT_DIR = "/nfs/data3/shuaicong/box_gt/MOT/MOT20_GTs/val"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # logging
    log_filename = os.path.join(
        OUTPUT_DIR,
        f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )

    # Load model & processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    with open(QUERY_JSON, "r") as f:
        query_dict = json.load(f)

    # =================== PROCESS VIDEOS ===========================
    for video_idx, (video_name, query_list) in enumerate(
            tqdm(query_dict.items(), desc="Videos"), start=1):

        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            logging.warning(f"Video not found: {video_path}")
            continue

        logging.info(f"===== Video {video_idx}: {video_name} =====")

        # ---------- GT path ----------
        seq_name = video_name.split('_')[0]  # MOT17-07_0.0_500.0 → MOT17-07
        gt_path = os.path.join(GT_DIR, seq_name, "gt.txt")

        if not os.path.exists(gt_path):
            logging.warning(f"GT not found: {gt_path}")
            continue

        # ---------- Video scaling info ----------
        video_info = get_scaled_video_info(video_path)

        # ---------- load GT boxes ----------
        boxes_by_obj = load_gt_boxes(gt_path, video_info["scale"])

        # NEW: For each query, generate one result file for ALL objects
        for q_idx, query in enumerate(query_list, start=1):
            logging.info(f"--- Query {q_idx}: {query} ---")

            output_json_path = os.path.join(
                OUTPUT_DIR,
                f"{os.path.splitext(video_name)[0]}_query{q_idx}.json"
            )

            # skip if exists
            if os.path.exists(output_json_path):
                logging.info(f"[SKIP] Already exist: {output_json_path}")
                continue

            # --- Build messages for all objects ---
            messages = build_multi_object_messages(
                video_path=video_path,
                query=query,
                boxes_by_obj=boxes_by_obj,  # ALL OBJECTS
                video_info=video_info
            )

            # --- prepare input ---
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

            # --- inference ---
            generated_ids = model.generate(
                **inputs, max_new_tokens=8192
            )

            out_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
            output_text = processor.decode(out_tokens, skip_special_tokens=True)
            logging.info(f"[Qwen Output]\n{output_text}")

            # ---------------------------------------------
            # You now get ALL objects’ result in output_text
            # ---------------------------------------------
            # Example output:
            # object 1: Yes, (3, 15, 0.92)
            # object 2: No, it does not happen.
            # object 5: Yes, (7, 29, 0.88)

            parsed_results = {
                "query": query,  # <--- 保存 query
                "raw_output": output_text,  # <--- 保存 output_text 原文（便于对照）
                "objects": {}  # <--- 存放每个 object 的结果
            }

            for line in output_text.strip().split("\n"):
                line = line.strip()
                if line.startswith("object"):
                    try:
                        obj_id = int(line.split()[1].replace(":", ""))
                        parsed_results["objects"][obj_id] = line
                    except:
                        logging.warning(f"Cannot parse line: {line}")

            # save results
            with open(output_json_path, "w") as f:
                json.dump(parsed_results, f, indent=2)

            logging.info(f"[SAVE] {output_json_path}")

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
