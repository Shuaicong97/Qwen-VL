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
import multiprocessing as mp
import time

# Experiment 1: video with ground truth bbox (one object each time) + query => Qwen-VL => (s, e, score)
# -----------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------
def build_event_prompt(query, obj_id):
    return f"""
You are checking whether an event occurs for object {obj_id} in the video.

Event: "{query}"

Bounding boxes for this object are provided for each frame.

If the event happens for this object:
output: Yes, (start_frame, end_frame, confidence_score)

If the event does NOT happen:
output: No, it does not happen.

Only output ONE line. No explanation.
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
# Build Qwen-VL message
# -----------------------------------------------------------
def build_messages(video_path, query, obj_id, boxes, video_info):
    content = [
        {
            "type": "video",
            "video": f"file://{video_path}",
            "max_pixels": video_info["scaled_size"][0] * video_info["scaled_size"][1],
            "fps": 1.0
        },
        {
            "type": "text",
            "text": build_event_prompt(query, obj_id)
        }
    ]

    # Add bounding boxes for each frame
    for frame_idx, box in boxes.items():
        content.append({
            "type": "box",
            "box": box,
            "frame_index": frame_idx,
            "label": f"object {obj_id}"
        })

    return [{"role": "user", "content": content}]


def sub_processor(pid, task_list, args, query_dict, VIDEO_DIR, OUTPUT_DIR, GT_DIR, progress_queue):
    torch.cuda.set_device(pid)
    device = f"cuda:{pid}"

    # Load model & processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    for video_name, obj_id in task_list:
        video_path = os.path.join(VIDEO_DIR, video_name)
        query_list = query_dict[video_name]

        # ---------- GT path ----------
        seq_name = video_name.split('_')[0]  # MOT17-07_0.0_500.0 → MOT17-07
        if args.dataset.lower() == "ovis":
            gt_path = os.path.join(GT_DIR, seq_name, "gt.txt")
        else:
            gt_path = os.path.join(GT_DIR, seq_name, "gt_filtered.txt")

        # ---------- Video scaling info ----------
        video_info = get_scaled_video_info(video_path)

        # ---------- load GT boxes ----------
        boxes_by_obj = load_gt_boxes(gt_path, video_info["scale"])

        boxes = boxes_by_obj[obj_id]

        output_json_path = os.path.join(
            OUTPUT_DIR,
            f"{os.path.splitext(video_name)[0]}_obj{obj_id}.json"
        )

        # skip if exists
        if os.path.exists(output_json_path):
            logging.info(f"[SKIP] Already exist: {output_json_path}")
            progress_queue.put(1)
            continue

        logging.info(f"--- Object {obj_id} ---")

        obj_results = {}

        for query in query_list:
            messages = build_messages(
                video_path=video_path,
                query=query,
                obj_id=obj_id,
                boxes=boxes,
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
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=8192
                )

            out_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
            output_text = processor.decode(out_tokens, skip_special_tokens=True)
            obj_results[query] = output_text

            torch.cuda.empty_cache()

        # save result for this object
        with open(output_json_path, "w") as f:
            json.dump(obj_results, f, indent=2)
            print(f"Saved to {output_json_path}")

        progress_queue.put(1)

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Object-level event grounding with Qwen2.5-VL")
    parser.add_argument("--dataset", type=str, default="mot20", help="ovis / mot17 / mot20")
    parser.add_argument("--num_gpus", type=int, default=4)

    args = parser.parse_args()

    # =============== PATHS BY DATASET ==================
    if args.dataset.lower() == "ovis":
        VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/ovis_valid_videos_137_V1"
        QUERY_JSON = "data/queries_ovis.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/ovis_spatial_exp1"
        GT_DIR = "/nfs/data3/shuaicong/box_gt/OVIS_GTs/val"
    elif args.dataset.lower() == "mot17":
        VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/mot17_valid_V1"
        QUERY_JSON = "data/queries_mot17.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/mot17_spatial_exp1"
        GT_DIR = "/nfs/data3/shuaicong/box_gt/MOT/MOT17_GTs/val"
    elif args.dataset.lower() == "mot20":
        VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/mot20_valid_V1"
        QUERY_JSON = "data/queries_mot20.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/mot20_spatial_exp1"
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

    with open(QUERY_JSON, "r") as f:
        query_dict = json.load(f)

    all_tasks = []

    for video_name in query_dict.keys():
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            continue

        seq_name = video_name.split('_')[0]
        if args.dataset.lower() == "ovis":
            gt_path = os.path.join(GT_DIR, seq_name, "gt.txt")
        else:
            gt_path = os.path.join(GT_DIR, seq_name, "gt_filtered.txt")

        video_info = get_scaled_video_info(video_path)
        boxes_by_obj = load_gt_boxes(gt_path, video_info["scale"])

        for obj_id in boxes_by_obj.keys():
            all_tasks.append((video_name, obj_id))

    print(f"Total objects: {len(all_tasks)}")

    task_splits = [
        all_tasks[i::args.num_gpus]
        for i in range(args.num_gpus)
    ]
    manager = mp.Manager()
    progress_queue = manager.Queue()
    print(f"GPU used: {args.num_gpus}")


    total_objs = len(all_tasks)
    mp.set_start_method("spawn", force=True)
    processes = []

    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=sub_processor,
            args=(
                gpu_id,
                task_splits[gpu_id],
                args,
                query_dict,
                VIDEO_DIR,
                OUTPUT_DIR,
                GT_DIR,
                progress_queue,  # 👈 传进去
            )
        )
        p.start()
        processes.append(p)

    with tqdm(total=total_objs, desc="Objects completed") as pbar:
        finished = 0
        while finished < total_objs:
            progress_queue.get()  # 阻塞，直到有 obj 完成
            finished += 1
            pbar.update(1)

    for p in processes:
        p.join()

    print("Over")

if __name__ == "__main__":
    main()
