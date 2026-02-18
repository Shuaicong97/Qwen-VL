import os
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
You are a spatio-temporal video grounding model.
Your task is to find ALL objects in the video that satisfy the query.

Query: "{query}"

Output ONLY valid JSON.
Each item must have:
- "id": instance id starting from 1
- "start_frame": first frame where this instance satisfies the query
- "end_frame": last frame where it satisfies the query
- "boxes": list of strings, each string in the format:
    frame,id,x1,y1,width,height,1,1,1
The frame number starts from 1. id is the instance id.

Do NOT include any explanation, markdown, backticks, or extra text.
The JSON must be parsable by Python's json.loads().
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
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/ovis1"
        OUTPUT_JSON = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/ovis_outputs.json"
    elif args.dataset.lower() == "mot17":
        VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/mot17_valid_V1"
        QUERY_JSON = "data/queries_mot17.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/mot17"
        OUTPUT_JSON = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/mot17_outputs.json"
    elif args.dataset.lower() == "mot20":
        VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/mot20_valid_V1"
        QUERY_JSON = "data/queries_mot20.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/mot20"
        OUTPUT_JSON = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/mot20_outputs.json"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Dataset: {args.dataset}")
    print(f"VIDEO_DIR: {VIDEO_DIR}")
    print(f"QUERY_JSON: {QUERY_JSON}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"OUTPUT_JSON: {OUTPUT_JSON}")

    # Load Model Default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
    )
    # Default Processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    with open(QUERY_JSON, "r") as f:
        query_dict = json.load(f)

    for video_name, query_list in query_dict.items():
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"[WARN] Video not found: {video_path}")
            continue

        output_json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_name)[0]}.json")
        if os.path.exists(output_json_path):
            with open(output_json_path, "r") as f:
                results = json.load(f)
            print(f"[INFO] Loaded existing results for {video_name}")
        else:
            results = {}

        print(f"\n=== Processing video: {video_name} ===")
        video_info = get_scaled_video_info(video_path, max_long_side=420, min_short_side=240)
        print(f"\n=== Video info: {video_info} ===")

        for query in query_list:
            if query in results:
                print(f"--> Query '{query}' already processed, skipping.")
                continue

            print(f"--> Query: {query}")

            # Construct messages for Qwen
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video",
                         "video": f"file://{video_path}",
                         "max_pixels": video_info["scaled_size"][0] * video_info["scaled_size"][1],
                         "fps": 1.0},
                        {"text": f"Given the query \"{query}\", for each frame, detect and localize the visual content described by the given textual query in JSON format. If the visual content does not exist in a frame, skip that frame. Output Format: [{{\"time\": 1.0, \"bbox_2d\": [x_min, y_min, x_max, y_max], \"label\": \"\"}}, {{\"time\": 2.0, \"bbox_2d\": [x_min, y_min, x_max, y_max], \"label\": \"\"}}, ...]."}
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
                # max_new_tokens=512
            )

            out_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
            output_text = processor.decode(out_tokens, skip_special_tokens=True)

            try:
                parsed = json.loads(output_text)
            except:
                print("[ERROR] Model did not output valid JSON. Saving raw text.")
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
        print(f"[INFO] Saved intermediate results for query '{query}'")

    print(f"[INFO] Finished video {video_name}, results saved to {output_json_path}")

if __name__ == "__main__":
    main()
# -----------------------------
# 配置区
# -----------------------------
# VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/mot20_videos_V1/"  # 本地视频目录
# QUERY = "Describe this video."  # 查询内容
# OUTPUT_JSON = "results.json"  # 输出文件
# FPS = 1.0  # 视频抽帧速率
# MAX_PIXELS = 360 * 420  # 输入图片大小限制
# DEVICE = "cuda"  # 使用 GPU

# -----------------------------
# 初始化模型与 processor
# -----------------------------

# model.to(DEVICE)
# model.eval()

# -----------------------------
# 遍历视频列表
# -----------------------------
# video_paths = [str(p.absolute()) for p in Path(VIDEO_DIR).glob("*.mp4")]
# results = {}
#
# for video_path in video_paths:
#     print(f"Processing video: {video_path}")
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "video",
#                     "video": f"file://{video_path}",
#                     "max_pixels": MAX_PIXELS,
#                     "fps": FPS,
#                 },
#                 {"type": "text", "text": QUERY},
#             ],
#         }
#     ]
#
#     # Preparation for inference
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#         **video_kwargs,
#     )
#     inputs = inputs.to("cuda")
#
#     # Inference: Generation of the output
#     # with torch.no_grad():
#     generated_ids = model.generate(**inputs, max_new_tokens=128)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     print(f"Output for {video_path}:\n{output_text[0]}\n")
#
#     # 清理显存
#     del inputs, image_inputs, video_inputs
#     torch.cuda.empty_cache()
#
