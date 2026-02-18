import os
import json
import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# -----------------------------
# 配置区
# -----------------------------
VIDEO_DIR = "/nfs/data3/shuaicong/videos_by_images/mot20_videos_V1/"  # 本地视频目录
QUERY = "Describe this video."  # 查询内容
OUTPUT_JSON = "results.json"  # 输出文件
FPS = 1.0  # 视频抽帧速率
MAX_PIXELS = 360 * 420  # 输入图片大小限制
# DEVICE = "cuda"  # 使用 GPU

# -----------------------------
# 初始化模型与 processor
# -----------------------------
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
)
# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# model.to(DEVICE)
# model.eval()

# -----------------------------
# 遍历视频列表
# -----------------------------
video_paths = [str(p.absolute()) for p in Path(VIDEO_DIR).glob("*.mp4")]
results = {}

for video_path in video_paths:
    print(f"Processing video: {video_path}")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "max_pixels": MAX_PIXELS,
                    "fps": FPS,
                },
                {"type": "text", "text": QUERY},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    # with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"Output for {video_path}:\n{output_text[0]}\n")

    # 清理显存
    del inputs, image_inputs, video_inputs
    torch.cuda.empty_cache()

