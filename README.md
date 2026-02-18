# Qwen-VL

## Temporal Inference
📄 [inference_qwen_temporal.py](inference_qwen_temporal.py)
```
def build_grounding_prompt(query):
        return f"""
You are performing temporal grounding for a video.

Event: "{query}"

Please answer using EXACTLY the following format:
{query} from frame <start_frame> to <end_frame>

Only output the final answer in ONE line, no explanation.
"""
```
## Spatial Inference
📄 [inference_qwen_spatial.py](inference_qwen_spatial.py)
```
def build_grounding_prompt(query):
        return f"""
You are an expert in spatial-temporal grounding for video.

Your task:
Given a query and a video, locate all visual objects that satisfy the description and track each object across all frames where it appears.

The video has been scaled before being fed to you.  
All bounding boxes you output must correspond to the scaled video resolution (scaled_size).  

For each object:
- Assign a unique and consistent object_id
- Track the object across frames
- For every frame in which the object appears, output its bbox

Bounding box format:
[x1, y1, x2, y2]  
Coordinates must be in pixel values of the scaled video (NOT normalized).

When no object appears in a specific frame, do not include that frame.

Required output JSON format (strict):
{{
  "scaled_size": [scaled_W, scaled_H],
  "scale": scale,
  "objects": [
    {{
      "object_id": 1,
      "frames": [
        {{"frame_id": 5, "bbox": [x1, y1, x2, y2]}},
        {{"frame_id": 6, "bbox": [x1, y1, x2, y2]}},
        ...
      ]
    }},
    {{
      "object_id": 2,
      "frames": [
       {{"frame_id": 8, "bbox": [x1, y1, x2, y2]}},
        ...
      ]
    }}
  ]
}}

Notes:
- Do NOT include explanations.
- Only output the JSON object described above.

User description:
{query}
"""
```