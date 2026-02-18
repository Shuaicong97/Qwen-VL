import json
import re
from pathlib import Path
import argparse

frame_patterns = [
    re.compile(r'from\s+frame\s*(\d+)\s*to\s*frame\s*(\d+)', re.I),
    re.compile(r'frame\s*(\d+)\s*to\s*(\d+)', re.I),
    re.compile(r'frames?\s*(\d+)\s*-\s*(\d+)', re.I),
    re.compile(r'(\d+)\s*to\s*(\d+)\s*frames?', re.I),
]

def parse_frames(text):
    for p in frame_patterns:
        m = p.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None

def generate_pred_template(gt_file, template_file):
    with open(gt_file, "r", encoding="utf-8") as fin, \
         open(template_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            gt_entry = json.loads(line)
            template_entry = {
                "qid": gt_entry["qid"],
                "query": gt_entry["query"],
                "vid": gt_entry["vid"],
                "pred_relevant_windows": []
            }
            fout.write(json.dumps(template_entry, ensure_ascii=False) + "\n")
    print(f"[INFO] Template generated: {template_file}")

def fill_predictions(template_file, output_dir, output_file):
    output_dir = Path(output_dir)
    preds = []
    with open(template_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            preds.append(json.loads(line))

    for item in preds:
        vid = item.get("vid")
        if not vid:
            continue
        pred_file = output_dir / f"{vid}.json"
        if not pred_file.exists():
            print(f"[WARN] prediction file not found for vid: {vid}")
            continue
        data = json.load(pred_file.open("r", encoding="utf-8"))
        query = item.get("query")
        if query in data:
            val = data[query]
            parsed = parse_frames(val)
            if parsed:
                start, end = parsed
                item["pred_relevant_windows"] = [[float(start), float(end), 1.0]]
            else:
                print(f"[WARN] could not parse frames from text: {val}")
        else:
            print(f"[WARN] query not found in prediction file for vid {vid}: query='{query}'")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in preds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[INFO] Final predictions written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate template and fill predictions for Temporal Video Grounding.")
    parser.add_argument("--gt_file", type=str, required=True, help="Path to ground truth JSONL file")
    parser.add_argument("--template_file", type=str, required=True, help="Path to output template JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="Path to final predictions JSONL")
    parser.add_argument("--qwen_output_dir", type=str, required=True, help="Path to directory containing Qwen JSON outputs")
    args = parser.parse_args()

    generate_pred_template(args.gt_file, args.template_file)
    fill_predictions(args.template_file, args.qwen_output_dir, args.output_file)