import os
import json
from collections import defaultdict

# step 1: parse results to generate json file.
def parse_prediction(value):
    """
    value: either "No, it does not happen." or "Yes, (s, e, score)"
    """
    if value.startswith("No"):
        return [0, 0, 1]

    # value like: "Yes, (15, 20, 0.9)"
    try:
        inside = value.split("Yes, (")[1].rstrip(")")
        nums = inside.split(",")
        return [float(nums[0]), float(nums[1]), float(nums[2])]
    except:
        print("Parse error:", value)
        return [0, 0, 1]

def collect_results(input_dir, output_path):

    results = defaultdict(lambda: defaultdict(list))  # {vid: {query: [predictions,...]}}

    # ---------------------------------------------------------
    # Traverse all json files in input_dir
    # ---------------------------------------------------------
    for file in os.listdir(input_dir):
        if not file.endswith(".json"):
            continue

        filepath = os.path.join(input_dir, file)

        # "xxx_0.0_70.0_obj1.json" → "xxx_0.0_70.0"
        vid = "_".join(file.split("_")[:-1])

        with open(filepath, "r") as f:
            data = json.load(f)

        for query, value in data.items():
            pred = parse_prediction(value)
            results[vid][query].append(pred)

    # ---------------------------------------------------------
    # Sort predictions + move [0,0,1] to the end if needed
    # ---------------------------------------------------------
    final_output = []

    for vid, query_dict in results.items():
        queries_list = []
        for query, preds in query_dict.items():
            # split into positive preds (not 001) and zero preds
            pos_preds = [p for p in preds if not (p[0] == 0 and p[1] == 0 and p[2] == 1)]
            zero_preds = [p for p in preds if (p[0] == 0 and p[1] == 0 and p[2] == 1)]

            # sort positives by score desc
            pos_preds_sorted = sorted(pos_preds, key=lambda x: x[2], reverse=True)

            # if there is at least one positive prediction → put 001s at end
            if pos_preds_sorted:
                final_preds = pos_preds_sorted + zero_preds
            else:
                # no valid preds → keep as original sorted (all 001)
                final_preds = zero_preds  # equivalent to preds

            queries_list.append({
                "query": query,
                "prediction": final_preds
            })

        final_output.append({
            "vid": vid,
            "queries": queries_list
        })

    # ---------------------------------------------------------
    # Write results.json
    # ---------------------------------------------------------
    with open(output_path, "w") as f:
        json.dump(final_output, f)

    print(f"Saved aggregated results to {output_path}")


# collect_results(
#     input_dir="/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/ovis_spatial_exp1",
#     output_path="ovis_exp1.json"
# )


# step 2: generate x_val_preds.jsonl
def fill_preds_from_results(template_path, results_path, output_path):
    """
    根据 results.json 填充 template_ovis_val_preds.jsonl 的 pred_relevant_windows
    并保存到 output_path
    """
    # ----------------------------
    # 1. 读取 results.json，组织为字典便于查找
    # structure: results_dict[vid][query] = prediction
    # ----------------------------
    with open(results_path, "r") as f:
        results_data = json.load(f)

    results_dict = {}
    for item in results_data:
        vid = item["vid"]
        results_dict[vid] = {}
        for q in item["queries"]:
            query_text = q["query"]
            results_dict[vid][query_text] = q["prediction"]

    # ----------------------------
    # 2. 读取 template_ovis_val_preds.jsonl，填充 pred_relevant_windows
    # ----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(template_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            vid = entry["vid"]
            query = entry["query"]

            # 默认空列表
            pred_windows = []

            if vid in results_dict and query in results_dict[vid]:
                pred_windows = results_dict[vid][query]

            entry["pred_relevant_windows"] = pred_windows
            fout.write(json.dumps(entry) + "\n")

    print(f"Filled predictions saved to {output_path}")

fill_preds_from_results(
    template_path="template_ovis_val_preds.jsonl",
    results_path="ovis_exp1.json",
    output_path="exp1/ovis_val_preds.jsonl"
)
