import os
import shutil

# 源目录（包含 gt.txt）
src_root = "/nfs/data3/shuaicong/TransRMOT/outputs/mot20/results_epoch99"
# 目标目录（需要复制 gt.txt）
dst_root = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/spatial_preds/mot20"

# 遍历源目录
for seq_folder in os.listdir(src_root):
    seq_src_path = os.path.join(src_root, seq_folder)
    seq_dst_path = os.path.join(dst_root, seq_folder)

    if not os.path.isdir(seq_src_path) or not os.path.isdir(seq_dst_path):
        continue  # 确保都是文件夹

    # 遍历每个子任务文件夹
    for sub_folder in os.listdir(seq_src_path):
        sub_src_path = os.path.join(seq_src_path, sub_folder)
        sub_dst_path = os.path.join(seq_dst_path, sub_folder)

        if not os.path.isdir(sub_src_path) or not os.path.isdir(sub_dst_path):
            continue

        # gt.txt 文件路径
        gt_src_file = os.path.join(sub_src_path, "gt.txt")
        gt_dst_file = os.path.join(sub_dst_path, "gt.txt")

        # 复制文件
        if os.path.exists(gt_src_file):
            shutil.copy(gt_src_file, gt_dst_file)
            print(f"Copied {gt_src_file} -> {gt_dst_file}")
        else:
            print(f"Warning: {gt_src_file} not found")
