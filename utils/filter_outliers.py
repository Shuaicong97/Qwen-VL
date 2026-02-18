import os
import shutil
import csv

# 目录路径
dir_a = "/nfs/data3/shuaicong/TempRMOT/refer-ovis/OVIS/valid"
dir_b = "/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/spatial_preds/ovis"

# 遍历目录a
for folder_name in os.listdir(dir_a):
    folder_a_path = os.path.join(dir_a, folder_name)
    if not os.path.isdir(folder_a_path):
        continue

    # 统计文件数量（假设只统计jpg）
    length = len([f for f in os.listdir(folder_a_path) if f.endswith('.jpg')])
    print(f'folder_a_path: {folder_a_path}, length: {length}')

    # 对应目录b的文件夹
    folder_b_path = os.path.join(dir_b, folder_name)
    if not os.path.isdir(folder_b_path):
        print(f"Warning: {folder_b_path} does not exist.")
        continue

    # 遍历b目录下的子文件夹
    for subfolder_name in os.listdir(folder_b_path):
        subfolder_path = os.path.join(folder_b_path, subfolder_name)
        if not os.path.isdir(subfolder_path):
            continue

        predict_file = os.path.join(subfolder_path, "predict.txt")
        if not os.path.isfile(predict_file):
            print(f"Warning: {predict_file} does not exist.")
            continue

        # 备份原始文件
        predict_ori_file = os.path.join(subfolder_path, "predict_ori.txt")
        shutil.copyfile(predict_file, predict_ori_file)

        # 读取predict.txt并筛选
        filtered_lines = []
        with open(predict_file, "r") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                filtered_lines.append(header)
            except StopIteration:
                print(f"Warning: {predict_file} is empty.")
                continue  # 跳过空文件

            for row in reader:
                try:
                    frame_id = int(row[0])
                    if frame_id <= length:
                        filtered_lines.append(row)
                except ValueError:
                    print(f"Skipping line with invalid frame_id: {row}")

        # 保存筛选后的内容
        with open(predict_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(filtered_lines)

        print(f"Processed {predict_file}, kept {len(filtered_lines)-1} lines.")

print("All done!")
