import argparse
import json
import os
import glob
import re

def query_to_dirname(query):
    return query.lower().replace(' ', '-')

def clean_objects_str(obj_str):
    """
    清理 'objects' 字符串：去掉 ```json 前缀和尾部 ``` ，然后解析为 dict
    """
    # 去掉 ```json 或 ``` 前缀
    obj_str = re.sub(r"^```json\s*", "", obj_str)
    obj_str = re.sub(r"```$", "", obj_str)
    obj_str = obj_str.strip()
    try:
        return json.loads(obj_str)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        # 尝试处理一些常见截断问题
        # 例如最后可能有不完整的尾巴，可以截断最后一个不完整的对象
        last_brace = obj_str.rfind("}")
        if last_brace != -1:
            obj_str = obj_str[:last_brace+1]
            try:
                return json.loads(obj_str)
            except:
                return None
        return None

def process_objects_field(objects_field):
    """安全解析 objects 字段，返回 dict 或 None"""
    obj_str = clean_objects_str(objects_field)
    if not obj_str:
        return None
    try:
        return json.loads(obj_str)
    except json.JSONDecodeError as e:
        print("Warning: Failed to parse objects field:", e)
        return None

def process_file(json_path, save_base_dir):
    video_name = os.path.basename(json_path).split('_')[0]
    with open(json_path, 'r') as f:
        data = json.load(f)

    for query, content in data.items():
        orig_size = content['orig_size']  # [W, H]
        objects_str = content['objects']

        # 清理 ```json 包裹
        prefix = '```json\n'
        suffix = '```'
        if objects_str.startswith(prefix):
            objects_str = objects_str[len(prefix):]
        if objects_str.endswith(suffix):
            objects_str = objects_str[:-len(suffix)]

        out_dir = os.path.join(save_base_dir, video_name, query_to_dirname(query))
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, 'predict.txt')

        lines = []
        # 提取 scaled_size
        scaled_size_match = re.search(r'"scaled_size"\s*:\s*\[(\d+),\s*(\d+)\]', objects_str)
        if scaled_size_match:
            scaled_size = [int(scaled_size_match.group(1)), int(scaled_size_match.group(2))]
            scale_w = orig_size[0] / scaled_size[0]
            scale_h = orig_size[1] / scaled_size[1]
            print(f"{json_path} query: {query}, scaled_size: {scale_w} x {scale_h}")

            object_id_matches = re.findall(r'"object_id"\s*:\s*(\d+)', objects_str)
            frames_blocks = re.findall(r'"frames"\s*:\s*\[(.*?)\](?=,?\s*"object_id"|$)', objects_str, re.DOTALL)

            if object_id_matches and frames_blocks and len(object_id_matches) == len(frames_blocks):
                for obj_id, frames_block in zip(object_id_matches, frames_blocks):
                    frame_matches = re.findall(
                        r'\{"frame_id"\s*:\s*(\d+),\s*"bbox"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\}',
                        frames_block
                    )
                    for frame in frame_matches:
                        frame_id, x1, y1, x2, y2 = map(int, frame)
                        x1_orig = x1 * scale_w
                        y1_orig = y1 * scale_h
                        x2_orig = x2 * scale_w
                        y2_orig = y2 * scale_h
                        w = x2_orig - x1_orig
                        h = y2_orig - y1_orig
                        line = f'{frame_id + 1},{obj_id},{x1_orig},{y1_orig},{w},{h},1,1,1\n'
                        lines.append(line)

            with open(out_file, 'w') as f:
                f.writelines(lines)

            if lines:
                print(f'Saved predict.txt with {len(lines)} frames for {video_name}/{query}')
            else:
                print(f'Saved empty predict.txt for {video_name}/{query} (no predictions)')
        else:
            print(f"Warning: No scaled_size for {json_path} query: {query}, skipping.")
            continue


def process(json_path, save_base_dir):
    video_name = os.path.basename(json_path).split('_')[0]
    with open(json_path, 'r') as f:
        data = json.load(f)

    for query, content in data.items():
        orig_size = content['orig_size']  # [W, H]
        objects_str = content.get('objects', '')

        print(f'orig_size: {orig_size}')
        # print(f'objects_str: {objects_str}, type: {type(objects_str)}')
        # clean ```json ... ```
        if objects_str.startswith('```json'):
            objects_str = objects_str[7:]
        if objects_str.endswith('```'):
            objects_str = objects_str[:-3]

        print(f'objects_str: {objects_str}, type: {type(objects_str)}')


        # 提取 objects 内的 scaled_size
        scaled_size_match = re.search(r'"scaled_size"\s*:\s*\[(\d+),\s*(\d+)\]', objects_str)
        if not scaled_size_match:
            print(f"Warning: No scaled_size in {json_path} query: {query}, creating empty predict.txt")
            scaled_size = orig_size
        else:
            scaled_size = [int(scaled_size_match.group(1)), int(scaled_size_match.group(2))]

        scale_w = orig_size[0] / scaled_size[0]
        scale_h = orig_size[1] / scaled_size[1]

        print(f'scaled_size: {scaled_size}, scale: {scale_w} x {scale_h}') # 以上是正确的

        out_dir = os.path.join(save_base_dir, video_name, query_to_dirname(query))
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, 'predict.txt')

        # 正则提取每个 object_id 和对应 frames
        object_blocks = re.findall(
            r'\{\s*"object_id"\s*:\s*\d+.*?(?=\n\s*\{\s*"object_id"|\n\s*\]|\Z)',
            objects_str,
            re.DOTALL
        )

        print("Found objects:", len(object_blocks), object_blocks)


        lines = []

        for block in object_blocks:
            m = re.search(r'"object_id"\s*:\s*(\d+)', block)
            if not m:
                continue
            object_id = int(m.group(1))

            print("block:", block)

            m2 = re.search(r'"frames"\s*:\s*\[(.*)\]', block, re.DOTALL)
            if not m2:
                print(f"Warning: No frames found for object_id {object_id}")
                continue
            frames_str = m2.group(1)

            if not frames_str.endswith(']}'):
                frames_str = frames_str.rstrip(',')  # 去掉末尾可能的逗号
                frames_str += ']}'

            print("Found frame_items:", len(frames_str), frames_str)

            # 4. 抓所有 frame 行
            frame_items = re.findall(
                # r'\{\s*"frame_id"\s*:\s*(\d+)\s*,\s*"bbox"\s*:\s*\[\s*([^\]]+?)\s*\]\s*\}',
                r'"frame_id"\s*:\s*(\d+)\s*,\s*"bbox"\s*:\s*\[\s*([\d\s,]+)\s*\]',
                frames_str,
                re.DOTALL
            )

            print("Found frame_items:", len(frame_items), frame_items)

            frames_json = []
            for frame_id_str, bbox_str in frame_items:
                bbox = list(map(float, bbox_str.split(",")))
                frames_json.append({
                    "frame_id": int(frame_id_str),
                    "bbox": bbox
                })

            # 5. 写入 predict.txt 格式内容（你要求的版本）
            for frame in frames_json:
                frame_id = frame['frame_id']
                bbox = frame.get('bbox')
                if not bbox or len(bbox) != 4:
                    # 如果 bbox 不存在或者长度不是 4，则跳过
                    continue

                # bbox unpack
                x1, y1, x2, y2 = bbox

                # scale back using scaled_size (你的要求)
                x1_orig = x1 * scale_w
                y1_orig = y1 * scale_h
                x2_orig = x2 * scale_w
                y2_orig = y2 * scale_h

                # 转换为 w h
                w = x2_orig - x1_orig
                h = y2_orig - y1_orig

                save_format = '{frame},{id},{x1},{y1},{w},{h},1,1,1\n'
                line = save_format.format(frame=int(frame_id+1), id=int(object_id), x1=x1_orig, y1=y1_orig, w=w, h=h)
                lines.append(line)

        # 即使没有 lines，也生成空 predict.txt
        with open(out_file, 'w') as f:
            f.writelines(lines)

        if lines:
            print(f"Saved predict.txt for {video_name}/{query}")
        else:
            print(f"Saved empty predict.txt for {video_name}/{query}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate template and fill predictions for Spatial Video Grounding.")
    parser.add_argument('--base_dir', type=str, required=True, help='Input JSON folder')
    parser.add_argument('--save_base_dir', type=str, required=True, help='Output folder for predict.txt')
    args = parser.parse_args()

    json_files = glob.glob(os.path.join(args.base_dir, '*.json'))
    for json_path in json_files:
        process(json_path, args.save_base_dir)

    base_dir = '/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/outputs/ovis_spatial_1'
    save_base_dir = '/nfs/data3/shuaicong/SVAG-Bench/Qwen-VL/spatial_preds/ovis'  # 输出根目录

