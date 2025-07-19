import os
import cv2
import re
from tqdm import tqdm

# 路径设置
base_path = r"M:\EEG_DATA\EEG_data_0410\heatmap_ready_final\Heatmaps_allgaze_nomotion"
input_path = os.path.join(base_path, "merged")
output_path = os.path.join(base_path, "stacked_precise")
os.makedirs(output_path, exist_ok=True)

# 文件分类
all_files = [f for f in os.listdir(input_path) if "bd" in f.lower()]
fixed_null_files = [f for f in all_files if "fixednull" in f.lower()]
other_files = [f for f in all_files if "fixednull" not in f.lower()]

print(f"找到 {len(fixed_null_files)} 个 FixedNull 文件")
print(f"找到 {len(other_files)} 个其他文件")


def extract_key(filename):
    """提取标准匹配键（第3个'_'到倒数第二个'_'）"""
    parts = filename.split('_')
    return '_'.join(parts[3:-2]) if len(parts) >= 5 else None


def extract_noveh_key(filename):
    """特殊提取：包含noveh的匹配键（保留完整结构）"""
    parts = filename.split('_')
    if "noveh" not in filename.lower():
        return None
    return '_'.join(parts[3:-2])  # 保持原始匹配键格式


def extract_number(filename, is_fixednull):
    """提取数字编号"""
    if is_fixednull:
        match = re.search(r'^[^_]+_(\d+)_', filename)  # FixedNull文件取第一个数字
    else:
        match = re.search(r'bd(\d+)', filename.lower())  # 其他文件取bd后数字
    return match.group(1) if match else None


# 预处理其他文件
other_files_dict = {}
noveh_files_dict = {}

for f in other_files:
    # 标准匹配键处理
    std_key = extract_key(f)
    num = extract_number(f, is_fixednull=False)

    if std_key and num:
        if std_key not in other_files_dict:
            other_files_dict[std_key] = {}
        if num not in other_files_dict[std_key]:
            other_files_dict[std_key][num] = []
        other_files_dict[std_key][num].append(f)

    # 特殊noveh处理
    noveh_key = extract_noveh_key(f)
    if noveh_key and num:
        if noveh_key not in noveh_files_dict:
            noveh_files_dict[noveh_key] = {}
        if num not in noveh_files_dict[noveh_key]:
            noveh_files_dict[noveh_key][num] = []
        noveh_files_dict[noveh_key][num].append(f)

for fixed_null_file in tqdm(fixed_null_files, desc="精确匹配处理"):
    try:
        # 基础信息提取
        std_key = extract_key(fixed_null_file)
        noveh_key = extract_noveh_key(fixed_null_file)
        fixed_num = extract_number(fixed_null_file, is_fixednull=True)

        if not fixed_num:
            print(f"跳过文件（无有效数字）: {fixed_null_file}")
            continue

        # 读取FixedNull图片
        img_top = cv2.imread(os.path.join(input_path, fixed_null_file))
        if img_top is None:
            continue

        # 增强标注（字号1.0，红色加粗）
        cv2.putText(img_top, fixed_null_file, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # 匹配逻辑选择
        matched_files = []
        if noveh_key:  # 优先使用noveh特殊匹配
            if noveh_key in noveh_files_dict and fixed_num in noveh_files_dict[noveh_key]:
                matched_files = noveh_files_dict[noveh_key][fixed_num]
            # 放宽条件：匹配去掉noveh部分的键（如wal_rev_3.6）
            short_key = noveh_key.replace("_noveh", "")
            if short_key in other_files_dict and fixed_num in other_files_dict[short_key]:
                matched_files += other_files_dict[short_key][fixed_num]
        elif std_key:  # 标准匹配
            if std_key in other_files_dict and fixed_num in other_files_dict[std_key]:
                matched_files = other_files_dict[std_key][fixed_num]

        if not matched_files:
            print(f"未匹配: {fixed_null_file} | 键: {noveh_key or std_key} | 需数字: {fixed_num}")
            cv2.imwrite(os.path.join(output_path, f"unmatched_{fixed_null_file}"), img_top)
            continue

        # 处理匹配文件
        imgs_bottom = []
        for f in matched_files:
            img = cv2.imread(os.path.join(input_path, f))
            if img is not None:
                cv2.putText(img, f, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                imgs_bottom.append(img)

        # 垂直堆叠（自动对齐宽度）
        max_width = max(img_top.shape[1], max(img.shape[1] for img in imgs_bottom)) if imgs_bottom else img_top.shape[1]


        def resize_img(img, width):
            if img.shape[1] != width:
                return cv2.resize(img, (width, img.shape[0]))
            return img


        img_top = resize_img(img_top, max_width)
        imgs_bottom = [resize_img(img, max_width) for img in imgs_bottom]

        result = cv2.vconcat([img_top, *imgs_bottom]) if imgs_bottom else img_top

        # 保存结果（质量100%）
        output_name = f"matched_{noveh_key or std_key}_num{fixed_num}.jpg"
        cv2.imwrite(os.path.join(output_path, output_name), result, [cv2.IMWRITE_JPEG_QUALITY, 100])

    except Exception as e:
        print(f"处理错误 {fixed_null_file}: {str(e)}")

print(f"处理完成！结果保存在: {output_path}")