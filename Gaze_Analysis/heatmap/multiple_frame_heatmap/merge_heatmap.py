import os
import cv2
from tqdm import tqdm

# 设置路径
base_path = r"M:\EEG_DATA\EEG_data_0410\heatmap_ready_final\Heatmaps_allgaze_nomotion"
exp_path = os.path.join(base_path, "Exp")
nov_path = os.path.join(base_path, "Nov")
output_path = base_path

# 确保输出目录存在
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 获取两个文件夹中的文件列表
exp_files = set(os.listdir(exp_path))
nov_files = set(os.listdir(nov_path))

# 找出两个文件夹中都存在的文件（交集）
common_files = exp_files & nov_files

print(f"找到 {len(common_files)} 对匹配的文件")

# 处理每对匹配的文件
for filename in tqdm(common_files, desc="处理文件中"):
    try:
        # 读取图片
        img_exp = cv2.imread(os.path.join(exp_path, filename))
        img_nov = cv2.imread(os.path.join(nov_path, filename))

        if img_exp is None or img_nov is None:
            print(f"无法读取 {filename}，跳过")
            continue

        # 确保两张图片高度相同，如果不相同则调整
        if img_exp.shape[0] != img_nov.shape[0]:
            min_height = min(img_exp.shape[0], img_nov.shape[0])
            img_exp = img_exp[:min_height, :]
            img_nov = img_nov[:min_height, :]

        # 水平拼接图片（Exp在左，Nov在右）
        combined = cv2.hconcat([img_exp, img_nov])

        # 保存结果
        output_filename = f"combined_{filename}"
        cv2.imwrite(os.path.join(output_path, output_filename), combined)

    except Exception as e:
        print(f"处理 {filename} 时出错: {str(e)}")

print("处理完成！")