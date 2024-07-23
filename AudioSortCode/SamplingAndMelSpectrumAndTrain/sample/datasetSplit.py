import os
import numpy as np

# 配置参数
NUM_MEL_FILTERS = 16
NUM_MEL_FRAMES = 20

# 文件夹名称
file_names = ["hunger.txt", "sleepy.txt", "burp.txt", "gassy.txt", "discomfort.txt", "xiaoxin.txt", "nothing.txt"]

def clean_data(row):
    """清理数据行，移除非数字字符"""
    cleaned_row = []
    for x in row.split(','):
        try:
            cleaned_row.append(float(x.strip()))
        except ValueError:
            continue
    return cleaned_row

def split_data_and_save(file_path, class_name, num_filters, num_frames):
    with open(file_path, 'r') as file:
        data = file.read()
    
    # 分割数据块
    data_blocks = data.split('------------------------------------\n')
    data_blocks = [block.strip() for block in data_blocks if block.strip()]
    
    # 确保输出目录存在
    if not os.path.exists(class_name):
        os.makedirs(class_name)
    
    # 遍历每个数据块，拆分成指定大小的矩阵并保存
    for idx, block in enumerate(data_blocks):
        rows = block.split('\n')
        matrix = []
        for row in rows:
            if row.strip():
                cleaned_row = clean_data(row)
                if len(cleaned_row) == num_frames:
                    matrix.append(cleaned_row)
        
        # 确认矩阵大小是否正确
        if len(matrix) == num_filters:
            # 保存文件
            output_file = os.path.join(class_name, f"{class_name}{idx+1:04d}.npy")
            np.save(output_file, np.array(matrix))
        else:
            print(f"错误：在文件 {file_path} 中找到的数据块 {idx+1} 格式不正确。")

# 对每个文件进行处理
for file_name in file_names:
    if os.path.exists(file_name):
        class_name = os.path.splitext(file_name)[0]
        split_data_and_save(file_name, class_name, NUM_MEL_FILTERS, NUM_MEL_FRAMES)
    else:
        print(f"警告：文件 {file_name} 不存在。")
