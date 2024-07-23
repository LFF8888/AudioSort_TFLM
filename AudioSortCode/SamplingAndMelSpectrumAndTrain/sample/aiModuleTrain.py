import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Softmax

from matplotlib import font_manager as fm

# 配置参数
NUM_MEL_FILTERS = 16  # 梅尔滤波器数量
NUM_MEL_FRAMES = 20  # 梅尔帧数量
EPOCHS = 100  # 训练轮数
BATCH_SIZE = 32  # 批量大小
DATA_PATH = './'  # 数据路径

# 设置中文字体
def set_chinese_font():
    font_path = '/System/Library/Fonts/STHeiti Medium.ttc'  # macOS系统字体路径
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    else:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 使用常见的中文支持字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 调用设置中文字体函数
set_chinese_font()

# 自动识别分类数
def get_num_classes(data_path):
    # 获取所有子目录作为分类名称
    class_names = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    return len(class_names), class_names

# 调用函数获取分类数和分类名称
NUM_CLASSES, class_names = get_num_classes(DATA_PATH)

# 输出0123和标签的对应关系
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

# 加载数据
def load_data(data_path, class_names):
    X = []
    y = []
    class_counts = {class_name: 0 for class_name in class_names}  # 统计各类样本数量
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_path, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(class_dir, file_name)
                data = np.load(file_path)
                if data.shape == (NUM_MEL_FILTERS, NUM_MEL_FRAMES):  # 确保数据维度一致
                    X.append(data)
                    y.append(i)  # 标签从0开始编号
                    class_counts[class_name] += 1  # 增加样本数量
    X = np.array(X)
    y = np.array(y)
    return X, y, class_counts

# 调用函数加载数据
X, y, class_counts = load_data(DATA_PATH, class_names)

# 输出各类数据集样本数量
for class_name, count in class_counts.items():
    print(f"类别 {class_name} 的样本数量: {count}")

# # 显示几张频谱图
# def show_sample_spectrograms(X, y, class_names, num_samples=5):
#     fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
#     for i in range(num_samples):
#         ax = axes[i]
#         ax.imshow(X[i], aspect='auto', origin='lower')
#         ax.set_title(f"类别: {class_names[int(y[i])]}")
#         ax.axis('off')
#     plt.show()

# # 调用函数显示样本频谱图
# show_sample_spectrograms(X, y, class_names)

# 数据预处理
y = np.clip(y, 0, NUM_CLASSES-1)  # 确保标签在0到NUM_CLASSES-1范围内
y = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)  # 将标签转换为one-hot编码

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建模型
model = Sequential([
    # 第一层卷积，考虑使用较小的卷积核和较多的过滤器来捕捉细节
    Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=(NUM_MEL_FILTERS, NUM_MEL_FRAMES, 1)),
    # 最大池化层，减少数据的空间大小
    MaxPooling2D(pool_size=(2, 2)),
    # 可以考虑添加更多卷积层和池化层来提取更复杂的特征
    Conv2D(4, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    # 展平层，将二维输出转化为一维，以便传入全连接层
    Flatten(),
    # 全连接层，数量可以根据问题的复杂性来调整
    Dense(4, activation='relu'),
    # 输出层，使用Softmax激活函数进行分类
    Dense(NUM_CLASSES, activation='softmax')
])

# 打印模型结构
model.summary()

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

# 绘制训练结果
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

plt.show()

# 将Keras模型转换为TFLite模型
tflite_model_name = 'audio_classification_model'
c_model_name = 'audio_classification_model'

converter = tf.lite.TFLiteConverter.from_keras_model(model)  # 创建转换器
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 优化模型大小
tflite_model = converter.convert()  # 转换模型

# 保存TFLite模型
with open(tflite_model_name + '.tflite', 'wb') as f:
    f.write(tflite_model)

def hex_to_c_array(hex_data, var_name):
    c_str = ''
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'
    c_str += 'const unsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'
    c_str += 'alignas(8) const unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data):
        hex_str = format(val, '#04x')  # 格式化十六进制字符串
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n  '
        hex_array.append(hex_str)
    c_str += '\n  ' + ' '.join(hex_array) + '\n};\n\n'
    c_str += '#endif //' + var_name.upper() + '_H\n'
    return c_str

# 将TFLite模型写入C头文件
with open(c_model_name + '.h', 'w') as file:
    file.write(hex_to_c_array(tflite_model, c_model_name))
