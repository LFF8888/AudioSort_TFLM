<div id="chinese-content">

<div lang="zh-CN">
<p align="center">
  <img src="https://storage.googleapis.com/gweb-developer-goog-blog-assets/images_archive/original_images/image1_v7xhr8h.png" alt="TensorFlow Lite Micro Logo" width="200"/>
</p>

<h1 align="center">ESP32-S3 婴儿哭声识别与分类 (TensorFlow Lite Micro)</h1>

<p align="center">
  <strong>一个在 ESP32-S3 微控制器上使用 TensorFlow Lite Micro 实现婴儿哭声实时识别与分类的项目。</strong>
</p>

<p align="center">
  <a href="https://www.bilibili.com/video/BV1uX8veJEGi" target="_blank">
    <img src="https://img.shields.io/badge/观看B站视频教程-► 点击这里-brightgreen?style=for-the-badge&logo=bilibili" alt="B站视频教程">
  </a>
</p>

<p align="center">
  <a href="#chinese-content"><img src="https://img.shields.io/badge/语言-中文-blue.svg?style=flat-square" alt="中文"></a>
  <a href="#english-content"><img src="https://img.shields.io/badge/Language-English-green.svg?style=flat-square" alt="English"></a>
</p>

---

### 📖 项目概述

本项目旨在演示如何在资源受限的 ESP32-S3 微控制器上部署一个 TensorFlow Lite Micro (TFLM) 模型，用于实时识别和分类婴儿的哭声。通过分析哭声，系统可以帮助父母或看护人初步判断婴儿可能的需求，例如：

*   `discomfort` (不舒服)
*   `burp` (胀气/打嗝)
*   `sleepy` (困了)
*   `hunger` (饿了)
*   以及一个 `nothing` (无特定声音/背景噪音) 类别和一个自定义唤醒词 `xiaoxin` (小鑫)。

整个流程涵盖了从音频数据采集、特征提取（梅尔频谱图）、模型训练、模型转换到最终在 ESP32-S3 上部署运行的全过程。

### ✨ 主要特性

*   **端侧智能**：直接在 ESP32-S3 上进行音频处理和机器学习推理，无需云端依赖。
*   **低成本方案**：采用常见的 ESP32-S3 开发板和 I2S 麦克风模块。
*   **实时响应**：能够对实时音频流进行分析和分类。
*   **可定制化**：可以根据需求训练模型以识别更多或不同的声音类别。
*   **完整教程**：提供从数据准备到部署的完整代码和步骤。

### 硬件需求 🛠️

1.  **ESP32-S3 开发板**：例如乐鑫官方 ESP32-S3-DevKitC-1 或其他兼容型号。
2.  **I2S 麦克风模块**：例如 INMP441, ICS43434, SPH0645 等。本项目视频中使用的是一个常见的 I2S MEMS 麦克风。
    *   确保麦克风的 `L/R` (或 `SEL`) 引脚正确配置以选择左声道或右声道（通常接地选左声道，接VCC选右声道，具体参考麦克风手册）。
3.  **USB 数据线**：用于供电和程序下载。
4.  **杜邦线若干**：用于连接麦克风和 ESP32-S3。

### 软件需求 💻

1.  **Arduino IDE**: 版本 1.8.19 或更高，或 VS Code + PlatformIO。
    *   **ESP32 Board Support Package**: 在 Arduino IDE 的开发板管理器中安装 ESP32 支持 (通常搜索 "esp32" by Espressif Systems)。
    *   **所需 Arduino 库**:
        *   `TensorFlow Lite for Microcontrollers` (通过 Arduino 库管理器安装最新版，通常搜索 "TensorFlowLite_ESP32")。
        *   `ArduinoFFT` (如果 `SamplingAndMelSpectrumAndTrain.ino` 中使用了此库进行 FFT，请安装)。
2.  **Python 环境**: Python 3.7 或更高。推荐使用 Anaconda 或 `venv` 创建虚拟环境。
    *   **所需 Python 包**:
        ```bash
        pip install -r requirements.txt
        ```
3.  **Git**: 用于克隆本项目仓库。推荐您先行 Fork 本项目，然后再克隆您 Fork 后的仓库。

### 📂 项目文件结构 (关键部分)

```
audiosort_tflm/
├── AudioSortCode/
│   ├── SamplingAndMelSpectrumAndTrain/  # 1. 用于采集音频并计算梅尔频谱图的Arduino Sketch
│   │   ├── SamplingAndMelSpectrumAndTrain.ino
│   │   └── sample/                        # 存放原始频谱数据(.txt)和处理后的(.npy)
│   │       ├── datasetSplit.py            # Python脚本: .txt -> .npy
│   │       ├── melSpectrumReader.py       # Python脚本: 可视化.npy频谱图
│   │       ├── aiModuleTrain.py           # Python脚本: 训练模型, 生成.tflite和.h
│   │       ├── requirements.txt           # Python依赖包列表
│   │       ├── (示例.txt 和 .npy 文件)
│   │       └── audio_classification_model.h  (训练后生成)
│   │       └── audio_classification_model.tflite (训练后生成)
│   ├── SamplingAndRecognize/            # 2. 最终部署在ESP32上进行实时识别的Sketch
│   │   ├── SamplingAndRecognize.ino
│   │   └── audio_classification_model.h   (从上面sample目录复制过来)
│   └── TFLM_TEST/                       # 3. 用于测试TFLM模型推理的简化Sketch
│       ├── TFLM_TEST.ino
│       └── audio_classification_model.h   (从上面sample目录复制过来)
│   ├── 01-TinyML ：在TensorFlow中为Arduino训练模型.md
│   ├── 02-TinyML 02：将 TensorFlow Lite模型部署到 Arduino.md
│   ├── 03-TinyML：使用TensorFlow Lite for Microcontrollers入门.md
│   ├── 05-ESP32实现神经网络音频分类.md
│   ├── 05-使用MTCNN和TensorFlow Lite进行ESP32-S3的面部检测.md
├── ArduinoSineFunction/                 # (可选) TFLM的正弦波演示示例
│   ├── ArduinoSineFunction.ino          # 主要的Arduino Sketch
│   ├── model.h                          # 示例模型头文件
│   └── TFLMArduinoSineFunction DEMO.md  # 示例说明文档
└── README.md                            # 本说明文件
```

### 🚀 手把手教程

#### 第零步：准备工作

1.  **克隆或下载项目**：
    ```bash
    git clone https://github.com/lff8888/audiosort_tflm.git
    cd audiosort_tflm
    ```
    或者直接从 GitHub 下载 ZIP 压缩包并解压。
2.  **安装 Arduino IDE 和 ESP32 支持**：
    *   下载并安装 [Arduino IDE](https://www.arduino.cc/en/software)。
    *   打开 Arduino IDE，进入 `文件 > 首选项`。
    *   在 `附加开发板管理器网址` 中添加 ESP32 的 URL：
        ```
        https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
        ```
    *   打开 `工具 > 开发板 > 开发板管理器`，搜索 "esp32" 并安装 "esp32 by Espressif Systems"。
    *   选择正确的开发板型号，例如 `ESP32S3 Dev Module`。
3.  **安装 Arduino 库**：
    *   打开 `工具 > 管理库`。
    *   搜索并安装 `TensorFlowLite_ESP32` (或 `TensorFlow Lite for Microcontrollers`，确保是 ESP32 兼容版本)。
    *   如果需要，搜索并安装 `ArduinoFFT`。
4.  **设置 Python 环境**：
    *   确保已安装 Python。
    *   在项目根目录（或 `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` 目录）打开终端，安装必要的包：
        ```bash
        pip install -r AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/requirements.txt
        ```

#### 第一步：数据采集与梅尔频谱图特征提取 (在 ESP32 上) 🎤

此步骤的目的是利用 ESP32 自身来采集音频并计算梅尔频谱图，确保训练时使用的特征提取方法与最终设备端推理时完全一致。

1.  **连接 I2S 麦克风到 ESP32-S3**：
    *   查阅您的 ESP32-S3 开发板引脚图和 I2S 麦克风模块手册。
    *   通常连接：
        *   `SCK` (BCLK) -> ESP32 I2S_SCLK 引脚
        *   `WS` (LRCK) -> ESP32 I2S_LRCK 引脚
        *   `SD` (DOUT) -> ESP32 I2S_SDIN 引脚
        *   `GND` -> ESP32 GND
        *   `VDD` -> ESP32 3.3V
    *   打开 `AudioSortCode/SamplingAndMelSpectrumAndTrain/SamplingAndMelSpectrumAndTrain.ino`。
    *   **重要**: 根据您的接线修改 Sketch 中的 I2S 引脚定义：
        ```cpp
        // I2S PINS
        #define I2S_SCLK_PIN 4  // 根据你的接线修改
        #define I2S_LRCK_PIN 41 // 根据你的接线修改
        #define I2S_SDIN_PIN 5  // 根据你的接线修改
        ```
2.  **上传 Sketch 并采集数据**：
    *   在 Arduino IDE 中打开 `SamplingAndMelSpectrumAndTrain.ino`。
    *   选择正确的开发板 (`ESP32S3 Dev Module`) 和端口。
    *   点击 "上传" 按钮。
    *   上传成功后，打开 `工具 > 串口监视器`。
    *   **重要**: 将串口监视器的波特率设置为 `2000000` (与 Sketch 中的 `Serial.begin(2000000);` 一致)。
3.  **录制并保存梅尔频谱数据**：
    *   对着麦克风播放或发出特定类别的声音（例如，播放一段婴儿饥饿的哭声）。
    *   串口监视器会持续输出计算得到的梅尔频谱图数据，每1秒（由 `AUDIO_DURATION` 定义）输出一组，以 "---" 分隔。
    *   **复制**串口监视器中对应声音类别的所有梅尔频谱图数据（包括 "---" 分隔符）。
    *   在 `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` 目录下创建一个文本文件，例如 `hunger.txt`。
    *   将复制的数据**粘贴**到 `hunger.txt` 中并保存。
    *   对您想要识别的**每个声音类别**重复此过程 (例如，创建 `sleepy.txt`, `discomfort.txt`, `burp.txt`, `nothing.txt`, `xiaoxin.txt`)。每个类别建议采集至少20-30个样本（即20-30个 "---" 分隔的数据块）。

#### 第二步：数据集预处理 (使用 Python 脚本) 🐍

此步骤将上一步采集到的 `.txt` 文件中的梅尔频谱数据转换为模型训练所需的 `.npy` 格式。

1.  **运行 `datasetSplit.py`**：
    *   确保您的 `.txt` 文件 (如 `hunger.txt`, `sleepy.txt` 等) 已经保存在 `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` 目录下。
    *   打开终端，导航到 `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` 目录。
    *   运行脚本：
        ```bash
        python datasetSplit.py
        ```
    *   脚本会自动读取当前目录下的所有 `.txt` 文件。对于每个 `.txt` 文件（例如 `hunger.txt`），它会：
        *   创建一个与文件名同名的子文件夹 (例如 `hunger/`)。
        *   将 `.txt` 文件中的每个梅尔频谱图样本（由 "---" 分隔）转换为一个 `.npy` 文件，并保存在对应的子文件夹中 (例如 `hunger/hunger0000.npy`, `hunger/hunger0001.npy` ...)。
2.  **(可选) 可视化检查梅尔频谱图**：
    *   运行 `melSpectrumReader.py` 来查看生成的 `.npy` 文件是否正确：
        ```bash
        python melSpectrumReader.py
        ```
    *   在弹出的GUI中，点击 "Load Directory" 并选择包含类别子文件夹的 `sample/` 目录。然后您可以浏览各个频谱图。

#### 第三步：模型训练 (使用 Python 脚本) 🧠

现在我们将使用处理好的 `.npy` 数据来训练我们的 CNN 模型。

1.  **运行 `aiModuleTrain.py`**：
    *   确保您位于 `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` 目录下。
    *   脚本会自动从当前目录下的子文件夹中加载数据进行训练。
    *   运行脚本：
        ```bash
        python aiModuleTrain.py
        ```
    *   脚本会：
        *   加载所有 `.npy` 数据和标签。
        *   构建 CNN 模型。
        *   训练模型，并在终端输出每个 epoch 的准确率和损失。
        *   显示训练过程中的准确率和损失曲线图。
        *   训练完成后，会在当前目录 (`sample/`) 下生成两个关键文件：
            *   `audio_classification_model.tflite`: TensorFlow Lite 模型文件。
            *   `audio_classification_model.h`: 包含模型数据的 C 头文件。
    *   **参数调整**: 您可以打开 `aiModuleTrain.py` 修改 `EPOCHS` (训练轮数) 和 `BATCH_SIZE` (批处理大小) 等参数以获得更好的性能。视频中提到 `EPOCHS = 500`, `BATCH_SIZE = 300` 时效果较好。

#### 第四步：模型部署与测试 (在 ESP32 上) 🚀

现在我们将训练好的模型部署到 ESP32-S3 上进行实时推理。

1.  **准备模型头文件**：
    *   将上一步生成的 `audio_classification_model.h` 文件从 `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` 目录复制到：
        *   `AudioSortCode/TFLM_TEST/` 目录下 (用于简化测试)
        *   `AudioSortCode/SamplingAndRecognize/` 目录下 (用于最终应用)

2.  **选项 A: 使用 `TFLM_TEST.ino`进行基本推理测试**：
    *   在 Arduino IDE 中打开 `AudioSortCode/TFLM_TEST/TFLM_TEST.ino`。
    *   这个 Sketch 使用一个**硬编码**在代码中的梅尔频谱图样本进行推理。
    *   编译并上传到 ESP32-S3。
    *   打开串口监视器 (波特率 `2000000`)。
    *   您应该能看到模型对这个硬编码样本的分类结果和推理时间 (通常几毫秒)。这能验证模型加载和基本推理流程是否正常。

3.  **选项 B: 使用 `SamplingAndRecognize.ino` 进行实时识别** (最终应用)：
    *   在 Arduino IDE 中打开 `AudioSortCode/SamplingAndRecognize/SamplingAndRecognize.ino`。
    *   **重要**: 再次检查并确保此 Sketch 中的 I2S 引脚定义与您的硬件连接一致 (与第一步中 `SamplingAndMelSpectrumAndTrain.ino` 的引脚配置应相同)。
        ```cpp
        // I2S PINS
        #define I2S_SCLK_PIN 4  // 根据你的接线修改
        #define I2S_LRCK_PIN 41 // 根据你的接线修改
        #define I2S_SDIN_PIN 5  // 根据你的接线修改
        ```
    *   编译并上传到 ESP32-S3。
    *   打开串口监视器 (波特率 `2000000`)。
    *   现在，对着麦克风发出声音，您应该能在串口监视器中看到实时的分类结果，显示每个类别的概率。

### 🎉 预期效果

当运行 `SamplingAndRecognize.ino` 时，串口监视器会实时输出类似以下格式的信息：

```
discomfort: 0.01 nothing: 0.85 burp: 0.02 xiaoxin: 0.01 sleepy: 0.05 hunger: 0.06
```

当您播放特定类型的婴儿哭声时，对应类别的概率应该会显著升高。

### 💡 注意事项与提示

*   **波特率**：所有 Arduino Sketch 的串口通信波特率均设置为 `2000000`，请确保串口监视器也使用此波特率。
*   **I2S 引脚**：务必根据您的实际硬件连接修改 Sketch 中的 I2S 引脚定义。
*   **数据质量和数量**：模型性能高度依赖于训练数据的质量和数量。尽量采集清晰、有代表性的音频样本，每个类别至少20-30个样本，越多越好。
*   **背景噪音**：`nothing` 类别对于区分有效声音和背景噪音非常重要。采集一些典型的环境噪音作为 `nothing` 类的样本。
*   **唤醒词 `xiaoxin`**：视频中提到 `xiaoxin` 唤醒词因为主要由演讲者本人录制，所以对其他人的声音可能不敏感。要提高泛化能力，需要更多不同人的语音数据，或者将普通说话声也加入 `nothing` 类别。
*   **内存限制 (`kTensorArenaSize`)**: 如果模型较大或 ESP32 内存不足，可能会遇到 `Failed to allocate tensors!` 错误。您可以尝试：
    *   在 `aiModuleTrain.py` 中构建更小的模型。
    *   在 Arduino Sketch 中增加 `kTensorArenaSize` 的值 (例如 `const int kTensorArenaSize = 16 * 1024;` 或更大)，但这受限于 ESP32 的可用 RAM。
*   **结果平滑**：为了提高识别的稳定性，可以在设备端代码中加入简单的滤波器，例如连续3-4次识别结果一致才最终确认。

---

恭喜您完成了整个流程！希望这个项目能帮助您入门嵌入式机器学习。如果您有任何问题或建议，欢迎提出 Issue。

<div id="english-content">

</div>

<div lang="en">
<p align="center">
  <img src="https://storage.googleapis.com/gweb-developer-goog-blog-assets/images_archive/original_images/image1_v7xhr8h.png" alt="TensorFlow Lite Micro Logo" width="200"/>
</p>

<h1 align="center">ESP32-S3 Baby Cry Recognition and Classification (TensorFlow Lite Micro)</h1>

<p align="center">
  <strong>A project to implement real-time baby cry recognition and classification on an ESP32-S3 microcontroller using TensorFlow Lite Micro.</strong>
</p>

<p align="center">
  <a href="https://www.bilibili.com/video/BV1uX8veJEGi" target="_blank">
    <img src="https://img.shields.io/badge/Watch%20Bilibili%20Video%20Tutorial-►%20Click%20Here-brightgreen?style=for-the-badge&logo=bilibili" alt="Bilibili Video Tutorial">
  </a>
</p>

<p align="center">
  <a href="#chinese-content"><img src="https://img.shields.io/badge/语言-中文-blue.svg?style=flat-square" alt="中文"></a>
  <a href="#english-content"><img src="https://img.shields.io/badge/Language-English-green.svg?style=flat-square" alt="English"></a>
</p>

---

### 📖 Project Overview

This project demonstrates how to deploy a TensorFlow Lite Micro (TFLM) model on a resource-constrained ESP32-S3 microcontroller for real-time recognition and classification of baby cries. By analyzing the cries, the system can help parents or caregivers make an initial assessment of a baby's potential needs, such as:

*   `discomfort`
*   `burp`
*   `sleepy`
*   `hunger`
*   As well as a `nothing` (no specific sound/background noise) category and a custom wake-word `xiaoxin`.

The entire workflow covers audio data collection, feature extraction (Mel Spectrogram), model training, model conversion, and final deployment on the ESP32-S3.

### ✨ Key Features

*   **Edge AI**: Audio processing and machine learning inference directly on the ESP32-S3, no cloud dependency.
*   **Low-Cost Solution**: Uses common ESP32-S3 development boards and I2S microphone modules.
*   **Real-Time Response**: Capable of analyzing and classifying live audio streams.
*   **Customizable**: The model can be trained to recognize more or different sound categories as needed.
*   **Complete Tutorial**: Provides full code and steps from data preparation to deployment.

### Hardware Requirements 🛠️

1.  **ESP32-S3 Development Board**: e.g., Espressif official ESP32-S3-DevKitC-1 or other compatible models.
2.  **I2S Microphone Module**: e.g., INMP441, ICS43434, SPH0645. The video tutorial uses a common I2S MEMS microphone.
    *   Ensure the microphone's `L/R` (or `SEL`) pin is correctly configured to select the left or right channel (usually GND for left, VCC for right; refer to your microphone's datasheet).
3.  **USB Cable**: For power and program flashing.
4.  **Dupont Wires**: For connecting the microphone to the ESP32-S3.

### Software Requirements 💻

1.  **Arduino IDE**: Version 1.8.19 or later, or VS Code + PlatformIO.
    *   **ESP32 Board Support Package**: Install ESP32 support in Arduino IDE's Board Manager (usually search "esp32" by Espressif Systems).
    *   **Required Arduino Libraries**:
        *   `TensorFlow Lite for Microcontrollers` (install the latest version via Arduino Library Manager, often found by searching "TensorFlowLite_ESP32").
        *   `ArduinoFFT` (if used in `SamplingAndMelSpectrumAndTrain.ino` for FFT, please install).
2.  **Python Environment**: Python 3.7 or higher. Anaconda or `venv` for virtual environments is recommended.
    *   **Required Python Packages**:
        ```bash
        pip install -r requirements.txt
        ```
3.  **Git**: For cloning this project repository. It is recommended to first fork this project and then clone your forked repository.

### 📂 Project File Structure (Key Parts)

```
audiosort_tflm/
├── AudioSortCode/
│   ├── SamplingAndMelSpectrumAndTrain/  # 1. Arduino Sketch for audio sampling & Mel spectrogram calculation
│   │   ├── SamplingAndMelSpectrumAndTrain.ino
│   │   └── sample/                        # Stores raw spectrum data (.txt) & processed (.npy)
│   │       ├── datasetSplit.py            # Python script: .txt -> .npy
│   │       ├── melSpectrumReader.py       # Python script: Visualize .npy spectrograms
│   │       ├── aiModuleTrain.py           # Python script: Train model, generate .tflite & .h
│   │       ├── requirements.txt           # Python dependency list
│   │       ├── (example .txt and .npy files)
│   │       └── audio_classification_model.h  (generated after training)
│   │       └── audio_classification_model.tflite (generated after training)
│   ├── SamplingAndRecognize/            # 2. Final Sketch for real-time recognition on ESP32
│   │   ├── SamplingAndRecognize.ino
│   │   └── audio_classification_model.h   (copied from 'sample/' directory above)
│   └── TFLM_TEST/                       # 3. Simplified Sketch for testing TFLM model inference
│       ├── TFLM_TEST.ino
│       └── audio_classification_model.h   (copied from 'sample/' directory above)
├── ArduinoSineFunction/                 # (Optional) TFLM Sine wave demo example
│   ├── ArduinoSineFunction.ino          # Main Arduino Sketch
│   ├── model.h                          # Example model header file
│   └── TFLMArduinoSineFunction DEMO.md  # Example documentation
└── README.md                            # This readme file
```

### 🚀 Step-by-Step Tutorial

#### Step 0: Preparations

1.  **Clone or Download the Project**:
    ```bash
    git clone https://github.com/lff8888/audiosort_tflm.git
    cd audiosort_tflm
    ```
    Or download the ZIP archive from GitHub and extract it.
2.  **Install Arduino IDE and ESP32 Support**:
    *   Download and install [Arduino IDE](https://www.arduino.cc/en/software).
    *   Open Arduino IDE, go to `File > Preferences`.
    *   In `Additional Boards Manager URLs`, add the ESP32 URL:
        ```
        https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
        ```
    *   Open `Tools > Board > Boards Manager`, search for "esp32" and install "esp32 by Espressif Systems".
    *   Select the correct board model, e.g., `ESP32S3 Dev Module`.
3.  **Install Arduino Libraries**:
    *   Open `Tools > Manage Libraries...`.
    *   Search for and install `TensorFlowLite_ESP32` (or `TensorFlow Lite for Microcontrollers`, ensure it's ESP32 compatible).
    *   If needed, search for and install `ArduinoFFT`.
4.  **Set up Python Environment**:
    *   Ensure Python is installed.
    *   Open a terminal in the project root (or `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` directory) and install necessary packages:
        ```bash
        pip install -r AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/requirements.txt
        ```

#### Step 1: Data Acquisition & Mel Spectrogram Feature Extraction (on ESP32) 🎤

The purpose of this step is to use the ESP32 itself to collect audio and compute Mel spectrograms, ensuring that the feature extraction method used during training is identical to that used during on-device inference.

1.  **Connect I2S Microphone to ESP32-S3**:
    *   Consult your ESP32-S3 development board's pinout diagram and your I2S microphone module's datasheet.
    *   Typical connections:
        *   `SCK` (BCLK) -> ESP32 I2S_SCLK pin
        *   `WS` (LRCK) -> ESP32 I2S_LRCK pin
        *   `SD` (DOUT) -> ESP32 I2S_SDIN pin
        *   `GND` -> ESP32 GND
        *   `VDD` -> ESP32 3.3V
    *   Open `AudioSortCode/SamplingAndMelSpectrumAndTrain/SamplingAndMelSpectrumAndTrain.ino`.
    *   **Important**: Modify the I2S pin definitions in the Sketch according to your wiring:
        ```cpp
        // I2S PINS
        #define I2S_SCLK_PIN 4  // Modify according to your wiring
        #define I2S_LRCK_PIN 41 // Modify according to your wiring
        #define I2S_SDIN_PIN 5  // Modify according to your wiring
        ```
2.  **Upload Sketch and Collect Data**:
    *   Open `SamplingAndMelSpectrumAndTrain.ino` in Arduino IDE.
    *   Select the correct board (`ESP32S3 Dev Module`) and port.
    *   Click the "Upload" button.
    *   After successful upload, open `Tools > Serial Monitor`.
    *   **Important**: Set the Serial Monitor's baud rate to `2000000` (to match `Serial.begin(2000000);` in the Sketch).
3.  **Record and Save Mel Spectrogram Data**:
    *   Play or make a specific category of sound into the microphone (e.g., play a recording of a baby's "hungry" cry).
    *   The Serial Monitor will continuously output the calculated Mel spectrogram data, one set per second (defined by `AUDIO_DURATION`), separated by "---".
    *   **Copy** all Mel spectrogram data for that sound category from the Serial Monitor (including the "---" separators).
    *   Create a text file in the `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` directory, e.g., `hunger.txt`.
    *   **Paste** the copied data into `hunger.txt` and save it.
    *   Repeat this process for **each sound category** you want to recognize (e.g., create `sleepy.txt`, `discomfort.txt`, `burp.txt`, `nothing.txt`, `xiaoxin.txt`) It's recommended to collect at least 20-30 samples (i.e., 20-30 "---" separated data blocks) per category.

#### Step 2: Dataset Preprocessing (using Python Scripts) 🐍

This step converts the Mel spectrogram data from the `.txt` files collected in the previous step into the `.npy` format required for model training.

1.  **Run `datasetSplit.py`**:
    *   Ensure your `.txt` files (e.g., `hunger.txt`, `sleepy.txt`) are saved in the `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` directory.
    *   Open a terminal and navigate to the `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` directory.
    *   Run the script:
        ```bash
        python datasetSplit.py
        ```
    *   The script will automatically read all `.txt` files in the current directory. For each `.txt` file (e.g., `hunger.txt`), it will:
        *   Create a subdirectory with the same name as the file (e.g., `hunger/`).
        *   Convert each Mel spectrogram sample (separated by "---") in the `.txt` file into an individual `.npy` file and save it in the corresponding subdirectory (e.g., `hunger/hunger0000.npy`, `hunger/hunger0001.npy` ...).
2.  **(Optional) Visualize and Check Mel Spectrograms**:
    *   Run `melSpectrumReader.py` to check if the generated `.npy` files are correct:
        ```bash
        python melSpectrumReader.py
        ```
    *   In the GUI that pops up, click "Load Directory" and select the `sample/` directory containing the class subfolders. You can then browse through the individual spectrograms.

#### Step 3: Model Training (using Python Script) 🧠

Now we will use the processed `.npy` data to train our CNN model.

1.  **Run `aiModuleTrain.py`**:
    *   Ensure you are in the `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` directory.
    *   The script will automatically load data from subfolders in the current directory for training.
    *   Run the script:
        ```bash
        python aiModuleTrain.py
        ```
    *   The script will:
        *   Load all `.npy` data and labels.
        *   Build the CNN model.
        *   Train the model, printing accuracy and loss for each epoch to the terminal.
        *   Display plots of training accuracy and loss.
        *   Upon completion, it will generate two key files in the current directory (`sample/`):
            *   `audio_classification_model.tflite`: The TensorFlow Lite model file.
            *   `audio_classification_model.h`: The C header file containing the model data.
    *   **Parameter Tuning**: You can open `aiModuleTrain.py` to modify parameters like `EPOCHS` (number of training rounds) and `BATCH_SIZE` to potentially achieve better performance. The video tutorial mentions that `EPOCHS = 500`, `BATCH_SIZE = 300` yielded good results.

#### Step 4: Model Deployment & Testing (on ESP32) 🚀

Now we'll deploy the trained model onto the ESP32-S3 for real-time inference.

1.  **Prepare the Model Header File**:
    *   Copy the `audio_classification_model.h` file generated in the previous step from the `AudioSortCode/SamplingAndMelSpectrumAndTrain/sample/` directory to:
        *   The `AudioSortCode/TFLM_TEST/` directory (for simplified testing).
        *   The `AudioSortCode/SamplingAndRecognize/` directory (for the final application).

2.  **Option A: Basic Inference Test with `TFLM_TEST.ino`**:
    *   Open `AudioSortCode/TFLM_TEST/TFLM_TEST.ino` in Arduino IDE.
    *   This Sketch uses a Mel spectrogram sample **hardcoded** in the code for inference.
    *   Compile and upload to your ESP32-S3.
    *   Open the Serial Monitor (baud rate `2000000`).
    *   You should see the model's classification result for this hardcoded sample and the inference time (usually a few milliseconds). This verifies that the model loading and basic inference pipeline are working correctly.

3.  **Option B: Real-Time Recognition with `SamplingAndRecognize.ino`** (Final Application):
    *   Open `AudioSortCode/SamplingAndRecognize/SamplingAndRecognize.ino` in Arduino IDE.
    *   **Important**: Double-check and ensure the I2S pin definitions in this Sketch match your hardware connections (should be the same as configured in Step 1 for `SamplingAndMelSpectrumAndTrain.ino`).
        ```cpp
        // I2S PINS
        #define I2S_SCLK_PIN 4  // Modify according to your wiring
        #define I2S_LRCK_PIN 41 // Modify according to your wiring
        #define I2S_SDIN_PIN 5  // Modify according to your wiring
        ```
    *   Compile and upload to your ESP32-S3.
    *   Open the Serial Monitor (baud rate `2000000`).
    *   Now, make sounds into the microphone. You should see real-time classification results in the Serial Monitor, showing the probability for each class.

### 🎉 Expected Outcome

When running `SamplingAndRecognize.ino`, the Serial Monitor will output information in a format similar to this in real-time:

```
discomfort: 0.01 nothing: 0.85 burp: 0.02 xiaoxin: 0.01 sleepy: 0.05 hunger: 0.06
```

When you play a specific type of baby cry, the probability for the corresponding class should increase significantly.

### 💡 Notes and Tips

*   **Baud Rate**: The serial communication baud rate for all Arduino Sketches is set to `2000000`. Ensure your Serial Monitor is also set to this baud rate.
*   **I2S Pins**: Always modify the I2S pin definitions in the Sketches according to your actual hardware connections.
*   **Data Quality and Quantity**: Model performance heavily depends on the quality and quantity of training data. Try to collect clear, representative audio samples, with at least 20-30 samples per category – more is better.
*   **Background Noise**: The `nothing` category is crucial for distinguishing valid sounds from background noise. Collect samples of typical ambient noise for the `nothing` class.
*   **Wake-word `xiaoxin`**: The video mentions that the `xiaoxin` wake-word, primarily recorded by the speaker, might not be sensitive to other people's voices. To improve generalization, more voice data from different individuals is needed, or common speech sounds could also be added to the `nothing` category.
*   **Memory Constraints (`kTensorArenaSize`)**: If your model is large or the ESP32 runs out of memory, you might encounter a `Failed to allocate tensors!` error. You can try:
    *   Building a smaller model in `aiModuleTrain.py`.
    *   Increasing the value of `kTensorArenaSize` in the Arduino Sketch (e.g., `const int kTensorArenaSize = 16 * 1024;` or larger), but this is limited by the ESP32's available RAM.
*   **Result Smoothing**: To improve recognition stability, you can add a simple filter in the on-device code, e.g., confirming a result only if it's recognized consistently 3-4 times in a row.

---

Congratulations on completing the entire process! Hopefully, this project helps you get started with embedded machine learning. If you have any questions or suggestions, feel free to open an Issue.

</div>