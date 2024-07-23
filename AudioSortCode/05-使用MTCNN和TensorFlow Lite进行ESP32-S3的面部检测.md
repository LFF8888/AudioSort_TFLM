[Face Detection with MTCNN and TensorFlow Lite for ESP32-S3 - Hackster.io](https://www.hackster.io/mauriciobarroso/face-detection-with-mtcnn-and-tensorflow-lite-for-esp32-s3-30b242)

# 使用MTCNN和TensorFlow Lite进行ESP32-S3的面部检测

这是一个使用[[MTCNN|多任务级联卷积网络]]（[[MTCNN]]，[[MTCNN|Multitask Cascading Convolutional Networks]]）和[[TensorFlow Lite|TensorFlow Lite]]（[[TensorFlow Lite]]，[[TensorFlow Lite|TensorFlow Lite]]）在[[ESP32-S3]]上检测和对齐人脸的实现。

## 中级
提供完整的说明
耗时：1小时
访问量：2,071

## 项目中使用的组件

### 硬件组件
- Espressif [[ESP32-S3-DevKitM-1-N8R8|ESP32-S3-开发板M-1-N8R8]] × 1
- Espressif [[ESP-LyraP-CAM v1.1|ESP-LyraP-CAM v1.1]] × 1

### 软件应用和在线服务
- [[TensorFlow|TensorFlow]]（[[TensorFlow]]，[[TensorFlow|TensorFlow]]）
- [[Google Colab|Google Colab]]（[[Google Colab]]，[[Google Colab|Google Colab]]）
- [[Espressif ESP-IDF|Espressif ESP-IDF]]（[[ESP-IDF]]，[[Espressif ESP-IDF|Espressif ESP-IDF]]）

### 手工工具和制造设备
- 面包板，普通面包板

## 故事

### 介绍
不久前，我第一次听到“嵌入式机器学习”这个术语，在那之前，我从未想到过可以在微控制器上运行人工智能模型。稍后，我开始了一个关于使用[[TensorFlow|TensorFlow]]进行[[计算机视觉|Computer Vision]]（[[计算机视觉]]，[[计算机视觉|Computer Vision]]）的课程，从此我对图像处理和创建及部署AI模型所需的所有过程有了清晰的认识。这个项目是我对嵌入式系统和这些新知识的热情的结果。

### 为什么？
[[MTCNN|多任务级联卷积网络]]（[[MTCNN]]，[[MTCNN|Multitask Cascading Convolutional Networks]]）是一个用于面部检测和对齐的框架。它包含三个卷积网络阶段，能够识别人脸及眼睛、鼻子和嘴巴等标志性位置。

- **[[P-Net|提案网络]]（[[P-Net]]，[[P-Net|Proposal Network]]）**：这是一个[[全卷积网络|FCN]]（[[FCN]]，[[FCN|Fully Convolutional Network]]），用于获取候选窗口及其边界框回归向量。边界框回归是一种预测目标物体预定义类的盒子位置的流行技术。通过边界框回归向量校准候选窗口并使用[[非极大值抑制|NMS]]（[[NMS]]，[[NMS|Non-Maximum Suppression]]）操作来组合重叠区域。

- **[[R-Net|精炼网络]]（[[R-Net]]，[[R-Net|Refine Network]]）**：进一步减少候选数目，使用边界框回归进行校准，并使用[[非极大值抑制|NMS]]合并重叠候选。这是一个[[卷积神经网络|CNN]]（[[CNN]]，[[CNN|Convolutional Neural Network]]），而不是像[[P-Net]]那样的[[全卷积网络|FCN]]，因为其架构的最后阶段有一个密集层。

- **[[O-Net|输出网络]]（[[O-Net]]，[[O-Net|Output Network]]）**：与[[R-Net]]相似，但这个输出网络旨在更详细地描述面部，并输出眼睛、鼻子和嘴巴的五个面部标志位置。

## [[TensorFlow|TensorFlow]]实现

为了实现[[MTCNN|MTCNN]]模型，使用了[[TensorFlow|TensorFlow]]和[[Google Colab|Google Colab]]。[[TensorFlow|TensorFlow]]是一个为Google开发的开源[[机器学习|ML]]（[[ML]]，[[ML|Machine Learning]]）库，能够构建和训练神经网络以检测模式和相关性。[[Google Colab|Google Colab]]是Google Research的产品，允许通过浏览器编写和执行任意[[Python|Python]]代码，特别适合于[[机器学习|ML]]、数据分析和教育。

要正确实现[[MTCNN|MTCNN]]，模型的输入和输出数据必须经过处理以确保最佳结果。下图显示了所实现管道的块图。

### 实现步骤
1. 进行图像金字塔操作，创建不同尺度的输入图像，检测不同大小的面部。新的缩放图像作为[[P-Net]]的输入，生成每个候选窗口的偏移和得分。然后这些输出进行后处理以获得面部位置的坐标。[[R-Net]]的输入必须使用之前的输出进行预处理，从而获得新的候选窗口。[[R-Net]]的输出是候选窗口的偏移和得分，进行后处理以获得面部位置的新的坐标。最后，对于[[O-Net]]，重复[[R-Net]]过程，获得输入图像中面部的坐标。

2. 预处理包括两个步骤：根据先前获得的边界框坐标裁剪输入图像，并调整裁剪图像的大小以匹配模型的输入形状。

3. 后处理包括三个步骤：应用[[非极大值抑制|NMS]]来合并重叠区域，使用之前获得的偏移校准边界框，平方并校正最终的边界框坐标。

所有上述过程以及[[TensorFlow]]、[[TensorFlow Lite]]和[[TFLM|TensorFlow Lite Micro]]的模型都在下面的[[Google Colab]]笔记本中进行了开发。

### 部署到[[ESP32-S3]]

模型开发的最后一步是创建所有[[MTCNN]]模型的.c文件和模型设置的.h文件，这些文件位于main/models/目录下。使用C/C++在utils.cc和utils.h文件中实现了[[MTCNN]]管道所需的预处理和后处理函数，这些文件位于main/目录下。

硬件包括[[ESP32-S3-DevKitC-1-N8R8]]和一个[[OV2640]]摄像头模块。必须使用[[PSRAM|PSRAM]]（[[PSRAM]]，[[PSRAM|Pseudo Static RAM]]），也可以使用其他带[[PSRAM|PSRAM]]的[[ESP32-S3]]设备。

### 步骤
1. 下载并安装[[ESP-IDF]]  
这个项目使用[[ESP-IDF]] v5.0开发，所以必须使用该版本或更高版本。链接中包含了下载和安装的必要说明，请进行手动安装。

2. 克隆这个仓库：
```bash
git clone --recursive https://github.com/mauriciobarroso/mtcnn_esp32s3.git
```

3. 配置项目：
在menuconfig->App Configuration->Camera Configuration中更改摄像头引脚配置，在menuconfig->App Configuration->Wi-Fi Configuration中设置Wi-Fi凭据。
```bash
cd mtcnn_esp32s3/
idf.py set-target esp32s3
idf.py menuconfig
```

4. 刷写和监控  
这个项目不需要屏幕显示[[MTCNN]]生成的图像和边界框，而是使用控制台字符打印输出图像和其他相关信息。运行以下命令监控控制台输出：
```bash
idf.py flash monitor
```

控制台应打印如下内容：

```plaintext
ESP-ROM:esp32s3-20210327
Build:Mar 27 2021
rst:0x1 (POWERON),boot:0x8 (SPI_FAST_FLASH_BOOT)
SPIWP:0xee
mode:DIO, clock div:1
load:0x3fce3810,len:0x17d8
...
```

5. 测试  
设备连接到之前配置的网络后，将摄像头对准任何面部。控制台输出应打印类似如下内容：
```plaintext
P-Net time : 65, bboxes : 3
R-Net time : 232, bboxes : 3
O-Net time : 789, bboxes : 3
MTCNN time : 1088, bboxes : 3
```

要查看由摄像头捕获并由[[MTCNN]]处理的图像，访问以下链接：http://ip_address/faces.jpg，其中“ip_address”应替换为分配给设备的IP地址。它应显示类似于下图的结果。

### 特性
- 输入图像尺寸：96x96
- 输入图像格式：RGB888
- 检测距离：10 cm - 100 cm
- 响应时间（1张面部）：约1000毫秒

### 代码
在[[TensorFlow Lite Micro|TensorFlow Lite Micro]]（[[TensorFlow Lite Micro]]，[[TensorFlow Lite Micro|TensorFlow Lite Micro]]）中实现[[MTCNN]]

的代码
可以从以下链接获取：
[mauriciobarroso / mtcnn_esp32s3](https://github.com/mauriciobarroso/mtcnn_esp32s3)

## Credits
### 作者
Mauricio Barroso Benavides  
Bolivian嵌入式系统开发人员，主要致力于设计和实施物联网无线解决方案。  
2个项目 • 2个粉丝

### 评论
请登录或注册以评论。

用户 naijzam 评论：
> 我最近几周一直在玩你的alexa唤醒词检测代码。你做的真的很棒。你的代码可以正常工作，非常清晰！我还在尝试训练模型来识别不在Google列表中的自定义唤醒词。还没有成功，但离成功很近了。对于那些想要加速训练的人来说，Google colab（付费版，每月14美元）确实值得。它将训练时间从2小时缩短到几分钟（30个周期）。