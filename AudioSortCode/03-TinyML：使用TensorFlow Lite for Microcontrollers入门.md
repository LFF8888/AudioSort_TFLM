📌 原文链接：[https://www.digikey.com/...](https://www.digikey.com/en/maker/projects/tinyml-getting-started-with-tensorflow-lite-for-microcontrollers/c0cdd850f5004b098d263400aa294023)

视频教程：[(629) TinyML: Getting Started with TensorFlow Lite for Microcontrollers | Digi-Key Electronics - YouTube](https://www.youtube.com/watch?v=gDFWCxrJruQ)

 2020-07-06 | 作者：[ShawnHymel](https://www.digikey.com/en/maker/profiles/72825bdd887a427eaf8d960b6505adac)

TensorFlow是一个由Google开发的流行的开源软件库，用于执行[[ML|机器学习]]（[[ML]]，[[ML|Machine Learning]]）任务。该库的一个子集是[[TensorFlow Lite for Microcontrollers]]（[[TensorFlow Lite for Microcontrollers]]，[[TensorFlow Lite for Microcontrollers]]），它允许我们在微控制器上运行推理。请注意，“推理”只是使用模型进行预测、分类或决策，不包括训练模型。

由于[[ML|机器学习]]（尤其是神经网络和[[DL|深度学习]]（[[DL]]，[[DL|Deep Learning]]））计算量大，[[TensorFlow Lite for Microcontrollers]]需要使用32位处理器，例如ARM Cortex-M或ESP32。还需要注意的是，该库主要用C++编写，因此需要使用C++编译器。

# 概述

由于微控制器资源有限，我们通常需要先在计算机（或远程服务器）上进行模型训练。一旦有了模型，我们将其转换为[[FlatBuffer]]（[[FlatBuffer]]，[[FlatBuffer]]）(.tflite)文件，然后将该模型转换为一个常量C数组，并将其包含在我们的固件程序中。

![流程图](http://www.kdocs.cn/api/v3/office/copy/djFiY0VXYWtxZHBLT0JkWnQ0c0JVODA5d2E4SE0rSnhmTzRxTDUzMFVXKzFSUjRNMG5DeVVXZjEvVmV5Lzk1WmR2TnpMT2xCVUNtZzMwZzBpQ01JYVdMUDU4ZTB6eTlHcFhEeG5wR0pJQkpWZDIxM0Z0b1drZ1ZIelowNCt6RjZUcEtkSmJCOGo2WWVKUzFWWHlMNUZSQXA1bytaaEZPZHNiS3pBOGozK1lXSHlOOUxnOU5ONzNrTFhndzgrOUVjSnl3ZDdoZjZJNXhsZTdNVnl0eGp1bDc0RDV3ODd6N0tTeVBUWXJSRVRiamN6NjNTT3ZIeFZCWGkramZCNkNHMk9vT3pveW1MV3JVPQ==/attach/object/f1bd94ea820c9734a92e99b659bbbe27a7c4101d?)

在微控制器上，我们运行[[TensorFlow Lite for Microcontrollers]]库，该库使用我们的模型进行推理。例如，假设我们训练了一个模型来分类照片中是否有猫。如果我们在微控制器上使用该模型，可以输入未见过的数据（即照片），它会告诉我们照片中是否有猫。

![](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/ad803b0c5114f94eac738e94987982d2dbbf2d6f?)

# 模型训练

我们将使用之前教程中训练的模型。请按照[先前教程](https://www.digikey.com/en/maker/projects/intro-to-tinyml-part-1-training-a-model-for-arduino-in-tensorflow/8f1fc8c0b83d417ab521c48864d2a8ec)中的步骤生成.h5、.tflite和.h格式的模型文件。将所有三个文件下载到计算机。

![](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/5b07207873df3e44b903552049b6109a44633918?)

# 开发环境

本教程将向您展示如何在[[TensorFlow Lite]]中生成源代码文件，您可以将这些文件用作任何微控制器构建系统（Arduino, make, Eclipse等）的库。然而，我将专门向您展示如何在[[STM32CubeIDE]]中包含该库。原因如下：我目前对该[[IDE]]最熟悉，并且我们可以在下一个教程中并排比较[[TensorFlow Lite]]和[[STM32 X-Cube-AI]]库。

我将在[Nucleo-L432KC](https://www.digikey.com/product-detail/en/stmicroelectronics/NUCLEO-L432KC/497-16592-ND/6132763)开发板上运行此演示。

![开发板](http://www.kdocs.cn/api/v3/office/copy/djFiY0VXYWtxZHBLT0JkWnQ0c0JVODA5d2E4SE0rSnhmTzRxTDUzMFVXKzFSUjRNMG5DeVVXZjEvVmV5Lzk1WmR2TnpMT2xCVUNtZzMwZzBpQ01JYVdMUDU4ZTB6eTlHcFhEeG5wR0pJQkpWZDIxM0Z0b1drZ1ZIelowNCt6RjZUcEtkSmJCOGo2WWVKUzFWWHlMNUZSQXA1bytaaEZPZHNiS3pBOGozK1lXSHlOOUxnOU5ONzNrTFhndzgrOUVjSnl3ZDdoZjZJNXhsZTdNVnl0eGp1bDc0RDV3ODd6N0tTeVBUWXJSRVRiamN6NjNTT3ZIeFZCWGkramZCNkNHMk9vT3pveW1MV3JVPQ==/attach/object/ea770541803065c0e6607995ea7b2e1573662aca?)

观看[此视频以了解STM32CubeIDE的入门](https://www.youtube.com/watch?v=hyZS2p1tW-g)。

# 生成[[TensorFlow Lite]]文件结构

[[TensorFlow]]的创建者希望您使用Make构建工具生成多个示例项目，这些项目可以用

作微控制器的模板。虽然这可以很好地工作，但我想展示如何将[[TensorFlow Lite]]用作库，而不是启动项目。

注意，接下来的部分需要使用Linux或macOS。我尚未在Windows中使[[TensorFlow Lite]]项目的自动生成工作。我能够在[[Raspberry Pi]]上完成这部分工作。

在一个新的终端中，安装以下内容：

```shell
sudo apt update
sudo apt install make git python3.7 python3-pip zip
```

注意，您可能需要为一些[[TensorFlow Lite]]的Make命令添加python3和pip3的别名。在`~/.bashrc`中，在末尾添加以下内容：

```shell
alias python=python3
alias pip=pip3
```

我们将获取[[TensorFlow]]库的最新版本（使用深度为1，以仅获取最新版本）：

```shell
git clone --depth 1 https://github.com/tensorflow/tensorflow.git
```

进入tensorflow目录并运行[[TensorFlow Lite for Microcontrollers]]目录中的Makefile：

```shell
cd tensorflow
make -f tensorflow/lite/micro/tools/make/Makefile TAGS="portable_optimized" generate_non_kernel_projects
```

这将花费几分钟时间，请耐心等待。它会生成许多示例项目和源代码，供您作为起点使用。

# 创建项目

在[[STM32CubeIDE]]中，为您的Nucleo-L432KC板创建一个新的STM32项目。为您的项目命名，并确保选择C++作为目标语言。

![](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/acbb51144bd2171c119be8c1999a3b67ae7938d8?)

启用定时器16（TIM16），并将其预分频器设为（80 - 1），以便它每微秒计数一次，并将重载值设为（65536 - 1）。

![](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/8966be0d833e5b1f1490164819c203652ecf3acc?)

时钟配置中，选择**HSI**作为PLL源，并在HCLK框中输入“80”。按下‘回车’，[[CubeMX]]软件应该会自动配置所有系统时钟和预分频器，以使系统时钟达到80 MHz。

![](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/627bd5f682f0b756a3ae532a422df92f796e22a6?)

将main.c重命名为main.cpp。

![](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/5b27ca9534b83bb6f7eeadbd76f2fe4decf824fa?)

# 将模型和[[TensorFlow Lite]]源代码文件复制到您的项目中

找到您在训练步骤中生成并下载的_sine_model.h_文件。将该文件复制到_<your_project_directory>/Core/Inc_。它包含一个以[[FlatBuffer]]格式存储的神经网络模型的字节数组。 

![](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/9ee976b9785cc8ba343faf412c7af33c13b2da8f?)

进入_<tensorflow_repo>/tensorflow/lite/micro/tools/make/gen/<os_cpu>/prj/hello_world/，并复制tensorflow和third_party目录。请注意，<os_cpu>会根据您运行make命令的操作系统和CPU而变化（例如，我在[[Raspberry Pi]]上运行make，所以是linux_armv7l）。

![](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/b81cf20c997b2505624a62ee88e5fb97458a1f39?)

将它们粘贴到_<your_project_directory>/tensorflow_lite_中，根据需要创建_tensorflow_lite_目录。

进入_<your_project_directory>/tensorflow_lite/tensorflow/lite/micro_并删除examples文件夹，因为它包含我们不需要的模板_main.c_应用程序。可以随意查看它，以了解[[TensorFlow]]推荐的固件项目创建方法。

![删除示例文件](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/451e124a26e58ed5dd1742e9ab437e660bd2e881?)

在[[STM32CubeIDE]]中，右键单击您的项目并选择**刷新**。现在应该可以在项目中看到模型文件和tensorflow_lite目录。

![刷新项目](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/2d0adfeaf8ddb9bc5126b9a9c034578dc18d293f?)

# 在构建过程中包含头文件和源文件

即使源文件在我们的项目中，我们仍需要告诉[[IDE]]将它们包含在构建过程中。转到**项目 > 属性**。在该窗口中，转到**C/C++ General > Paths and Symbols > Includes tab > GNU C**。点击**添加**。在弹出窗口中，点击**工作区**。选择项目中的**tensorflow_lite**目录。选中**添加到所有配置**和**添加到所有语言**。

![添加路径](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/1fcf1344c2e3e0bcf80c9aa5fec24cfd4350b5d7?)

重复此过程以添加_tensorflow_lite/third_party_中的以下目录：

- flatbuffers/include
- gemmlowp
- ruy

![](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/08a3f0d2fb251b6390e20c29f528dd139e63de4a?)

前往**源位置**标签，并将_<your_project_directory>/tensorflow_lite_添加到**调试**和**发布**配置中。

![源位置](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/b71229e5812bd88a3ae9aae6b25bc1ba3f705126?)

点击**应用并关闭**和**是**（如果要求重建索引）。

**更新调试代码**

我们需要对[[TensorFlow]]库进行一个更改，以支持通过串行端口进行调试。打开_<your_project_directory>/tensorflow_lite/tensorflow/lite/micro/debug_log.cc_。

将代码更新为以下内容：

```cpp
#include "tensorflow/lite/micro/debug_log.h"

//#include <cstdio>
//
//extern "C" void DebugLog(const char* s) { fprintf(stderr, "%s", s); }

extern "C" void __attribute__((weak)) DebugLog(const char* s) {
  // To be implemented by user
}
```

我们注释掉了DebugLog的原始实现（因为我们不支持fprintf），并添加了带有弱属性的自己的实现。这允许我们在main.cpp中提供实际实现，我们可以使用STM32 HAL通过UART输出调试信息。

保存并关闭文件。

**编写主程序**

打开main.cpp，并在用户头文件保护之间添加以下部分（例如/* USER CODE BEGIN … */）。请注意，如果您使用的是不同的微控制器或开发板，自动生成的代码可能会有所变化。还请注意底部的DebugLog自定义实现，它覆盖了debug_log.cc中的定义。

```cpp
/* USER CODE BEGIN Header */
/**
******************************************************************************
* @file : main.c
* @brief : Main program body
******************************************************************************
* @attention
*
* <h2><center>&copy; Copyright (c) 2020 STMicroelectronics.
* All rights reserved.</center></h2>
*
* This software component is licensed by ST under BSD 3-Clause license,
* the "License"; You may not use this file except in compliance with the
* License. You may obtain a copy of the License at:
* opensource.org/licenses/BSD-3-Clause
*
******************************************************************************
*/
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"
#include "sine_model.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim16;
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

// TFLite 全局变量
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// 创建一个内存区域用于输入、输出和其他 TensorFlow 数组。
// 你需要通过编译、运行和查看错误来调整这个大小。
constexpr int kTensorArenaSize = 2 * 1024;
__attribute__((aligned(16))) uint8_t tensor_arena[kTensorArenaSize];
} // namespace

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM16_Init(void);
/* USER CODE BEGIN PFP */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* USER CODE END 0 */

/**
* @brief The application entry point.
* @retval int
*/
int main(void)
{
/* USER CODE BEGIN 1 */
char buf[50];
int buf_len = 0;
TfLiteStatus tflite_status;
uint32_t num_elements;
uint32_t timestamp;
float y_val;
/* USER CODE END 1 */

/* MCU Configuration--------------------------------------------------------*/
/* Reset of all peripherals, Initializes the Flash interface and the Systick. */
HAL_Init();

/* USER CODE BEGIN Init */
/* USER CODE END Init */

/* Configure the system clock */
SystemClock_Config();

/* USER CODE BEGIN SysInit */
/* USER CODE END SysInit */

/* Initialize all configured peripherals */
MX_GPIO_Init();
MX_USART2_UART_Init();
MX_TIM16_Init();
/* USER CODE BEGIN 2 */

// 启动定时器/计数器
HAL_TIM_Base_Start(&htim16);

// 设置日志记录（修改 tensorflow/lite/micro/debug_log.cc）
static tflite::MicroErrorReporter micro_error_reporter;
error_reporter = &micro_error_reporter;

// 打印测试信息以验证错误报告器
error_reporter->Report("STM32 TensorFlow Lite test");

// 将模型映射为可用的数据结构
model = tflite::GetModel(sine_model);
if (model->version() != TFLITE_SCHEMA_VERSION)
{
  error_reporter->Report("Model version does not match Schema");
  while(1);
}

// 仅导入所需的操作（应匹配神经网络层）。模板参数
// <n> 是要添加的操作数。可用操作：
// tensorflow/lite/micro/kernels/micro_ops.h
static tflite::MicroMutableOpResolver<1> micro_op_resolver;
// 添加全连接神经网络层操作
tflite_status = micro_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_FULLY_CONNECTED,
    tflite::ops::micro::Register_FULLY_CONNECTED());
if (tflite_status != kTfLiteOk)
{
  error_reporter->Report("Could not add FULLY CONNECTED op");
  while(1);
}

// 构建一个解释器以运行模型。
static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
interpreter = &static_interpreter;

// 从 tensor_arena 分配内存给模型的张量。
tflite_status = interpreter->AllocateTensors();
if (tflite_status != kTfLiteOk)
{
  error_reporter->Report("AllocateTensors() failed");
  while(1);
}

// 分配模型输入和输出缓冲区（张量）给指针
model_input = interpreter->input(0);
model_output = interpreter->output(0);

// 获取输入张量中的元素数量
num_elements = model_input->bytes / sizeof(float);
buf_len = sprintf(buf, "Number of input elements: %lu\r\n", num_elements);
HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

/* USER CODE END 2 */

/* Infinite loop */
/* USER CODE BEGIN WHILE */
while (1)
{
  // 填充输入缓冲区（使用测试值）
  for (uint32_t i = 0; i < num_elements; i++)
  {
    model_input->data.f[i] = 2.0f;
  }

  // 获取当前时间戳
  timestamp = htim16.Instance->CNT;

  // 运行推理
  tflite_status = interpreter->Invoke();
  if (tflite_status != kTfLiteOk)
  {
    error_reporter->Report("Invoke failed");
  }

  // 读取神经网络的输出（预测的 y 值）
  y_val = model_output->data.f[0];

  // 打印神经网络的输出以及推理时间（微秒）
  buf_len = sprintf(buf,
                    "Output: %f | Duration: %lu\r\n",
                    y_val,
                    htim16.Instance->CNT - timestamp);
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

  // 等待一段时间再重新执行
  HAL_Delay(500);
}
/* USER CODE END WHILE */
/* USER CODE BEGIN 3 */
}
/* USER CODE END 3 */
}

/**
* @brief System Clock Configuration
* @retval None
*/
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the CPU, AHB and APB busses clocks
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 10;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB busses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART2;
  PeriphClkInit.Usart2ClockSelection = RCC_USART2CLKSOURCE_PCLK1;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
* @brief TIM16 Initialization Function
* @param None
* @retval None
*/
static void MX_TIM16_Init(void)
{
  /* USER CODE BEGIN TIM16_Init 0 */
  /* USER CODE END TIM16_Init 0 */

  /* USER CODE BEGIN TIM16_Init 1 */
  /* USER CODE END TIM16_Init 1 */
  htim16.Instance = TIM16;
  htim16.Init.Prescaler = 80 - 1;
  htim16.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim16.Init.Period = 65536 - 1;
  htim16.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim16.Init.RepetitionCounter = 0;
  htim16.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim16) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM16_Init 2 */
  /* USER CODE END TIM16_Init 2 */
}

/**
* @brief USART2 Initialization Function
* @param None
* @retval None
*/
static void MX_USART2_UART_Init(void)
{
  /* USER CODE BEGIN USART2_Init 0 */
  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */
  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */
  /* USER CODE END USART2_Init 2 */
}

/**
* @brief GPIO Initialization Function
* @param None
* @retval None
*/
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : LD3_Pin */
  GPIO_InitStruct.Pin = LD3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD3_GPIO_Port, &GPIO_InitStruct);
}

/* USER CODE BEGIN 4 */
// 自定义 TensorFlow 的 DebugLog 实现
extern "C" void DebugLog(const char* s)
{
  HAL_UART_Transmit(&huart2, (uint8_t *)s, strlen(s), 100);
}
/* USER CODE END 4 */

/**
* @brief This function is executed in case of error occurrence.
* @retval None
*/
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* 用户可以在此处添加自己的代码来报告 HAL 错误返回状态 */
  /* USER CODE END Error_Handler_Debug */
}

#ifdef USE_FULL_ASSERT
/**
* @brief Reports the name of the source file and the source line number
* where the assert_param error has occurred.
* @param file: pointer to the source file name
* @param line: assert_param error line source number
* @retval None
*/
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* 用户可以在此处添加自己的代码来报告文件名和行号，
  例如：printf("错误的参数值：文件 %s 在第 %d 行\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/

```

请参考视频了解这些代码部分的作用。

**添加printf浮点支持**

在[[STM32CubeIDE]]中，printf（及其变体）不支持浮点值。要添加支持，需要前往**项目 > 属性 > C/C++构建 > 设置 > 工具设置标签 > MCU G++链接器杂项**。在其他标志面板中，添加以下行：

```shell
-u_printf_float
```

对**调试**和**发布**配置都执行此操作。

![添加浮点支持](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/b68237b9329ef0606e5cc5376b7f03d61197c2a3?)

**运行调试模式**

构建项目并点击**运行 > 调试**。在调试视图中，点击播放/暂停按钮开始在微控制器上运行代码。打开

串行终端（例如PuTTY）连接到开发板，应该可以看到推理输出，应该与Google Colab中的测试（估计sin(2.0)）匹配。还可以看到运行推理所需的时间（微秒）。

![调试模式](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/16b59fab0623b3c08506017c718ae8268d022542?)

**运行发布模式**

调试配置在g++编译期间包含-DDEBUG标志，这在[[TensorFlow Lite]]库中启用了一些选项（但会减慢速度）。

选择**项目 > 构建配置 > 设置活动 > 发布**。然后，选择**项目 > 构建**。

打开**运行 > 运行配置...**。在该窗口中，将C/C++应用程序设置为**Release/<your_project_name>.elf**。将构建配置设置为**发布**。

![发布模式](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/a39c7699406af38aa9c57091f3d65ac489416076?)

点击**运行**。项目应该会重新构建。如果查看控制台中的输出，可以估算程序的闪存和RAM使用情况。查找_arm-none-eabi-size_工具的输出。

文本+数据给出了闪存使用情况（在本例中约为50032字节），数据+ bss给出了估计的RAM使用情况（在本例中约为4744字节）。

![内存使用](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/1c31191d23053fa73cf704ff27b0dba0db768a3b?)

打开串行终端，应该会看到程序的输出。通过切换到发布配置（移除-DDEBUG标志），我们将推理时间从大约368微秒减少到大约104微秒。

![终端输出](http://www.kdocs.cn/api/v3/office/copy/NytVczFKdXFUTHU2eHZpaW82b21uR2lha0xjNDdWSXUyT3JXM2JydTVaN3o5SmpBYktoeURJaWRQYVVBems3UkU1Smdra21xWFBVUDhnMTUrTHgrZWdpZDM2OVRZR1ZEeHJ2QXhwa0RubTNIU3pKcWNSQk5DVlV3U3d6a0xwdEFtazM0cDE5TzZBSkdpT0xiOEJYQWpXZzRMdmw5T3hGcmdXNnd5akJtTDlZNWxIeTJjeC82UHNjYnFGS2VMOVFFWUs2T09xclZVRGgvZjFEZmRyOWkyMHVRb1JQMEJRS1R2Q0R5QWVwbUluTWpyY2gwWVE1TFJWeEVVV0dsaytYQWkyRnlYNTF5cUFJPQ==/attach/object/d09fe19ca0b83b621926c0cd0ec43641e97aca59?)

**更进一步**

希望这有助于您开始使用[[TensorFlow Lite for Microcontrollers]]！虽然示例程序不是特别有用（预测正弦值），但它应该提供一个不错的模板，以开始创建自己的项目。

以下是一些关于使用[[TensorFlow Lite for Microcontrollers]]的其他文章：

- [https://www.tensorflow.org/lite/microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [https://www.tensorflow.org/lite/microcontrollers/get_started](https://www.tensorflow.org/lite/microcontrollers/get_started)
- [将[[TensorFlow Lite]]部署到Arduino](https://en/maker/projects/intro-to-tinyml-part-2-deploying-a-tensorflow-lite-model-to-arduino/59bf2d67256f4b40900a3fa670c14330)
- [边缘AI异常检测](https://en/maker/projects/edge-ai-anomaly-detection-part-1-data-collection/7bb112f76ef644edaedc5e08dba5faae)

# 视频教程

# TinyML介绍 第1部分 - 使用TensorFlow为Arduino训练神经网络 | Digi-Key Electronics

2024年07月14日 17:54

## 00:09 - TinyML简介
[[TensorFlow|张量流]]（[[TensorFlow]]，[[TensorFlow|TensorFlow]]）是一个免费且开源的库，使各种机器学习算法的实现变得更容易。有很多关于如何开始使用它的教程，但我想展示一个运行在微控制器上的小众应用。[[TensorFlow Lite|Tensor流轻量版]]（[[TFLite]]，[[TensorFlow Lite|TensorFlow Lite]]）是[[TensorFlow|张量流]]库的一个子集，旨在运行于嵌入式设备上，而这个子集的一小部分是专为微控制器设计的。

## 00:35 - 微控制器与TinyML的工作流程
由于微控制器通常资源非常有限，因此一般不建议在其上训练机器学习模型。基本工作流程是使用全功能的[[TensorFlow|张量流]]在计算机或其他服务器上训练模型，然后将该模型转换为[[TensorFlow Lite|Tensor流轻量版]]平面缓冲文件（.tflite文件）。这个.tflite文件可以与常规的[[TensorFlow Lite|Tensor流轻量版]]一起使用，但你需要将其转换为微控制器所需语言的数组。我们将创建一个C数组，并将其传输到我们的微控制器上，并使用[[TensorFlow Lite for Microcontrollers|微控制器Tensor流轻量版]]库加载到内存中，这样我们就可以在微控制器上使用我们的模型进行推理。推理是指将模型未见过的新数据输入模型，并让模型推断出一些关于该数据的信息。例如，如果我们输入一张照片，一个训练来识别猫的模型会告诉我们照片中是否有猫。

## 01:31 - 预训练模型的使用
这一集中我们不会涉及神经网络的训练，因为这需要很多时间。有很多优秀的视频展示了如何进行训练。在这一集中，我们将从我在前一集中创建的预训练模型开始。请查看TinyML第1集视频，我会在描述中链接。在视频中，我展示了如何在[[Google Colab|谷歌协作平台]]上使用[[TensorFlow|张量流]]和[[Keras|凯拉斯]]（[[Keras]]，[[Keras|Keras]]）创建一个三层神经网络来预测正弦函数的输出。虽然在微控制器上创建正弦波是一种糟糕的方法，但它很好地展示了如何训练一个非常简单的神经网络。

## 02:14 - 模型测试
训练完模型后，我们会通过输入一些数字来测试它。具体来说，我会输入2，如果模型在我们的微控制器上正确运行，我们应该会看到输出0.9059，这应接近于sin(2)，大约是0.9093。接着，我们将[[Keras|凯拉斯]]模型保存为.h5文件。尽管这个文件对我们没多大用处，但我们首先需要通过调用tflite_converter.from_keras_model函数将其转换为[[TensorFlow Lite|Tensor流轻量版]]文件。注意，我们优化了模型大小，但仍然使用浮点值处理所有内容。还有一种方法可以将模型的输入、输出、权重和偏置项量化为8位，以节省更多空间，但这里不做讨论。最后，我编写了一个快速函数，将[[TensorFlow Lite|Tensor流轻量版]]文件转换为C字节数组，并将其保存为.h文件。

## 03:11 - 模型文件下载与查看
在[[Google Colab|谷歌协作平台]]的文件浏览窗格中，你应该能看到我们创建的模型文件。我建议下载所有文件。你可以使用Netron等程序查看[[Keras|凯拉斯]]和[[TensorFlow Lite|Tensor流轻量版]]模型文件，这有助于你了解输入和输出格式以及层如何连接。我们生成的.h文件应包含转换为C数组的[[TensorFlow Lite|Tensor流轻量版]]模型，我将其全局变量命名为sine_model。只需在我们的微控制器代码中包含这个头文件即可导入模型。

## 03:45 - TensorFlow Lite库准备
现在我们有了模型，需要将[[TensorFlow Lite|Tensor流轻量版]]准备为库。我发现[[TensorFlow|张量流]]并不希望你像使用库那样使用它。他们希望你修改自动生成的源文件，然后在他们的文件结构中编译所有内容。这在某些情况下可能有效，但当你已经设置了嵌入式构建系统时，我发现它很难使用。

## 04:09 - 在STM32 Cube IDE中导入TensorFlow Lite
在这个例子中，我将展示如何在STM32 Cube IDE中导入[[TensorFlow Lite|Tensor流轻量版]]。这是因为我最熟悉STM32的相关内容。目前我使用的是STM32L432KC Nucleo板，因为它小巧但运行的是Arm Cortex M4处理器。

## 04:30 - 构建系统设置
接下来的步骤适用于任何构建系统，包括make。你需要在[[TensorFlow|张量流]]中运行一个make文件，这将生成一个模板文件结构。然后你将这些文件复制到你的项目中，并在构建系统中包含必要的头文件和源文件，以生成[[TensorFlow Lite|Tensor流轻量版]]文件结构。你需要使用Linux或Mac OS。我还未在Windows上使这部分工作成功。如果你使用的是Windows，建议双启动或使用Linux的live USB发行版。

## 05:01 - 远程SSH操作
我将SSH连接到我的[[Raspberry Pi|树莓派]]，因为我总是有一个闲置的。你需要安装make、Python 3、pip 3、git和zip。使用你喜欢的包管理器来安装它们。从这里使用Git克隆整个[[TensorFlow|张量流]]库。我将使用深度为1的克隆，这意味着我只抓取最后的修订版，以节省下载时间。接着进入[[TensorFlow|张量流]]的基本目录并运行make命令。然后指向位于[[TensorFlow Lite for Microcontrollers|微控制器Tensor流轻量版]]的micro/tools/make目录中的make文件，并使用便携优化标志。我不确定这是否对这个演示必要，但这个标志似乎生成了几种为微控制器优化的源文件实现，所以我保留它。

## 05:50 - 生成模板项目目录
我们给make命令添加生成非内核项目的目标，这省略了一些我们不需要的模板项目。在这个例子中，make不会构建或编译任何东西，它只是为我们生成一堆模板项目目录，可能需要几分钟时间。

## 06:07 - 创建微控制器项目
在make运行时，我们来创建微控制器项目。我将使用STM32 Cube IDE，但正如我提到的，你可以使用几乎任何东西。只是包含源文件和链接目标文件的步骤可能因开发环境而异，但过程基本相同。

## 06:25 - 配置Nucleo板
我要为我的Nucleo L432KC板创建一个新项目。注意，我们必须将目标语言设置为C++，这意味着选择使用C++编译器，因为大部分[[TensorFlow|张量流]]是用C++编写的。我将启用一个定时器，使其每微秒连续滴答一次，以便我们可以测量模型推理的运行速度。为了使其在我的微控制器上尽可能快地运行，我将使用高速内部振荡器，并将主时钟频率提高到最大80 MHz。

## 06:58 - 生成代码
保存后，让Cube MX生成代码。在这个IDE中，我需要将main.c重命名为main.cpp，以便构建系统将其视为C++文件。

## 07:09 - 复制模型头文件
此时，Linux上的make进程应该已经完成。首先，我将sine_model头文件复制到我的项目目录中。然后，我将SSH连接到映射到Windows机器上的[[Raspberry Pi|树莓派]]文件系统，导航到[[TensorFlow|张量流]]库，然后进入light/micro/tools/make/gen/目录。这个文件夹的名称可能会根据你运行make的具体计算机和操作系统而有所不同。

## 07:40 - 项目模板
在prj目录中，你应该看到一整套项目模板。我建议暂时忽略测试模板，但你可以查看其中的任何一个以了解如何构建[[TensorFlow|张量流]]项目。注意，[[TensorFlow|张量流]]希望你修改这些模板中的主程序以使你的项目工作，但我们不会这样做。

## 07:58 - 导入库文件
进入hello_world，你应该看到为Hello World模板生成的许多构建系统。我们再次忽略这些，因为我要在我选择的构建系统中将[[TensorFlow Lite|Tensor流轻量版]]用作库。进入该目录并复制[[TensorFlow|张量流]]和third_party目录。这些将作为我们的[[Tensor

Flow Lite|Tensor流轻量版]]库。导航回你的微控制器项目目录并创建一个名为TensorFlow_lite的文件夹，将[[TensorFlow|张量流]]和third_party目录粘贴进去。third_party文件夹包含一些[[TensorFlow|张量流]]所需的开源工具，这些工具应该在你运行make时自动下载。在[[TensorFlow|张量流]]文件夹中，进入light/micro/examples/hello_world目录。

## 08:45 - 主应用程序模板
这个文件夹包含我们应该修改的主应用程序。可以随意查看main和main_functions文件，看看它们如何指导你创建[[TensorFlow Lite for Microcontrollers|微控制器Tensor流轻量版]]应用程序。虽然你可以使用这个模板并使其与特定的微控制器一起工作，但我们不会这样做。因此，删除examples文件夹及其中的所有内容。

## 09:06 - 刷新项目资源管理器
回到我们的IDE中，我们要刷新项目资源管理器，使其看到我们新添加的文件。这包括我们添加的[[TensorFlow Lite|Tensor流轻量版]]文件夹以及模型文件。现在我们需要告诉构建系统包含这些头文件和源文件。

## 09:22 - Eclipse项目设置
如果你在使用Eclipse或其衍生版本，请进入项目属性，在C/C++ General中进入Paths and Symbols选项卡，点击Add并从工作区中选择[[TensorFlow Lite|Tensor流轻量版]]文件夹。你需要将其添加到所有语言和配置中。对third_party、flatbuffers、gemmlowp和ruy文件夹重复此过程。由于[[TensorFlow|张量流]]源代码中的包含结构，你必须像这样添加这些third_party目录。确保在C和C++语言以及发布配置中显示包含的目录。进入Source Location选项卡，将[[TensorFlow Lite|Tensor流轻量版]]目录添加到调试和发布配置中。点击Apply，如果被问到是否要重建索引，点击Yes。如果你使用make，确保你的头文件和源文件搜索路径包括这些目录。

## 10:21 - 修改debug_log文件
有一个文件需要在[[TensorFlow Lite|Tensor流轻量版]]库中编辑，以允许我们打印调试信息。进入TensorFlow Lite Micro并打开debug_log.cc，注释掉包含C标准输入输出行以及debug_log函数，重写debug_log函数，但包含weak符号属性并留空函数体。通过使用weak符号，我们可以在主程序中覆盖此函数。需要保留extern "C"，因为这是一个C函数，将在C++程序中调用。回到main函数，实现你认为合适的debug_log函数。

## 11:06 - 打印调试信息
我将简单地通过UART连接打印字符串参数，该连接连接到我的Nucleo板上的USB到串行转换器。由于函数不提供数组长度，我将使用字符串长度函数来计算，并祈祷字符串是以null结尾。需要包含cstring库才能使其工作。为了节省时间，我将只提供此代码的概述，但会在描述中添加链接，以便你可以深入研究。

## 11:35 - 添加包含文件
首先需要添加包含文件，这些文件指向我们程序中需要的[[TensorFlow Lite|Tensor流轻量版]]函数。注意，我还包含了我们在视频开头创建的模型文件。接下来，我们要定义一些全局变量，将它们放在匿名命名空间中，使它们仅在此文件中可访问。这些是指向我们的错误报告器、模型、输入和输出缓冲区的指针。

## 12:01 - 张量竞技场大小
需要注意的重要事项是张量竞技场大小。这里需要进行一些猜测游戏。我将从2KB的竞技场空间开始，这只是[[TensorFlow Lite|Tensor流轻量版]]执行计算所需的一块内存。如果你的allocate_tensors函数在后面失败，可能需要增加此大小。在main函数中，我将添加一些程序需要的变量，例如用于通过UART打印字符串的缓冲区、返回状态代码和微秒级的时间戳。

## 12:21 - 初始化代码
首先启动定时器，以便我们可以测量推理所需的时间。接下来创建一个错误报告器，推理引擎需要它来帮助我们解决问题。我会通过错误报告器写出一条消息，以确保它出现在我们的串行终端中。接下来读取我们的模型数组，注意这里的变量名应与sine_model头文件中的数组名匹配。我们还检查模型中的[[TensorFlow Lite|Tensor流轻量版]]操作模式版本是否与我们使用的[[TensorFlow Lite|Tensor流轻量版]]模式版本匹配。然后创建一个操作解析器，仅包含我们的模型所需的特定操作。如果你想查看支持的[[TensorFlow Lite|Tensor流轻量版]]操作，请查看micro_ops.h文件。由于我们的模型仅使用全连接或密集层，我们只需要向ops注册全连接操作。

## 13:30 - 创建解释器对象
我们创建一个解释器对象，并传入指向我们的模型、操作解析器、竞技场缓冲区和错误报告器的指针。然后调用allocate_tensors函数来配置我们创建的竞技场缓冲区。如果这在你的代码中失败，可能意味着你没有创建足够大的竞技场，需要回去调整张量竞技场大小。最后，分配一些指针给我们的输入和输出缓冲区，以便更容易地访问它们。

## 13:57 - 检查输入张量
此时我们准备开始运行，但我喜欢先检查输入张量的维度。因此，我们将打印出输入张量缓冲区中的元素数量，在这种情况下应该等于一个浮点数。在while循环中，我会将浮点数2.0放入输入张量缓冲区。注意，我在这里使用了for循环来演示如何将多个元素填充到输入缓冲区，但对于这个特定模型，应该只有一个元素。

## 14:29 - 运行推理
我们将获取当前的微秒数，然后调用invoke函数运行推理。这是一个阻塞函数，所以在此时处理器可能会挂起一段时间。完成后，我将获取输出张量缓冲区中的唯一元素。同样，对于其他神经网络，此输出缓冲区可能有多个元素。然后，我将构建一个字符串，显示神经网络的输出以及自原始时间戳以来经过的时间。我会通过UART端口打印这个字符串。

## 14:59 - 打印浮点值
在一些构建系统中，如我的这里，可能需要添加一个特殊的链接器标志来使用%f打印浮点值。因此，我进入项目属性，CC++ Build设置工具设置选项卡，MCU GCC链接器杂项。在其他标志中，我为调试和发布配置添加了-U printf_float标志。最后，在再次进行推理之前添加500毫秒的延迟。

## 15:30 - 构建项目
构建项目时可能会出现一些警告。我建议阅读它们以查看是否有需要修复的内容。对我来说，这个警告似乎是[[TensorFlow Lite|Tensor流轻量版]]库中的某些内容，我可能不会使用，所以暂时忽略它。

## 15:45 - 上传程序
一切构建完毕后，我们可以在STM32 Cube IDE中上传程序。我通过点击Run和Debug来实现这一点。我接受默认的调试配置设置并打开调试视图。运行我的程序并将串行终端连接到我的Nucleo板上。如果一切正常，我应该会看到一些数据，显示神经网络的输出为0.9059，这与本集开始时的测试匹配。我们还可以看到，在我们的80 MHz微控制器上，运行这个简单的三层神经网络需要大约368微秒。这很棒，一切似乎都在工作。我只想再尝试一件事，因为368微秒似乎有点太长了。

## 16:35 - 切换到发布配置
我要进入项目并将我们的活动构建配置设置为发布。然后我将重建项目。

## 16:43 - 发布配置优化
如果你查看调试配置的工具选项，可以看到设置了一个调试标志，而在发布配置中没有这个标志。我的理解是，[[TensorFlow Lite|Tensor流轻量版]]文件中的某些代码会根据这个标志进行更改，从而加快速度。如果我们移除它，这意味着我们将放弃一些有用的调试信息。

## 17:05 - 创建新运行配置
创建一个新的运行配置，并将新构建的发布elf文件设置为我们的应用程序。这是将上传到微控制器的文件。将构建配置设置为发布，以防需要再次构建项目。点击Run，看起来任何新代码都会被构建。

## 17:23 - 使用GNU Size工具检查内存使用
这是一个有趣的时刻，停止并查看GNU Size工具的输出。据我所知，将文本和数据字段相加以获得闪存使用量，大约为50000字节。然后将数据和BSS相加以获得预测的RAM使用量，在这种情况下大约为4700字节。完成后，

程序将上传到STM32，但不会加载调试视图。

## 17:51 - 观察发布版本的运行效果
打开串行终端，你应该会看到程序已经在运行。没有调试标志的情况下，神经网络的运行速度快了三倍以上。现在运行推理只需约104微秒。这对于运行一个神经网络来说似乎相当快，但真正的问题是，我们能否用这么小的神经网络做任何有用的事情？这将是另一个时间讨论的话题。希望大家订阅，如果你想看到更多类似的视频，祝你编程愉快。