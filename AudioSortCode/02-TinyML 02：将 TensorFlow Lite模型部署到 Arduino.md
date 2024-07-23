📌 原文链接：[Intro to TinyML Part 2: Deploying a TensorFlow Lite Model to Arduino (digikey.com)](https://www.digikey.com/en/maker/projects/intro-to-tinyml-part-2-deploying-a-tensorflow-lite-model-to-arduino/59bf2d67256f4b40900a3fa670c14330)

视频教程：[Intro to TinyML Part 2: Deploying a TensorFlow Lite Model to Arduino | Digi-Key Electronics - YouTube](https://www.youtube.com/watch?v=dU01M61RW8s)

2020-04-20 | 作者：Shawn Hymel
**许可证**: AttributionArduino

在前一篇教程中，我们训练了一个 [[TensorFlow Lite|TensorFlow Lite]]（[[TFLite]]，[[TensorFlow Lite]]）模型，以在输入0到2π之间的值时预测正弦函数的值。然后我们使用组成 [[TensorFlow Lite|TensorFlow Lite]] 模型文件的常量字节创建了一个 .h 头文件，该文件可以加载到 C 程序中。

在本教程中，我们将使用 [[TensorFlow Lite|TensorFlow Lite]] 库在 Arduino 中加载模型，并使用它运行推理来生成正弦波的近似值。

请注意，这是一种绕道而行的创建正弦波的方式，但它提供了一个有用的示例，展示了如何将非线性神经网络模型部署到嵌入式系统中。

您可以通过以下视频观看本教程的步骤：

### 模型描述

我们的模型是一个3层全连接神经网络，具有单一的浮点输入和单一的浮点输出。

![](http://www.kdocs.cn/api/v3/office/copy/VUNOb0lsRGVBeHZRRWpMZSs2U01YSThHUmZMeXpoZkxKT1M2WVRlR2dUNmRickRHVGs0NUYyOWxYS2FBTXRuSjhQNjV3NllDcVJKZkVjM2lGbUpMYk0zS0l4bnNSdUEyNFhQUXpmaU5DcVpOaWJVUnhIdUM0Mm9HS3ovUjUxUG52NDNjU3JuNWZUcXZZMy90R1kzY3hGMVJMNmk4UXJyN28yeHQyZERYUTZiUUVBcVY0bDBwRzNNVCtuV09RWW5yMFFWSDhIQkF2b0VrOVppRnR1SHJhSVlGT0c5S05IbjhzUCsyWHo0ZE1Yd0E2RCtJS1NSczlMcm91RU9rbElzRXNWT053RmhnelY4PQ==/attach/object/2272f91815c32654d7b03927eae774307e4dedb5?)

如果您在前一个教程中下载了 .tflite 文件，可以使用 [[Netron]] 查看模型的图形界面。运行 [[Netron]] 并使用它打开 .tflite 文件。您可以点击各个层以获取更多关于它们的详细信息，例如输入/输出张量形状和数据类型。

![](http://www.kdocs.cn/api/v3/office/copy/VUNOb0lsRGVBeHZRRWpMZSs2U01YSThHUmZMeXpoZkxKT1M2WVRlR2dUNmRickRHVGs0NUYyOWxYS2FBTXRuSjhQNjV3NllDcVJKZkVjM2lGbUpMYk0zS0l4bnNSdUEyNFhQUXpmaU5DcVpOaWJVUnhIdUM0Mm9HS3ovUjUxUG52NDNjU3JuNWZUcXZZMy90R1kzY3hGMVJMNmk4UXJyN28yeHQyZERYUTZiUUVBcVY0bDBwRzNNVCtuV09RWW5yMFFWSDhIQkF2b0VrOVppRnR1SHJhSVlGT0c5S05IbjhzUCsyWHo0ZE1Yd0E2RCtJS1NSczlMcm91RU9rbElzRXNWT053RmhnelY4PQ==/attach/object/82d4108e06d276f9f6c24b5cce560125e5ef8024?)

正如您所见，我们的模型期望输入是一个具有1个浮点元素（一个标量值）的张量，并且输出另一个标量值。输入应在0到2π之间，输出应在-1到1之间。

### 安装 [[TensorFlow Lite|TensorFlow Lite]] Arduino 库

[[TensorFlow Lite|TensorFlow Lite]] 支持一些微控制器板，这些板列在此处。在本教程发布时，只有8个微控制器板受到支持。我们将使用预编译的 [[TensorFlow Lite|TensorFlow Lite]] 库，但请注意，目前仅支持 Nano 33 BLE Sense。

打开您的 Arduino IDE（本教程在 v1.8.11 上测试）。转到Sketch > Include Library > Manage Libraries…，然后搜索“TensorFlow”。安装最新版本的 [[Arduino_TensorFlowLite|Arduino_TensorFlowLite]] 库（本教程测试了 1.15.0-ALPHA 版本）。

![](http://www.kdocs.cn/api/v3/office/copy/VUNOb0lsRGVBeHZRRWpMZSs2U01YSThHUmZMeXpoZkxKT1M2WVRlR2dUNmRickRHVGs0NUYyOWxYS2FBTXRuSjhQNjV3NllDcVJKZkVjM2lGbUpMYk0zS0l4bnNSdUEyNFhQUXpmaU5DcVpOaWJVUnhIdUM0Mm9HS3ovUjUxUG52NDNjU3JuNWZUcXZZMy90R1kzY3hGMVJMNmk4UXJyN28yeHQyZERYUTZiUUVBcVY0bDBwRzNNVCtuV09RWW5yMFFWSDhIQkF2b0VrOVppRnR1SHJhSVlGT0c5S05IbjhzUCsyWHo0ZE1Yd0E2RCtJS1NSczlMcm91RU9rbElzRXNWT053RmhnelY4PQ==/attach/object/f561c1cb24ff21010f1a1d1bb69b9d3552bf57a4?)

在Tools > Board中，选择Arduino Nano 33 BLE。插入您的 Nano 33 BLE Sense 并在Tools > Port中选择相关的串口。

### 测试推理

注意：此代码最初由 [[TensorFlow Lite|TensorFlow Lite]] 团队的 Pete Warden 开发，以演示 [[TensorFlow Lite|TensorFlow Lite]] 在各种微控制器平台上的功能。我在这里对其进行了修改，以减少 Arduino 中的依赖文件数量，希望这样更易于理解。

将以下代码复制到您的 Arduino 草图中：

```cpp
/**
 * 测试正弦波神经网络模型
 * 
 * 作者：Pete Warden
 * 修改者：Shawn Hymel
 * 日期：2020年3月11日
 * 
 * 版权所有 2019 The TensorFlow Authors。保留所有权利。
 *
 * 根据 Apache 许可证，版本 2.0（“许可证”）获得许可；
 * 除非符合许可证，否则不得使用此文件。
 * 您可以在以下地址获取许可证副本：
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * 除非适用法律要求或书面同意，否则根据许可证分发的软件
 * 按“原样”分发，不附带任何明示或暗示的保证或条件。
 * 请参阅许可证以了解管理权限和限制的特定语言。
 */
// 导入 TensorFlow 相关内容
#include "TensorFlowLite.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/version.h"
// 我们的模型
#include "sine_model.h"
// 了解模型的运行情况
#define DEBUG 1
// 一些设置
constexpr int led_pin = 2;
constexpr float pi = 3.14159265;                  // 一些 pi
constexpr float freq = 0.5;                       // 正弦波频率（Hz）
constexpr float period = (1 / freq) * (1000000);  // 周期（微秒）
// TFLite 全局变量，用于与 Arduino 风格的草图兼容
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
  // 创建一个内存区域，用于输入、输出和其他 TensorFlow 数组
  // 需要通过编译、运行并查找错误来调整它
  constexpr int kTensorArenaSize = 5 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace
void setup() {
  // 等待串行连接
#if DEBUG
  while(!Serial);
#endif
  // 使 LED 亮度变化
  pinMode(led_pin, OUTPUT);
  // 设置日志记录（即使在 TFLite 函数内也会报告给串行）
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  // 将模型映射到可用的数据结构中
  model = tflite::GetModel(sine_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("模型版本与架构不匹配");
    while(1);
  }
  // 仅引入所需的操作（应匹配神经网络层）
  // 可用的操作:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_FULLY_CONNECTED,
    tflite::ops::micro::Register_FULLY_CONNECTED(),
    1, 3);
  // 构建一个解释器以运行模型
  static tflite::MicroInterpreter static_interpreter(
    model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;
  // 为模型的张量分配内存
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() 失败");
    while(1);
  }
  // 将模型输入和输出缓冲区（张量）分配给指针
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  // 获取用于模型输入的内存区域的信息
  // 支持的数据类型:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h#L226
#if DEBUG
  Serial.print("维数数量: ");
  Serial.println(model_input->dims->size);
  Serial.print("维 1 大小: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("维 2 大小: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("输入类型: ");
  Serial.println(model_input->type);
#endif
}
void loop() {
#if DEBUG
  unsigned long start_timestamp = micros();
#endif
  // 获取当前时间戳并取模周期
  unsigned long timestamp = micros();
  timestamp = timestamp % (unsigned long)period;
  // 计算要输入模型的 x 值
  float x_val = ((float)timestamp * 2 * pi) / period;
  // 将值复制到输入缓冲区（张量）
  model_input->data.f[0] = x_val;
  // 运行推理
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke 失败，输入: %f\n", x_val);
  }
  // 从输出缓冲区（张量）读取预测的 y 值
  float y_val = model_output->data.f[0];
  // 转换为 PWM LED 亮度
  int brightness = (int)(255 * y_val);
  analogWrite(led_pin, brightness);
  // 打印值
  Serial.println(y_val);
#if DEBUG
  Serial.print("推理时间（微秒）: ");
  Serial.println(micros() - start_timestamp);


#endif
}
```

对于第一次测试，我们只需将 pi（近似为 3.14159265）的值分配给输入张量：

```cpp
model_input->data.f[0] = pi;
```

这样，我们将对一个数字（在循环函数中反复进行）运行推理。

当您运行草图时，打开串行监视器以查看输出值。还请注意，由于我们将 DEBUG 标志定义为 1，我们可以估算推理所需的时间。

![](http://www.kdocs.cn/api/v3/office/copy/VUNOb0lsRGVBeHZRRWpMZSs2U01YSThHUmZMeXpoZkxKT1M2WVRlR2dUNmRickRHVGs0NUYyOWxYS2FBTXRuSjhQNjV3NllDcVJKZkVjM2lGbUpMYk0zS0l4bnNSdUEyNFhQUXpmaU5DcVpOaWJVUnhIdUM0Mm9HS3ovUjUxUG52NDNjU3JuNWZUcXZZMy90R1kzY3hGMVJMNmk4UXJyN28yeHQyZERYUTZiUUVBcVY0bDBwRzNNVCtuV09RWW5yMFFWSDhIQkF2b0VrOVppRnR1SHJhSVlGT0c5S05IbjhzUCsyWHo0ZE1Yd0E2RCtJS1NSczlMcm91RU9rbElzRXNWT053RmhnelY4PQ==/attach/object/b65e38bb7e0baa2ced6bc8a191537a01c782bf32?)

注意，模型的输出总是 0.03。虽然 sin(π) 应该是 0，但 0.03 似乎已经足够接近了。此外，注意推理大约需要 1 毫秒。这个数字基于微控制器的性能能力以及模型的大小/复杂性。这个模型相对简单，因此当您开始使用更复杂的模型时，预计推理时间会更长。

### 代码分析

让我们快速看看代码中发生了什么。

在顶部，我们定义了一些将在草图的其余部分中使用的指针。请注意，它们在匿名命名空间中，尽管这可能不是必须的，但遵循了其他 [[TensorFlow Lite|TensorFlow Lite]] 示例的做法。

```cpp
// TFLite 全局变量，用于与 Arduino 风格的草图兼容
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
  // 创建一个内存区域，用于输入、输出和其他 TensorFlow 数组
  // 需要通过编译、运行并查找错误来调整它
  constexpr int kTensorArenaSize = 5 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace
```

注意，我们为一个“arena”保留了一块内存（本质上是 [[TensorFlow Lite|TensorFlow Lite]] 用于执行计算和存储张量的 RAM 沙箱）。不幸的是，我们必须预测 arena 的大小。对于这个模型，5 kB 似乎是合适的，但如果在稍后的“分配张量”步骤中遇到问题，您应该尝试增加 arena 的大小。

我们设置了日志记录，对于 Arduino，将调试信息输出到串行端口。请注意，这将输出一些 [[TensorFlow Lite|TensorFlow Lite]] 函数内的信息，当某些东西不起作用时，这会非常有帮助。

```cpp
// 设置日志记录（即使在 TFLite 函数内也会报告给串行）
static tflite::MicroErrorReporter micro_error_reporter;
error_reporter = &micro_error_reporter;
```

为了减少所需空间，我们只引入所需的 [[TensorFlow Lite|TensorFlow Lite]] 操作，代码如下：

```cpp
// 仅引入所需的操作（应匹配神经网络层）
static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
micro_mutable_op_resolver.AddBuiltin(
  tflite::BuiltinOperator_FULLY_CONNECTED,
  tflite::ops::micro::Register_FULLY_CONNECTED(),
  1, 3);
```

您需要挑选与构建模型时定义的层（和其他操作）一致的操作。查看 [[Netron]] 以了解所需操作可能会有所帮助。

在接下来的部分中，我们使用刚刚定义的参数构建解释器，然后使用 arena 分配必要的内存。然后，我们获取输入和输出张量的句柄。请注意，这些张量只有一个元素（标量）。

```cpp
// 构建一个解释器以运行模型
static tflite::MicroInterpreter static_interpreter(
  model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
  error_reporter);
interpreter = &static_interpreter;
// 为模型的张量分配内存
TfLiteStatus allocate_status = interpreter->AllocateTensors();
if (allocate_status != kTfLiteOk) {
  error_reporter->Report("AllocateTensors() 失败");
  while(1);
}
// 将模型输入和输出缓冲区（张量）分配给指针
model_input = interpreter->input(0);
model_output = interpreter->output(0);
```

如果在运行时 AllocateTensors() 函数失败，您可能需要尝试增加先前定义的 arena 大小。内存管理似乎需要一些反复试验。

在 loop 中，我们将输入值复制到输入张量中。注意索引为 0；如果这是一个多元素张量，我们希望使用 for 循环将所有内容复制进去。

```cpp
// 将值复制到输入缓冲区（张量）
model_input->data.f[0] = pi;
```

我们告诉解释器使用 Invoke() 函数运行推理：

```cpp
// 运行推理
TfLiteStatus invoke_status = interpreter->Invoke();
if (invoke_status != kTfLiteOk) {
  error_reporter->Report("Invoke 失败，输入: %f\n", x_val);
}
```

注意此时 Invoke 是阻塞的，因此我们必须等待它执行其计算。

完成后，我们可以在 model_output 句柄中访问输出值：

```cpp
float y_val = model_output->data.f[0];
```

希望这能让您了解在微控制器上运行推理所需的 [[TensorFlow|TensorFlow]] 操作！

### 将推理连接到硬件

将一个 LED 从引脚2连接到 GND（通过一个100Ω电阻）。

在程序顶部，将 DEBUG 标志更改为以下内容：

```cpp
#define DEBUG 0
```

然后，将 loop() 函数更改为以下内容：

```cpp
void loop() {
#if DEBUG
  unsigned long start_timestamp = micros();
#endif
// 获取当前时间戳并取模周期
  unsigned long timestamp = micros();
  timestamp = timestamp % (unsigned long)period;
  // 计算要输入模型的 x 值
  float x_val = ((float)timestamp * 2 * pi) / period;
  // 将值复制到输入缓冲区（张量）
  model_input->data.f[0] = x_val;
  // 运行推理
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke 失败，输入: %f\n", x_val);
  }
  // 从输出缓冲区（张量）读取预测的 y 值
  float y_val = model_output->data.f[0];
  // 转换为 PWM LED 亮度
  int brightness = (int)(255 * y_val);
  analogWrite(led_pin, brightness);
  // 打印值
  Serial.println(y_val);
#if DEBUG
  Serial.print("推理时间（微秒）: ");
  Serial.println(micros() - start_timestamp);
#endif
}
```

注意，我们现在根据当前时间戳计算输入到正弦预测模型的值。通过这样做，我们可以有效地使用程序顶部的 freq 变量来控制正弦波的频率。

上传此代码并打开串行监视器。您应该会看到模型输出的值不断飞过。

![](http://www.kdocs.cn/api/v3/office/copy/VUNOb0lsRGVBeHZRRWpMZSs2U01YSThHUmZMeXpoZkxKT1M2WVRlR2dUNmRickRHVGs0NUYyOWxYS2FBTXRuSjhQNjV3NllDcVJKZkVjM2lGbUpMYk0zS0l4bnNSdUEyNFhQUXpmaU5DcVpOaWJVUnhIdUM0Mm9HS3ovUjUxUG52NDNjU3JuNWZUcXZZMy90R1kzY3hGMVJMNmk4UXJyN28yeHQyZERYUTZiUUVBcVY0bDBwRzNNVCtuV09RWW5yMFFWSDhIQkF2b0VrOVppRnR1SHJhSVlGT0c5S05IbjhzUCsyWHo0ZE1Yd0E2RCtJS1NSczlMcm91RU9rbElzRXNWT053RmhnelY4PQ==/attach/object/bebb6f94793d1066d3b9b09e7e9f21534e15813c?)

这不是特别有用，因此关闭它并选择Tools > Serial Plotter。这应该会显示一个随时间变化的缓慢移动的正弦波。

![](http://www.kdocs.cn/api/v3/office/copy/VUNOb0lsRGVBeHZRRWpMZSs2U01YSThHUmZMeXpoZkxKT1M2WVRlR2dUNmRickRHVGs0NUYyOWxYS2FBTXRuSjhQNjV3NllDcVJKZkVjM2lGbUpMYk0zS0l4bnNSdUEyNFhQUXpmaU5DcVpOaWJVUnhIdUM0Mm9HS3ovUjUxUG52NDNjU3JuNWZUcXZZMy90R1kzY3hGMVJMNmk4UXJyN28yeHQyZERYUTZiUUVBcVY0bDBwRzNNVCtuV09RWW5yMFFWSDhIQkF2b0VrOVppRnR1SHJhSVlGT0c5S05IbjhzUCsyWHo0ZE1Yd0E2RCtJS1NSczlMcm91RU9rbElzRXNWT053RmhnelY4PQ==/attach/object/55a177fd5d62a9a34af64f3cdda5d2dba77fe039?)

如果您查看连接到 Arduino 板的 LED，您应该会看到它的亮度在增加和减少，近似一个正弦模式。

![](http://www.kdocs.cn/api/v3/office/copy/VUNOb0lsRGVBeHZRRWpMZSs2U01YSThHUmZMeXpoZkxKT1M2WVRlR2dUNmRickRHVGs0NUYyOWxYS2FBTXRuSjhQNjV3NllDcVJKZkVjM2lGbUpMYk0zS0l4bnNSdUEyNFhQUXpmaU5DcVpOaWJVUnhIdUM0Mm9HS3ovUjUxUG52NDNjU3JuNWZUcXZZMy90R1kzY3hGMVJMNmk4UXJyN28yeHQyZERYUTZiUUVBcVY0bDBwRzNNVCtuV09RWW5yMFFWSDhIQkF2b0VrOVppRnR1SHJhSVlGT0c5S05IbjhzUCsyWHo0ZE1Yd0E2RCtJS1NSczlMcm91RU9rbElzRXNWT053RmhnelY4PQ==/attach/object/45c8724691d0286b3353ce5183484bb7adb6584a?)

尝试将频率更改为更高的值，例如：

```cpp
constexpr float freq = 100;
```

上传此代码并查看串行绘图仪。您应该会看到一个更好地表示我们正弦波的图。

![](http://www.kdocs.cn/api/v3/office/copy/VUNOb0lsRGVBeHZRRWpMZSs2U01YSThHUmZMeXpoZkxKT1M2WVRlR2dUNmRickRHVGs0NUYyOWxYS2FBTXRuSjhQNjV3NllDcVJKZkVjM2lGbUpMYk0zS0l4bnNSdUEyNFhQUXpmaU5DcVpOaWJVUnhIdUM0Mm9HS3ovUjUxUG52NDNjU3JuNWZUcXZZMy90R1kzY3hGMVJMNmk4UXJyN28yeHQyZERYUTZiUUVBcVY0bDBwRzNNVCtuV09RWW5yMFFWSDhIQkF2b0VrOVppRnR1SHJhSVlGT0c5S05IbjhzUCsyWHo0ZE1Yd0E2RCtJS1NSczlMcm91RU9rbElzRXNWT053RmhnelY4PQ==/attach/object/3f9a553bc0f2f086d7f7d10901b016c16e817d34?)

### 更进一步

再次强调，这不是一个完美的正弦波，当然也不是计算正弦的高效方式，但它确实是测试 [[TensorFlow Lite|TensorFlow Lite]] 在 Arduino 中功能的有用方式。希望这能帮助您入门 [[TensorFlow Lite|TensorFlow Lite]] 的微控制器！

要了解更多关于 [[TensorFlow Lite|TensorFlow Lite]] 的微控制器信息，请参阅以下文章：
- https://www.tensorflow.org/lite/microcontrollers 
- https://www.tensorflow.org/lite/microcontrollers/get_started 
- https://www.digikey.com/en/maker/projects/tensorflow-lite-for-microcontrollers-kit-quickstart/1b372b69e44f4d988b5363741a61882d 

### 推荐阅读
- TinyML 入门第一部分：在 [[TensorFlow|TensorFlow]] 中为 Arduino 训练模型

有问题或意见吗？请在 DigiKey 的在线社区和技术资源 TechForum 上继续讨论。
访问 TechForum