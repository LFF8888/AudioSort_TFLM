#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"  // 引入MicroMutableOpResolver库
#include "tensorflow/lite/micro/micro_interpreter.h"         // 引入MicroInterpreter库
#include "tensorflow/lite/micro/system_setup.h"              // 引入系统设置库
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"         // 引入Schema生成的头文件

#include "model.h"       // 引入模型头文件

// 全局变量，用于与Arduino风格的代码兼容
namespace {
const tflite::Model *model = nullptr; // 模型指针
tflite::MicroInterpreter *interpreter = nullptr; // 解释器指针
TfLiteTensor *input = nullptr; // 输入张量指针
TfLiteTensor *output = nullptr; // 输出张量指针
int inference_count = 0; // 推理计数

constexpr int kTensorArenaSize = 2000; // 张量区域大小
uint8_t tensor_arena[kTensorArenaSize]; // 张量区域内存
}  // namespace

// 常量定义
const float kXrange = 2.f * 3.14159265359f;
const int kInferencesPerCycle = 20;

// 函数声明
void InitializeModel(); // 模型初始化函数声明
void RunInference(); // 模型推理函数声明

// 模型初始化函数实现
void InitializeModel() {
  model = tflite::GetModel(g_model); // 获取模型实例
  if (model->version() != TFLITE_SCHEMA_VERSION) { // 检查模型版本是否兼容
    Serial.print("模型版本");
    Serial.print(model->version());
    Serial.print("与所需的版本");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println("不兼容");
    return;
  }

  static tflite::MicroMutableOpResolver<1> resolver; // 操作解析器，指定所需的操作数量
  if (resolver.AddFullyConnected() != kTfLiteOk) { // 添加全连接层操作
    Serial.println("添加操作失败");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr); // 创建解释器
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) { // 为模型分配内存
    Serial.println("内存分配失败");
    return;
  }

  input = interpreter->input(0); // 获取输入张量
  output = interpreter->output(0); // 获取输出张量
}

// 模型推理函数实现
void RunInference() {
  float position = static_cast<float>(inference_count) / static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  int8_t x_quantized = x / input->params.scale + input->params.zero_point; // 输入量化
  input->data.int8[0] = x_quantized;

  if (interpreter->Invoke() != kTfLiteOk) { // 执行模型推理
    Serial.print("在 x: ");
    Serial.print(x);
    Serial.println(" 的推理失败");
    return;
  }

  int8_t y_quantized = output->data.int8[0]; // 读取输出
  float y = (y_quantized - output->params.zero_point) * output->params.scale; // 输出去量化

  Serial.print(x);
  Serial.print(",");
  Serial.println(y);

  inference_count++; // 推理计数加1
  if (inference_count >= kInferencesPerCycle) { // 如果达到循环次数，重置计数器
    inference_count = 0;
  }
}

void setup() {
  Serial.begin(2000000); // 初始化串口，设置波特率
  InitializeModel(); // 初始化模型
}

void loop() {
  RunInference(); // 执行推理
}
