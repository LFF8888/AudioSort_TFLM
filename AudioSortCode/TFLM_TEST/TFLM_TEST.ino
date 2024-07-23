#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"  // 引入MicroMutableOpResolver库
#include "tensorflow/lite/micro/micro_interpreter.h"         // 引入MicroInterpreter库
#include "tensorflow/lite/micro/system_setup.h"              // 引入系统设置库
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"         // 引入Schema生成的头文件

#include "audio_classification_model.h"       // 引入模型头文件

// 全局变量，用于与Arduino风格的代码兼容
namespace {
const tflite::Model *model = nullptr; // 模型指针
tflite::MicroInterpreter *interpreter = nullptr; // 解释器指针
TfLiteTensor *input = nullptr; // 输入张量指针
TfLiteTensor *output = nullptr; // 输出张量指针

constexpr int kTensorArenaSize = 16 * 1024; // 张量区域大小，根据需要调整
uint8_t tensor_arena[kTensorArenaSize]; // 张量区域内存
}  // namespace

// 声明梅尔频谱图数据数组
constexpr int NUM_MEL_FILTERS = 16;
constexpr int NUM_MEL_FRAMES = 20;
float melSpectrogram[NUM_MEL_FILTERS][NUM_MEL_FRAMES] = {
    {-3.44, -3.68, -3.47, -3.43, -2.19, -0.72, -2.68, -2.40, -3.43, -2.19, -0.38, -4.91, -3.90, -3.43, -2.19, -3.99, -3.57, -3.60, -3.44, -2.19},
    {-3.71, -3.71, -3.52, -3.66, -3.10, -1.55, -3.03, -3.53, -3.66, -3.10, -1.43, -4.06, -4.29, -3.66, -3.10, -4.55, -4.00, -4.12, -3.66, -3.10},
    {-4.36, -4.23, -4.31, -3.42, -3.48, -1.21, -3.07, -4.00, -3.42, -3.48, -1.64, -4.47, -4.40, -3.42, -3.48, -4.38, -4.29, -4.76, -3.41, -3.48},
    {-4.49, -4.01, -4.22, -3.75, -3.47, -1.32, -3.51, -3.62, -3.75, -3.47, -1.95, -5.17, -5.46, -3.75, -3.47, -4.15, -4.66, -5.49, -3.75, -3.47},
    {-3.83, -2.66, -3.26, -3.78, -2.16, -1.02, -2.12, -3.26, -3.78, -2.16, -2.08, -4.53, -3.89, -3.78, -2.16, -3.53, -2.16, -3.92, -3.78, -2.16},
    {-4.73, -1.58, -4.24, -4.13, -2.07, -0.88, -4.10, -3.82, -4.13, -2.07, -1.92, -4.88, -4.43, -4.13, -2.07, -2.86, -1.52, -3.62, -4.23, -2.07},
    {-2.57, -1.90, -2.65, -1.72, -1.61, -0.89, -2.20, -1.79, -1.72, -1.61, -1.86, -4.45, -3.89, -1.72, -1.61, -2.58, -1.21, -3.24, -1.70, -1.61},
    {-1.44, -1.82, -0.95, -1.05, -1.55, -0.72, -1.09, -1.56, -1.05, -1.55, -1.10, -1.75, -1.65, -1.05, -1.55, -1.16, -2.30, -3.66, -1.05, -1.55},
    {-3.01, -2.47, -3.29, -1.25, -2.33, -0.59, -3.39, -2.33, -1.25, -2.33, -2.05, -3.14, -2.82, -1.25, -2.33, -1.91, -1.75, -3.07, -1.25, -2.33},
    {-0.68, -1.39, -1.90, -0.84, -1.72, -0.53, -1.67, -1.63, -0.84, -1.72, -1.02, -1.12, -1.06, -0.84, -1.72, -1.03, -1.75, -3.28, -0.84, -1.72},
    {-2.98, -2.33, -4.03, -3.33, -2.40, -0.37, -3.49, -3.60, -3.33, -2.40, -1.31, -1.53, -1.95, -3.33, -2.40, -1.32, -2.95, -4.24, -3.24, -2.40},
    {-1.20, -1.61, -2.38, -1.51, -1.92, -0.24, -2.13, -2.22, -1.51, -1.92, -0.84, -0.91, -1.11, -1.51, -1.92, -0.86, -2.14, -3.54, -1.51, -1.92},
    {-1.66, -1.91, -2.03, -1.03, -1.99, -0.10, -2.06, -1.90, -1.03, -1.99, -1.74, -2.49, -2.11, -1.03, -1.99, -1.69, -1.82, -3.21, -1.03, -1.99},
    {-2.31, -2.03, -1.98, -1.75, -1.78, 0.01, -1.94, -2.04, -1.75, -1.78, -1.69, -2.90, -2.74, -1.75, -1.78, -2.05, -1.68, -3.57, -1.75, -1.78},
    {-4.51, -2.61, -4.27, -4.10, -3.03, 0.10, -3.35, -4.19, -4.10, -3.03, -2.34, -5.19, -4.86, -4.10, -3.03, -4.06, -2.92, -4.67, -4.06, -3.03},
    {-4.69, -3.00, -4.68, -4.15, -4.00, -4.00, -4.15, -4.00, -4.15, -4.00, -4.15, -4.00, -4.15, -4.00, -4.15, -4.00, -4.15, -4.00, -4.15, -4.00}
};

// 函数声明
void InitializeModel();
void RunInference();

// 模型初始化函数实现
void InitializeModel() {
    model = tflite::GetModel(audio_classification_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("模型版本与所需的版本不兼容");
        return;
    }

    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("内存分配失败");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
}

// 模型推理函数实现
void RunInference() {
    // 从melSpectrogram数组加载数据到输入张量
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
        for (int j = 0; j < NUM_MEL_FRAMES; j++) {
            input->data.f[i * NUM_MEL_FRAMES + j] = melSpectrogram[i][j];
        }
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("推理执行失败");
        return;
    }

    // 解析输出结果
    for (int i = 0; i < output->dims->data[1]; i++) {
        float output_prob = output->data.f[i];
        Serial.print("  Class");
        Serial.print(i);
        Serial.print(": ");
        Serial.print(output_prob, 2);
    }
    Serial.println();
}

void setup() {
    Serial.begin(115200);
    InitializeModel();
}

void loop() {
    unsigned long startTime = millis(); // 记录开始时间
    RunInference(); // 进行推理
    unsigned long endTime = millis(); // 记录结束时间

    unsigned long inferenceTime = endTime - startTime; // 计算推理时间

    Serial.print("Inference Time: ");
    Serial.print(inferenceTime);
    Serial.println(" ms");

    delay(1000); 
}
