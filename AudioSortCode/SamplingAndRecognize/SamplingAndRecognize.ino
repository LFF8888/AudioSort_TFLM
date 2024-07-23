#include <arduinoFFT.h>
#include <Arduino.h>
#include <driver/i2s.h>

#include "audio_classification_model.h"       // 引入模型头文件
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"  // 引入MicroMutableOpResolver库
#include "tensorflow/lite/micro/micro_interpreter.h"         // 引入MicroInterpreter库
#include "tensorflow/lite/micro/system_setup.h"              // 引入系统设置库
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"         // 引入Schema生成的头文件

#define SAMPLE_RATE 8000              // 采样率
#define AUDIO_DURATION 1000           // 音频采集持续时间（毫秒）
#define SLIDE_DURATION 250            // 音频滑动步长（毫秒）
#define NUM_MEL_FILTERS 16            // Mel滤波器的数量
#define NUM_MEL_FRAMES 20             // Mel滤波器的数量
#define FRAME_SIZE (SAMPLE_RATE / NUM_MEL_FRAMES)  // 每帧的采样点数
#define HOP_SIZE (SAMPLE_RATE / NUM_MEL_FRAMES)    // 帧移
#define FFT_SIZE 256                  // FFT大小
#define MIN_MEL_FREQ 80.0             // Mel滤波器的最低频率
#define MAX_MEL_FREQ 4000.0           // Mel滤波器的最高频率

const int numSamples = SAMPLE_RATE * AUDIO_DURATION / 1000; // 总采样点数
const int slideSamples = SAMPLE_RATE * SLIDE_DURATION / 1000; // 滑动步长的采样点数

float vReal[FFT_SIZE];
float vImag[FFT_SIZE];
ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, FFT_SIZE, SAMPLE_RATE);

// I2S引脚定义
#define I2S_SCLK_PIN 4  // SCLK (串行时钟)
#define I2S_LRCK_PIN 41 // LRCK (左右通道选择时钟)
#define I2S_SDIN_PIN 5  // SDOUT (数据输出)
const i2s_port_t I2S_PORT = I2S_NUM_0;  // I2S端口编号

// 声明存储梅尔滤波器系数的二维数组
float melFilterCoeffs[NUM_MEL_FILTERS][FFT_SIZE] = {0};

// 声明TFLM相关的全局变量
namespace {
const tflite::Model *model = nullptr; // 模型指针
tflite::MicroInterpreter *interpreter = nullptr; // 解释器指针
TfLiteTensor *input = nullptr; // 输入张量指针
TfLiteTensor *output = nullptr; // 输出张量指针
constexpr int kTensorArenaSize = 16 * 1024; // 张量区域大小
uint8_t tensor_arena[kTensorArenaSize]; // 张量区域内存
}  // namespace

// 分类类名数组
const char* classNames[] = {"discomfort", "nothing", "burp", "xiaoxin", "sleepy", "hunger"};


void setupMicrophone() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX), // 设置I2S为主模式并接收数据
        .sample_rate = SAMPLE_RATE, // 设置采样率为8000 Hz
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT, // 每个采样点为16位
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // 仅使用左声道
        .communication_format = I2S_COMM_FORMAT_STAND_I2S, // 标准I2S通信格式
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, // 中断分配标志
        .dma_buf_count = 8, // DMA缓冲区数量
        .dma_buf_len = 1024, // 每个DMA缓冲区的长度
        .use_apll = false // 不使用APLL
    };
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCLK_PIN, // 设置SCLK引脚
        .ws_io_num = I2S_LRCK_PIN, // 设置LRCK引脚
        .data_out_num = I2S_PIN_NO_CHANGE, // 不使用数据输出引脚
        .data_in_num = I2S_SDIN_PIN // 设置数据输入引脚
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL); // 安装I2S驱动
    i2s_set_pin(I2S_PORT, &pin_config); // 设置I2S引脚
    i2s_start(I2S_PORT); // 启动I2S
}

void computeMelFilterCoeffs() {
    for (int mel = 0; mel < NUM_MEL_FILTERS; mel++) {
        float melStartFreq = melToHz(mel * (hzToMel(MAX_MEL_FREQ) - hzToMel(MIN_MEL_FREQ)) / (NUM_MEL_FILTERS - 1));
        float melCenterFreq = melToHz((mel + 1) * (hzToMel(MAX_MEL_FREQ) - hzToMel(MIN_MEL_FREQ)) / (NUM_MEL_FILTERS - 1));
        float melEndFreq = melToHz((mel + 2) * (hzToMel(MAX_MEL_FREQ) - hzToMel(MIN_MEL_FREQ)) / (NUM_MEL_FILTERS - 1));

        int startIdx = freqToIndex(melStartFreq);
        int centerIdx = freqToIndex(melCenterFreq);
        int endIdx = freqToIndex(melEndFreq);

        for (int i = startIdx; i < centerIdx; i++) {
            melFilterCoeffs[mel][i] = (i - startIdx) / (float)(centerIdx - startIdx);
        }
        for (int i = centerIdx; i < endIdx; i++) {
            melFilterCoeffs[mel][i] = 1 - (i - centerIdx) / (float)(endIdx - centerIdx);
        }
    }
}

void computeMelSpectrogram(float *audioData, float melSpectrogram[NUM_MEL_FILTERS][NUM_MEL_FRAMES]) {
    for (int frame = 0; frame < NUM_MEL_FRAMES; frame++) {
        int startIdx = frame * HOP_SIZE;
        for (int i = 0; i < FFT_SIZE; i++) {
            if (startIdx + i < numSamples) {
                vReal[i] = audioData[startIdx + i];
            } else {
                vReal[i] = 0;
            }
            vImag[i] = 0;
        }

        FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);  // 应用汉明窗
        FFT.compute(FFT_FORWARD);                        // 计算FFT
        FFT.complexToMagnitude();                        // 计算幅值

        float melEnergies[NUM_MEL_FILTERS] = {0};
        applyMelFilterBank(vReal, melEnergies);

        for (int i = 0; i < NUM_MEL_FILTERS; i++) {
            melSpectrogram[i][frame] = log(melEnergies[i] + 1e-9); // 对数刻度
        }
    }
}

void applyMelFilterBank(float *spectrum, float *melEnergies) {
    for (int mel = 0; mel < NUM_MEL_FILTERS; mel++) {
        melEnergies[mel] = 0;
        for (int i = 0; i < FFT_SIZE; i++) {
            melEnergies[mel] += spectrum[i] * melFilterCoeffs[mel][i];
        }
    }
}

// 频率转换辅助函数
float hzToMel(float hz) {
    return 2595.0 * log10(1.0 + hz / 700.0);
}

float melToHz(float mel) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

int freqToIndex(float freq) {
    return round((FFT_SIZE + 1) * freq / SAMPLE_RATE);
}

// 非阻塞读取I2S数据
bool readNonBlockingI2S(int16_t* buffer, size_t buffer_size, size_t* bytesRead) {
    size_t bytesReadNow; // 本次读取的字节数

    if (*bytesRead < buffer_size) {
        i2s_read(I2S_PORT, (void*)(buffer + *bytesRead / sizeof(int16_t)), buffer_size - *bytesRead, &bytesReadNow, 0);
        *bytesRead += bytesReadNow;
        return false; // 还没有读取完
    } else {
        *bytesRead = 0; // 重置已读取字节数
        return true; // 读取完毕
    }
}


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
void process_audio_and_recognize() {
    static int16_t *buffer = (int16_t *)malloc(numSamples * sizeof(int16_t)); // 动态分配缓冲区用于存储采样数据
    static size_t totalBytesRead = 0; // 已读取的总字节数
    static int16_t *slideBuffer = (int16_t *)malloc(numSamples * sizeof(int16_t)); // 用于滑动窗口的缓冲区
    static size_t slideBytesRead = 0; // 滑动窗口已读取的字节数

    // 每次读取SLIDE_DURATION时长的音频数据
    if (!readNonBlockingI2S(buffer + slideBytesRead / sizeof(int16_t), slideSamples * sizeof(int16_t), &slideBytesRead)) {
        return; // 如果还没有读取完，直接返回
    }

    // 将新的数据添加到滑动窗口缓冲区
    memmove(slideBuffer, slideBuffer + slideSamples, (numSamples - slideSamples) * sizeof(int16_t));
    memcpy(slideBuffer + (numSamples - slideSamples), buffer, slideSamples * sizeof(int16_t));

    // 将缓冲区数据转换为浮点型向量
    float *signal = (float *)malloc(numSamples * sizeof(float));
    for (int i = 0; i < numSamples; ++i) {
        signal[i] = slideBuffer[i] / 32768.0f; // 转换为浮点型
    }

    // 计算Mel频谱图
    float melSpectrogram[NUM_MEL_FILTERS][NUM_MEL_FRAMES] = {0};
    computeMelSpectrogram(signal, melSpectrogram);
    free(signal); // 释放信号缓冲区

    // 将Mel频谱图数据加载到模型输入张量
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
        for (int j = 0; j < NUM_MEL_FRAMES; j++) {
            input->data.f[i * NUM_MEL_FRAMES + j] = melSpectrogram[i][j];
        }
    }

    // 执行推理
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("推理执行失败");
        return;
    }

  // 解析输出结果
  for (int i = 0; i < output->dims->data[1]; i++) {
      float output_prob = output->data.f[i];
      Serial.print("  ");
      Serial.print(classNames[i]);
      Serial.print(": ");
      Serial.print(output_prob, 2);
  }
  Serial.println();
}

void setup() {
    Serial.begin(2000000); // 初始化串口通信，波特率为2000000
    setupMicrophone(); // 初始化麦克风
    computeMelFilterCoeffs(); // 计算并存储梅尔滤波器系数
    InitializeModel(); // 初始化TFLM模型
}

void loop() {
    process_audio_and_recognize(); // 处理音频并进行识别
}