📌 原文链接：[Intro to TinyML Part 1: Training a Model for Arduino in TensorFlow (digikey.com)](https://www.digikey.com/en/maker/projects/intro-to-tinyml-part-1-training-a-model-for-arduino-in-tensorflow/8f1fc8c0b83d417ab521c48864d2a8ec)

视频教程：[(629) Intro to TinyML Part 1: Training a Neural Network for Arduino in TensorFlow | Digi-Key Electronics - YouTube](https://www.youtube.com/watch?v=BzzqYNYOcWc)

 2020-04-06 | 作者：[ShawnHymel](https://www.digikey.com/en/maker/profiles/72825bdd887a427eaf8d960b6505adac)

当我们大多数人想到[[AI|人工智能]]（[[AI]]，[[AI|Artificial Intelligence]]）和[[ML|机器学习]]（[[ML]]，[[ML|Machine Learning]]）时，我们通常会想到家用助手、自动机器人和自动驾驶汽车。然而，[[ML|机器学习]]的世界远不止这些。每当我们训练数学模型来帮助计算机在没有明确指示的情况下进行预测、决策或分类时，我们就在使用[[ML|机器学习]]。

在大多数情况下，有用的[[ML|机器学习]]算法需要大量的计算资源（[[CPU|中央处理器]]（[[CPU]]，[[CPU|Central Processing Unit]]）周期和[[RAM|随机存取存储器]]（[[RAM]]，[[RAM|Random Access Memory]]））。然而，[[TensorFlow Lite]]最近发布了一个实验版本，可以在多个微控制器上运行。假设我们可以创建一个适合资源有限设备的模型，我们可以开始将嵌入式系统转变为微型[[ML|机器学习]]（[[TinyML]]）设备。

在本教程中，我们将创建一个能够预测正弦函数输出的神经网络。然后我们将把这个模型转换为[[TensorFlow Lite]]（[[TFLite]]）模型，并使用[[Netron]]进行检查。

如果您想观看这些步骤的视频说明，请查看这个YouTube视频：

**模型描述**

我们将创建一个三层全连接神经网络，该网络可以预测正弦函数的输出。因此，我们将其用作[[回归模型]]（[[Regression Model]]）。 

这个想法是训练一个模型，该模型接受0到2π之间的值，然后输出-1到1之间的值。如果我们将输入值标记为$x$，输出值标记为$y$，那么模型应该能够预测$y = \sin(x)$。

这可能是计算正弦波最无效率、最迂回的方法之一。然而，这让我们可以玩一个带有一些非线性的微小神经网络，并将其加载到微控制器上。

[[TensorFlow]]包含一个转换器类，该类允许我们将[[Keras]]模型转换为[[TensorFlow Lite]]模型。[[TensorFlow Lite]]模型存储为[[FlatBuffer]]，这对于一次读取大块数据非常有用（而不是必须将所有数据加载到[[RAM|随机存取存储器]]中）。

一旦我们创建并训练了模型，我们将运行[[TensorFlow Lite]]转换器以创建一个tflite模型。然后，我们需要将tflite模型存储为C常量数组中的一系列字节（在.c或.h文件中）。这将允许我们加载模型并使用[[TensorFlow Lite for Microcontrollers]]库进行推理。

[[TensorFlow Lite]]支持一些微控制器板，您可以在[此处](https://www.tensorflow.org/lite/microcontrollers)查看支持的列表。在本教程发布时，仅支持8个微控制器板。虽然有一个适用于[[TensorFlow Lite]]的[[Arduino]]库，但仅支持Nano 33 BLE Sense。因此，在本教程系列的第二部分中，我们将使用Nano 33 BLE Sense。

**[[Google Colab]]**

[[Google Colab]]是一个基于网页的[[Jupyter Notebook]]界面，运行在[[Linux]]虚拟机（[[VM|虚拟机]]，[[VM|Virtual Machine]]）上。它预装了大多数流行的[[ML|机器学习]]包，并且是免费的！

注册一个[[Google GMail]]账户并前往[https://colab.research.google.com/](https://colab.research.google.com/)。

虽然您可以在Colab中试验各种[[ML|机器学习]]算法，但您在[[VM|虚拟机]]资源和时间上受到限制。如果您在90分钟内不与界面交互，您将被断开连接，并且每12小时，运行时将重置。

如果您希望进行更长时间的模型训练或需要更多资源，您将需要支付专业版费用或设置自己的[[ML|机器学习]]开发机器（本地或服务器）。

请注意，以下大多数代码也可以在本地的Python或[[Jupyter Notebook]]上运行。如果您希望本地运行[[TensorFlow]]，请参见[本教程](https://www.tensorflow.org/install)。

**模型训练**

请注意，原始代码基于[[TensorFlow Lite]]的[[Pete Warden]]的工作。我对示例进行了一些调整，使其在视频中更好地工作。

前往Colab并点击“New Notebook”。为该笔记本命名一个独特的名称，如“tflite-sinewave-training.ipynb”。

代码最好在Notebook形式中查看，这样可以看到示例输出。我将在本教程中讨论一些重要的代码部分，但如果您有兴趣自己运行Notebook并剖析代码，请参见[此Gist](https://gist.github.com/ShawnHymel/79237fe6aee5a3653c497d879f746c0c)。

在顶部，我们选择我们希望使用的[[TensorFlow]]版本。虽然我最初指定的是“2.1”，但此魔术命令仅处理主要版本。因此，建议使用：

```python
%tensorflow_version 2.x
```

之后，我们导入必要的包并打印它们的版本，这在各种论坛上寻求帮助时非常有用，因为人们通常会要求查看您使用的版本。

然后，我们指定各种设置，包括我们希望使用的样本数量以及我们希望为验证和测试集保留的样本百分比。

从那里，我们生成用于训练模型的一组随机$x$值：

```python
# Generate some random samples
np.random.seed(1234)
x_values = np.random.uniform(low=0, high=(2 * math.pi), size=nsamples)
plt.plot(x_values)
```

然后，我们计算每个样本的$\sin(x)$值，并在输出中添加一些随机的高斯噪声。这有助于确保模型是一个估计，而不是正弦波的精确表示。

```python
# Create a noisy sinewave with these values
y_values = np.sin(x_values) + (0.1 * np.random.randn(x_values.shape[0]))
plt.plot(x_values, y_values, '.')
```

接下来，我们将数据分成训练、验证和测试集。通常，查看（或绘制）输入数据和标签是个好主意。如果可能的话，因为输入和输出这里只有一个值（与多维数组相对），我们可以更容易地可视化输入和输出之间的关系：

```python
# Split the dataset into training, validation, and test sets
val_split = int(val_ratio * nsamples)
test_split = int(val_split + (test_ratio * nsamples))
x_val, x_test, x_train = np.split(x_values, [val_split, test_split])
y_val, y_test, y_train = np.split(y_values, [val_split, test_split])
# Check that our splits add up correctly
assert(x_train.size + x_val.size + x_test.size) == nsamples
# Plot the data in each partition in different colors:
plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_test, y_test, 'r.', label="Test")
plt.plot(x_val, y_val, 'y.', label="Validate")
plt.legend()
plt.show()
```

这应该会给我们一个如下所示的图：

![数据分割图](https://example.com/sine_wave_data_plot.png)

现在我们有了准备好的数据，是时候创建模型了。我们将使用一个三层全连接（密集）神经网络。

```python
# Create a model
model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(1,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))
```

然后，我们可以用设置的优化器和损失函数编译它，接着是实际的训练过程：

```python
# Add optimizer, loss function, and metrics to model and compile it
model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])

# Train model
history = model.fit(x_train,
                    y_train,
                    epochs=500,
                    batch_size=100,
                    validation_data=(x_val, y_val))
```

几分钟后，您应该有一个完全训练的模型。我们可以绘制模型预测的正弦函数与测试数据的对比，以确保我们有一个近似的正弦波：

```python
# Plot the training history
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1

, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
```

此代码的输出应该给我们一个如下所示的图：

![训练历史图](https://example.com/training_history_plot.png)

正如您所见，我们的模型应该给我们一个近似的正弦波，尽管它不完全正确。

**转换为[[TensorFlow Lite]]模型**

现在我们有了一个训练好的[[Keras]]模型，我们需要将其转换为我们的微控制器可以使用的东西。[[TensorFlow]]有一个内置的转换器函数，它将模型保存为[[TensorFlow Lite]]模型文件（存储为[[FlatBuffer]]）：

```python
# Convert Keras model to a tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
open(tflite_model_name + '.tflite', 'wb').write(tflite_model)
```

有一些高级技术，如量化，您可以在[此处](https://www.tensorflow.org/lite/microcontrollers/build_convert)阅读相关内容。请注意，许多这些技术每天都在变化，可能需要大量工作才能在微控制器上正常运行。

尽管如此，基本转换似乎现在可以工作，尽管它不提供量化所提供的相同级别的内存节省。

要在微控制器上运行tflite文件，我们需要将其保存为.c或.h文件中的常量字节数组。

注意：为了良好的C代码，您应该在.h文件中声明字节数组，并在.c文件中定义（实际字节）。因为我想保持这个示例简单，我将把所有内容保存在一个.h文件中。

此函数可用于读取tflite模型文件，将其转换为十六进制字节，并生成包含我们模型和任何必要C代码的.h文件字符串：

```python
# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):
    c_str = ''
    # Create header guard
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'
    # Add array length at top of file
    c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'
    # Declare C variable
    c_str += 'unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data):
        # Construct string from hex
        hex_str = format(val, '#04x')
        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)
    # Add closing brace
    c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'
    # Close out header guard
    c_str += '#endif //' + var_name.upper() + '_H'
    return c_str
```

我们将调用此函数并将.h文件保存到我们的Colab虚拟机：

```python
# Write TFLite model to a C source (or header) file
with open(c_model_name + '.h', 'w') as file:
    file.write(hex_to_c_array(tflite_model, c_model_name))
```

**保存和检查模型文件**

点击Colab窗口左侧面板中的“Files”按钮，以检查虚拟机中的文件。您应该看到sine_model.h。如果没有，点击“Refresh”按钮。右键点击sine_model.h并选择“Download”。

您可以检查.h文件，以查看它是否包含合法的C代码（数组长度和大量十六进制值的数组）。

您还可以下载sine_model.tflite模型文件。如果这样做，我建议安装[[Netron]]以可视化模型。只需用Netron打开.tflite文件，您应该会看到模型描述图。如果点击某一层，您可以获得该层的信息，如输入和输出类型以及张量大小。

**进一步学习**

希望这有助于您开始使用Colab创建一个简单的[[TensorFlow Lite]]模型，并将其部署到微控制器上！在下一教程中，我们将在[[Arduino]]上运行[[TensorFlow Lite]]推理引擎，并使用我们的模型来预测正弦函数值。

请参阅以下文章以了解更多关于[[TensorFlow Lite for Microcontrollers]]的内容：

- [https://www.tensorflow.org/lite/microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [https://www.tensorflow.org/lite/microcontrollers/get_started](https://www.tensorflow.org/lite/microcontrollers/get_started)
- [https://www.digikey.com/en/maker/projects/tensorflow-lite-for-microcontrollers-kit-quickstart/1b372b69e44f4d988b5363741a61882d](https://www.digikey.com/en/maker/projects/tensorflow-lite-for-microcontrollers-kit-quickstart/1b372b69e44f4d988b5363741a61882d)

**推荐阅读**

- [TinyML简介第2部分：将[[TensorFlow Lite]]模型部署到[[Arduino]]](https://www.digikey.com/en/maker/projects/tensorflow-lite-for-microcontrollers-kit-quickstart/1b372b69e44f4d988b5363741a61882d)

有问题或意见吗？请继续在[[TechForum]]，[[DigiKey]]的在线社区和技术资源上交流。

[访问[[TechForum]]](https://forum.digikey.com/)



# 视频教程提取

# TinyML介绍 第1部分 - 使用TensorFlow为Arduino训练神经网络 | Digi-Key Electronics

2024年07月14日 17:54

## 00:10 - TinyML简介
Tiny Machine Learning，即TinyML，是AI领域新兴的开发方向，旨在将各种机器学习算法加载到微控制器上。这项技术不仅新颖，而且非常酷。由于它还很新，许多库每天都在变化。因此，今天展示的内容可能在明天就无法使用了。

### 为什么要关注TinyML
在之前的视频中，我制作了一个通过唤醒词关闭铣床的设备。当我说“停止”时，这个设备需要运行在[[Raspberry Pi|树莓派]]上。如果我们能在微控制器上运行这个模型，它将需要更少的功率，并且我们可以更容易地将其嵌入到铣床中，而不是使用完整的唤醒词模型。

## 00:55 - 示例模型介绍
为了让你了解如何训练模型并将其部署到微控制器上，我将展示一个简单的示例。我们将创建一个模型，当输入一个在0到2π之间的值时，输出一个在-1到1之间的值，近似一个正弦波。这是创建正弦波的最荒谬的方法之一，但它可以让我们创建一个足够小的神经网络，能够在微控制器上运行。

我们将使用[[TensorFlow|张量流]]（[[TensorFlow]]，[[TensorFlow|TensorFlow]]）来训练模型。这次我会在[[Google Colab|谷歌协作平台]]上展示，所以你不需要安装Python、Anaconda、TensorFlow、NumPy等其他包。如果不想，你可以跳过这些安装步骤。注意，我们在做的是一种回归，因为我们已经知道输出应该是一个正弦波，因此不需要收集样本集进行训练。

## 01:49 - 模型转换与部署
一旦我们有了模型，我们将使用TensorFlow内置的转换器将其转换为[[TensorFlow Lite|Tensor流轻量版]]（[[TFLite]]，[[TensorFlow Lite|TensorFlow Lite]]）模型。虽然一些嵌入式系统可以直接读取该模型文件，但我们会将其转换为保存为C头文件的原始字节。然后在我们的Arduino草图中包含该头文件，并运行[[TensorFlow Lite|Tensor流轻量版]]推理引擎，以近似正弦波。

## 02:11 - 使用Google Colab
如果你已经在计算机上安装了Python、TensorFlow和[[Jupyter Notebook|Jupyter笔记本]]，欢迎使用它们。然而，在接下来的几集里，我将展示如何使用[[Google Colab|谷歌协作平台]]，这是一款带有[[Jupyter Notebook|Jupyter笔记本]]的在线编辑器，并且可以免费使用。注意，你需要一个Gmail账户才能使用它。前往colab.research.google.com并登录你的Gmail账户。

## 02:40 - Google Colab的使用限制
Google Colab每90分钟断开一次连接，如果你没有与它互动的话。你还可以在每个会话中获得长达12小时的GPU支持。这些限制允许你玩弄机器学习，但如果需要训练更大的模型，你可能需要支付Pro版或自行设置机器学习环境。

## 03:07 - TensorFlow版本选择与环境设置
在Colab中，我们可以使用魔术函数`tensorflow_version`来指定所需的TensorFlow版本。我将使用2.1版，虽然解释器只允许选择主要版本1或2。然后我们导入[[TensorFlow|张量流]]、[[NumPy|数值Python]]（[[NumPy]]，[[NumPy|NumPy]]）、[[Matplotlib|绘图库]]（[[Matplotlib]]，[[Matplotlib|Matplotlib]]）、math和来自[[Keras|凯拉斯]]（[[Keras]]，[[Keras|Keras]]）的layers。因为TensorFlow不断变化，我喜欢打印出我所使用的Python和各种包的版本号，这在需要在论坛上寻求帮助时非常有用。

## 04:00 - 数据集生成
我们将生成1000个样本，并将20%的样本用于验证，20%的样本用于测试。最后，我们定义我们的TF Lite模型和模型名称，它们将用于文件命名。

我们定义一个随机种子，以便在0到2π之间均匀分布生成一些X值，并绘制这些X值以确保它们符合标准。最后，我们使用这些X值计算正弦值。为了增加趣味性，我们会在Y值上添加一些高斯噪声，这将使模型的预测更加不精确，因为模型需要基于这个嘈杂的信号估计值。

## 04:29 - 数据集划分
然后我们将数据集划分为训练集、验证集和测试集。训练集用于更新模型参数。验证集在训练过程中用于评估模型在未见过的数据上的表现。测试集将在最后用于测试。

## 04:47 - 数据集检查
确保训练集、验证集和测试集的总和等于数据集中的样本数，不要遗漏任何数据或在集合之间有任何重叠。然后我们可以一起绘制这些集合。大多数点应该属于训练集，验证集和测试集各占20%的点。每个集合应该沿整个正弦波分布，而不仅仅集中在一个区域。

## 05:12 - 模型架构
我们将使用两个16节点的全连接层或密集层，并添加一个仅有一个节点的最终层。输入是一个数字，在许多机器学习应用中，这将是一个包含许多数字的数组，但在这里我们只给出一个在0到2π之间的数字。模型将执行必要的计算，以预测一个与输入值的正弦对应的值，希望输出介于-1到1之间。

## 05:46 - 计算量
如你所见，每一步我们都需要进行大量的计算。这个模型相对较小，但对于微控制器来说，在短时间内处理所有这些乘法仍然很困难，但对于我们的演示来说，应该足够了，可以创建一个100Hz的正弦波。

## 06:03 - 使用Keras创建模型
在Colab中，我们使用Keras创建我们刚刚描述的模型。我们告诉模型输入张量的形状仅为一个元素，并且除最后一个节点外，所有节点的激活函数都为[[ReLU|修正线性单元]]（[[ReLU]]，[[Rectified Linear Unit]]）。

我们将优化器设置为RMSprop，损失函数设置为绝对误差。你可以使用其他优化器，如Adam，以及其他损失函数，如均方误差。这些对小模型的训练影响不大。

## 06:31 - 模型训练
然后我们使用fit函数进行训练，需提供存储在X_train中的训练输入数据和存储在Y_train中的训练输入标签。我们将训练500个epoch，批量大小为100，并在每个epoch后测试验证集。训练只需几分钟。训练完成后，绘制训练历史图表无论是损失还是准确度值都是个好主意。这里我们可以看到训练和验证损失都收敛到约0.1以下，这意味着它应该可以准确预测我们的正弦波。验证曲线与训练曲线对齐，意味着我们不太可能出现模型过拟合的情况。

## 07:12 - 测试模型
最后，我们要看看模型在未见过的数据上的表现。这是我们拿出之前保留的测试集的时候。我们将基于测试集中的输入值创建一组预测。我们使用蓝色点绘制正弦函数的实际输出（加上一些噪声），使用红色点绘制预测值。正如你所见，模型创建了一个近似的正弦波，但似乎没有完全跟随曲线。没关系，对于我们的演示来说已经足够了。

## 07:40 - 转换为TensorFlow Lite模型
现在我们有了一个表现足够好的模型，我们想将其转换为[[TensorFlow Lite|Tensor流轻量版]]模型。我们将使用内置的TF Lite转换器来完成此操作。由于大多数有用的神经网络在推理时需要大量内存，因此优化模型大小是个好主意。我们调用convert函数，然后将结果模型保存为.TF Lite文件。注意，Colab在其自己的虚拟Linux机器中运行。如果我们点击文件夹图标，可以看到虚拟机中的目录列表。默认工作目录为content目录，因此如果进入content目录，我们可以看到刚刚创建的TF Lite文件，右键点击并下载。

## 08:21 - 模型检查工具
一个很棒的模型检查工具是Netron，你可以从其GitHub Readme页面下载安装程序。安装后，只需运行程序并打开模型文件。我选择我们刚刚创建的TF Lite文件。你可以点击各个层来获取它们的信息。例如，你可以看到输入需要一个仅包含一个元素的浮点数组。两个16节点层使用浮点数，模型输出另一个仅包含一个元素的浮

点数组。

## 08:48 - 转换为C代码
一旦我们确认模型看起来不错，我们需要将其转换为C代码。在Linux中，有一个叫xxd的工具可以读取TF Lite文件（只是一个字节数组），并将其转换为另一个文件中的C数组。虽然这在Colab中可行，但如果你在Windows上本地运行，xxd不会帮助你。因此，我将展示如何使用hex2Carray函数将一组原始字节转换为C数组。我们只需在Python中传递一个原始字节数组以及变量名，该函数将创建一个我们可以写入文件的字符串。实际上，我们让Python为我们编写C代码。

## 09:23 - 创建C头文件
我们通常应该在一个.c文件中保持这个常量数组的定义，然后创建一个单独的.h文件进行声明。但我很懒，这在Arduino中需要更多工作。因此，我们只创建一个.h文件，并将这个常量数组放在里面。我们将.h文件保存在Colab虚拟机上。如果点击刷新，该文件应该会出现。右键点击文件并下载，就像我们之前对TF Lite文件做的那样。如果在文本编辑器中打开该文件，它应该看起来像一个标准的C头文件，顶部定义了模型长度，然后是一个巨大的字节数组。我们将在下一集将这个文件复制到我们的嵌入式项目中。

## 10:00 - 运行环境要求
虽然有一些机器学习算法可以在8位微控制器上运行，但运行TensorFlow Lite的最低要求似乎是Arm Cortex M3或M4。我发现如果RAM少于100KB，几乎无法做任何有用的事情。此外，TensorFlow Lite仅支持少数几款微控制器。

## 10:20 - Arduino演示
在视频制作时，下一集我们将使用Arduino，因为这是设置TensorFlow Lite工具链的最简单方法。目前，仅支持[[Arduino Nano 33 BLE Sense]]，因为它基于NRF 5240芯片。如果你想尝试代码，可以在描述中找到链接。下次，我们将模型加载到Arduino板上，并使用TensorFlow Lite推理引擎实时预测正弦值。如果你想看到更多类似的视频，请订阅并祝你编程愉快。