<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# AI 系统与程序代码关系

模型算法的开发者一般会通过使用 AI 框架提供 Python 等高级语言的API，来编写对应的人工智能程序，而人工智能程序的底层系统问题被当前层抽象隐藏。到底在每个代码部分具体底层发生了什么？有哪些有意思的系统设计问题？

本节我们将从一个具体的 PyTorch 实现一个 LeNet5 神经网络模型作为实例开始，启发读者和后面 AI 系统的每一层和各个章节构建起桥梁与联系。

![](images/04Sample01.png)

## 神经网络样例

### AI 训练流程原理

如图所示，可以看到一个神经网络模型可以接受输入（如当前手写数字图片），产生输出（如数字分类），这个过程叫前向传播（Forward Propagation）。

![](images/04Sample02.png)

那么如何得到一个针对当前已有的输入输出数据上，预测效果最好的神经网络模型呢？这个时候需要通过对网络模型进行训练，训练过程可以抽象为数学上的优化问题，优化目标为:

$$\theta = argmin_{\theta}\sum[Loss(f_{\theta}(x), y)]$$

其中：

- $f_{\theta}$ 表示神经网络模型，例如 LeNet；
- $Loss$ 表示损失函数；
- $x$ 表示输入数据，数据中的输入也就是图像；
- $y$ 表示标签值，也代表网络模型的输出；

训练的过程就是找到最小化 $Loss$ 的 $\theta$ 取值，$\theta$ 也称作权重，即网络模型中的参数。在训练过程中将通过梯度下降等数值优化算法进行求解：

$$\theta = \theta - \alpha \delta_{\theta}Loss(\theta)$$

其中， $\alpha$ 也称为学习率(Learning Rate)。当神经网络模型训练完成，就可以通过 $\hat{y} = f_\theta(x)$ 进行推理，使用和部署已经训练好的网络模型。

如图所示，左上角展示输入为手写数字图像，输出为分类向量，中间矩形为各层输出的特征图（Feature Map），我们将其映射为具体的实现代码，其结构通过图右侧定义出来。

可以看到神经网络模型就是通过各个层，将输入图像通过多个层的算子进行计算，得到为类别输出概率向量。

> 算子：深度学习算法由一个个计算单元组成，称这些计算单元为算子（Operator，Op）。AI 框架中对张量计算的种类有很多，比如加法、乘法、矩阵相乘、矩阵转置等，这些计算被称为算子（Operator）。
> 
> 为了更加方便的描述计算图中的算子，现在来对**算子**这一概念进行定义：
>
> **数学上定义的算子**：一个函数空间到函数空间上的映射O：X→X，对任何函数进行某一项操作都可以认为是一个算子。
> 
> - **狭义的算子（Kernel）**：对张量 Tensor 执行的基本操作集合，包括四则运算，数学函数，甚至是对张量元数据的修改，如维度压缩（Squeeze），维度修改（reshape）等。
> 
> - **广义的算子（Function）**：AI 框架中对算子模块的具体实现，涉及到调度模块，Kernel 模块，求导模块以及代码自动生成模块。
>
> 对于神经网络模型而言，算子是网络模型中涉及到的计算函数。在 PyTorch 中，算子对应层中的计算逻辑，例如：卷积层（Convolution Layer）中的卷积算法，是一个算子；全连接层（Fully-connected Layer， FC layer）中的权值求和过程，也是一个算子。

### 网络模型构建

开发者一般经过两个阶段进行构建: 

1. 定义神经网络结构，如图中和代码实例中构建的 LeNet5 网络模型，其中包含有卷积（Conv2D）层，最大池化层（MaxPool2D），全连接（Linear）层。

2. 开始训练，遍历一个批大小（Batch Size）的数据，设置计算的 NPU/GPU 资源数量，执行前向传播计算，计算损失值（Loss），通过反向传播实现优化器计算，从而更新权重。

![](images/04Sample03.png)

现在使用 PyTorch 在 MNIST 数据集上训练一个卷积神经网络 [LeNet](http://yann.lecun.com/exdb/lenet/)[<sup>[1]</sup>](#lenet) 的代码实例。

```
...
import torch
import torch_npu
...

# 如果模型层数多，权重多到无法在单 GPU 显存放置，我们需要通过模型并行方式进行训练
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 通过循环 Loop 实现卷积理解卷积的执行逻辑，可以深入思考其中编译和硬件执行问题。我们将会在第二章、第三章详细展开计算到芯片的关系
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
    	  # 具体的执行 API 单位是算子，实际编译器或者硬件执行的是 Kernel。我们将会在第四章推理引擎Kernel优化详细介绍算子计算执行的方式
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def train(args, model, device, train_loader, optimizer, epoch):
    # 如何进行高效的训练，运行时 Runtime 是如何执行的？我们将在第五章 AI 框架基础进行介绍
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        ...


def test(model, device, test_loader):
    model.eval()
    ... 
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # 推理系统如何高效进行模型推理？我们将在第思章 AI 推理系统进行介绍
            output = model(data)
            ...


def main():
    ...
    # 当前语句决定了使用哪种 AI 加速芯片，可以通过第二章的 AI 芯片基础去了解不同 AI 加速芯片的体系结构及芯片计算的底层原理。
    device = torch.device("npu" if use_cuda else "cpu")
    
    # 如果 batch size 过大，造成单 NPU/GPU HBM 内存无法容纳模型及中间激活的张量，读者可以参考第六章的分布式训练算法，进行了解如何分布式训练
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    ...
    
    # 如果数据量过大，那么可以使用分布式数据并行进行处理，利用集群的资源，可以通过第六章去了解其中的内容
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    model = LeNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    ... 
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # 训练完成需要部署，如何压缩和量化后再部署？可以参考第四章推理系统进行了解
        test(model, device, test_loader)
        ... 

# 如果用户提交多个这样的训练作业，系统如何调度和管理资源？读者可以参考第7章异构计算集群调度与资源管理系统"进行了解
if __name__ == '__main__':
    main()
```

## 算子实现的系统问题

在神经网络中所描述的层（Layer），在 AI 框架中称为算子，或者叫做操作符（Operator）；底层算子的具体实现，在 AI 编译器或者在 AI 芯片时称为 Kernel，对应具体 Kernel 执行的时候会先将其映射或转换为对应的矩阵运算（例如，通用矩阵乘 GEMM），再由其对应的矩阵运算翻译为对应的循环 Loop 指令。

### 卷积实现原理

下图的卷积层实例中，每次选取输入数据一层的一个窗口（和卷积核一样的宽高），然后和对应的卷积核（$5 \times 5$ 卷积核代表高 5 维宽 5 维的矩阵）进行 [矩阵内积（Dot Product）](https://en.wikipedia.org/wiki/Dot_product) 运算，最后将所有的计算结果与偏置项 $b$ 相加后输出。

```
import torch

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        ...
        self.conv2 = nn.Conv2d(3, 2, 5)
        ...
        
    def forward(self, x):
    	  out = self.conv1(x)
    	  ...
```

首先一次沿着行进行滑动一定的步长 Step，再进行下次矩阵内积计算，直到滑到边界后再沿着一定步长跳到下一列重复刚才的滑动窗口。最终把每一步的结果组合成输出矩阵，即产生特征图（Feature Map）。

![](images/04Sample04.png)

图中输入张量形状（Tensor Shape）为 $3 \times 32 \times 32$（3 代表通道数，32 代表张量高度和宽度），经过 $2 \times 3 \times 5 \times 5$ 的卷积（2 代表输出通道数，3 代表输入通道数，5 代表卷积核高度和宽度）后，输出张量形状为 $2 \times 28 \times 28$（2 代表通道，28 代表高度和宽度）。

### 卷积执行样例

示例的卷积计算，最终在程序上表达为多层嵌套循环，为简化计算过程，循环展开中没有呈现维度（Dimension）的形状推导（Shape Inference）。以 Conv2D 转换为如下 7 层循环进行 Kerenl 计算的代码：

```
# 批尺寸维度 batch_size
for n in range(batch_size):
   # 输出张量通道维度 output_channel
   for oc in range(output_channel):
       # 输入张量通道维度 input_channel
       for ic in range(input_channel):
          # 输出张量高度维度 out_height
          for h in range(out_height):
              # 输出张量宽度维度 out_width
              for w in range(out_width):
                  # 卷积核高度维度 filter_height
                  for fh in range(filter_height):
                      # 卷积核宽度维度 filter_width
                      for fw in range(filter_width):
                          # 乘加（Multiply Add）运算
                          output[h, w, oc] += input[h + fw, w + fh, ic]\
                                            * kernel[fw, fh, c, oc]  
```

### AI 系统遇到的问题

在实际 Kernel 的计算过程中有很多有趣的问题：

- 硬件加速： 通用矩阵乘是计算机视觉和自然语言处理模型中的主要的计算方式，同时 NPU/GPU，如 TPU 脉动阵列的矩阵乘单元等其他专用人工智能芯片 ASIC 是否会针对矩阵乘作为底层支持？（第二章 AI 芯片体系结构相关内容）

- 片上内存：其中参与计算的输入、权重和输出张量能否完全放入 NPU/GPU 缓存（L1、L2、Cache）？如果不能放入则需要通过循环块（Loop Tile）编译优化进行切片。（第二章 AI 芯片体系结构相关内容）

- 局部性：循环执行的主要计算语句是否有局部性可以利用？空间局部性（缓存线内相邻的空间是否会被连续访问）以及时间局部性（同一块内存多久后还会被继续访问），这样我们可以通过预估后，尽可能的通过编译调度循环执行。（第三章 AI 编译器相关内容）

- 内存管理与扩展（Scale Out）：AI 系统工程师或者 AI 编译器会提前计算每一层的输出（Output）、输入（Input）和内核（Kernel）张量大小，进而评估需要多少计算资源、内存管理策略设计，以及换入换出策略等。（第三章 AI 编译器相关内容）

- 运行时调度：当算子与算子在运行时按一定调度次序执行，框架如何进行运行时管理？（第四章推理引擎相关内容）

- 算法变换：从算法来说，当前多层循环的执行效率无疑是很低的，是否可以转换为更加易于优化和高效的矩阵计算？（第四章推理引擎相关内容）

- 编程方式：通过哪种编程方式可以让神经网络模型的程序开发更快？如何才能减少或者降低算法工程师的开发难度，让其更加聚焦 AI 算法的创新？（第五章 AI 框架相关内容）

## AI 系统执行具体计算

目前算法工程师或者上层应用开发者只需要使用 AI 框架定义好的 API 使用高级编程语言如 Python 等去编写核心的神经网络模型算法，而不需要关注底层的执行细节和对一个的代码。底层通过层层抽象，提升了开发效率，但是对系统研发却隐藏了众多细节，需要 AI 系统开发的工程师进一步探究。

在上面的知识中，开发者已经学会使用 Python 去编写 AI 程序，以及深度学习代码中的一个算子（如卷积）是如何翻译成底层 for 循环从而进行实际的计算，这类 for 循环计算通常可以被 NPU/GPU 计算芯片厂商提供的运行时算子库进行抽象，不需要开发者不断编写 for 循环执行各种算子操作（如 cuDNN、cuBLAS 等提供卷积、GEMM等 Kernel的实现和对应的API）。

目前已经直接抽象到 Kernel 对具体算子进行执行这一层所提供的高级 API，似乎已经提升了很多开发效率，那么有几个问题：

- 为什么还需要 AI 框架（如 PyTorch、MindSpore 等）？
- AI 框架在 AI System 中扮演什么角色和提供什么内容？
- 用户编写的 Python 代码如何翻译给硬件去执行？

我们继续以上面的例子作为介绍。

![](images/04Sample05.png)

### AI 框架层

如果没有 AI 框架，只将算子 for 循环抽象提供算子库（例如，cuDNN）的调用，算法工程师只能通过 NPU/GPU 厂商提供的底层 API 编写神经网络模型。例如，通过 CUDA + cuDNN 库书写卷积神经网络，如 [cuDNN书写的卷积神经网络LeNet实例](https://github.com/tbennun/cudnn-training)。

1. 通过cuDNN + CUDA API 编程实现 LeNet
 
[参考实例 cudnn-training](https://github.com/tbennun/cudnn-training/blob/master/lenet.cu)，需要~1000行实现模型结构和内存管理等逻辑。

```C++
// 内存分配，如果用深度学习框架此步骤会省略
...
cudaMalloc(&d_data, sizeof(float) * context.m_batchSize * channels * height * width);
cudaMalloc(&d_labels, sizeof(float) * context.m_batchSize * 1  * 1 * 1);
cudaMalloc(&d_conv1, sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height * conv1.out_width);
...

// 前向传播第一个卷积算子（仍需要写其他算子）
...
cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
                        data, conv1filterDesc, pconv1, conv1Desc, 
                        conv1algo, workspace, m_workspaceSize, &beta,
                        conv1Tensor, conv1);
...

// 反向传播第一个卷积算子（仍需要写其他算子），如果用深度学习框架此步骤会省略
cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor,
                             dpool1, &beta, conv1BiasTensor, gconv1bias);

cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor,
                               data, conv1Tensor, dpool1, conv1Desc,
                               conv1bwfalgo, workspace, m_workspaceSize, 
                               &beta, conv1filterDesc, gconv1));

// 第一个卷积权重梯度更新（仍需要写其他算子），如果用深度学习框架此步骤会省略
cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
            &alpha, gconv1, 1, pconv1, 1);
cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
            &alpha, gconv1bias, 1, pconv1bias, 1);

// 内存释放，如果用深度学习框架此步骤会省略
...
cudaFree(d_data);
cudaFree(d_labels);
cudaFree(d_conv1);
...
```

2. 通过 PyTorch 编写 LeNet5

只需要 10 行构建模型结构。

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
```

通过PyTorch + LeNet5 对比 cuDNN + CUDA，明显 cuDNN + CUDA 其抽象还不足以让算法工程师非常高效的设计神经网络模型和算法。同样实现 LeNet5，使用 AI 框架只需要 9 行代码，而通过 cuDNN 需要上千行代码，而且还需要精心的管理内存分配释放，拼接模型计算图，效率十分低下。

因此 AI 框架对算法工程师开发神经网络模型、训练模型等流程非常重要。从而可知 AI 框架一般会提供以下功能：

1. 以 Python API 供读者编写网络模型计算图结构；
2. 提供调用基本算子实现，大幅降低开发代码量；
2. 自动化内存管理、不暴露指针和内存管理给用户；
3. 实现自动微分功能，自动构建反向传播计算图；
4. 调用或生成运行时优化代码，调度算子在指定设备的执行；
6. 并在运行期应用并行算子，提升设备利用率等优化（动态优化）。

AI 框架帮助开发者解决了很多 AI System 底层问题，隐藏了很多工程的实现细节，但是这些细节和底层实现又是 AI System 工程师比较关注的点。

### AI 框架层

（1）前端程序转换为数据流图：如图 1-4-6 所示，这个阶段框架会将用户书写的模型程序，通过预先定义的接口，翻译为中间表达（Intermediate Representation），并且构建算子直接的依赖关系，形成前向数据流图（Data-Flow Graph）。

![](images/04Sample06.png)

（2）反向求导：如图 1-4-7 所示，这个阶段框架会分析形成前向数据流图，通过算子之前定义的反向传播算子，构建反向传播数据流图，并和前向传播数据流图一起形成整体的数据流图。

![](images/04Sample07.png)

（3）产生运行期代码：如图 1.4.8 所示，这个阶段框架会分析整体的数据流图，并根据运行时部署所在的设备（CPU，GPU 等），将算子中间表达产生为算子针对特定设备的运行期的代码，例如图中的 CPU 的 C++ 算子实现或者 GPU 的 CUDA 算子实现。

### AI 编译器与算子库

![](images/04Sample08.png)

（4）调度并运行代码：如图 1.4.9 所示，这个阶段框架会将算子及其运行期的代码实现，依次根据依赖关系，调度到计算设备上进行执行。对一些不方便静态做优化的选择，可以通过运行期调度达到，例如，并发（Concurrent）计算与 I/O，如有空闲资源并行执行没有依赖的算子等。目前框架例如，PyTorch 一般选择单 CUDA Stream 在 NVIDIA GPU 侧进行算子内核调度，数据加载会选择再设置其他 Stream。

以一种让他们合作共享 GPU 的方式编写 CUDA 内核较为困难，因为精确的调度是硬件控制。在实践中，内核编写者通常组合多个任务形成单片内核。数据加载和分布式计算实用程序是单 Stream 设计的例外，它们小心地插入额外的同步以避免与内存分配器的不良交互。

![](images/04Sample09.png)

如果没有 AI 框架、AI编译器、算子库的支持，算法工程师进行简单的神经网络模型设计与开发都会举步维艰，所以应该看到 AI 算法本身飞速发展的同时，也要看到底层系统对提升整个算法研发的生产力起到了不可或缺的作用。

# 本节总结

本章主要通过 PyTorch 的实例启发大家建立 AI 系统各个章节之间的联系，由于系统的多层抽象造成 AI 实践和算法创新的过程中已经无法感知底层系统的运行机制。希望能够结合后面章节的学习后，看到 AI System 底层的作用和复杂性，从而指导上层 AI 作业、算法、代码更加高效的执行和编写。

请读完后面章节后再回看当前章节，并重新思考当前开发使能层下面的每一层的 AI System 底层发生了什么变化？执行了哪些操作和计算？