## 5.4. 谷歌TPU历史发展

### 5.4.1 为什么需要TPU

早在2006年，谷歌的内部就讨论过在自家的数据中心中部署图形处理器（GPU）、现场可编程门阵列（FPGA）或自研专用集成电路（ASIC）的可能性。但当时能够在特殊硬件上运行的少数应用程序可以几乎0代价的利用当前谷歌大型数据中心的过剩算力完成，那什么比免费的午餐更有吸引力呢？于是此项目并没有落地。但是在2013年，风向突变，当时谷歌的研究人员做出预测：如果人们每天使用语音搜索并通过深度神经网络（DNN）进行3分钟的语音识别，那么当时谷歌的数据中心需要双倍的算力才能满足日益增长的计算需求，而仅仅依靠传统CPU来满足这种需求是非常昂贵的。[<sup>[1]</sup>](#ref1)于是，在这个背景下，谷歌开始了TPU的设计。

通常一个芯片的开发需要几年的时间，然而谷歌不愧是谷歌，TPU从立项到大规模部署只用了15个月。TPU项目的领头人Norm Jouppi说到：“我们的芯片设计过程异常迅速，这本身就是一项非凡的成就。令人惊叹的是，我们首批交付的硅片无需进行任何错误修正或掩模的更改。考虑到在整个芯片构建过程中，我们还在同步进行团队的组建，紧接着迅速招募RTL（寄存器传输级）设计专家，并且急切地补充设计验证团队，整个工作节奏非常紧张。”[<sup>[2]</sup>](#ref2)

### 5.4.2 历代TPU芯片与产品

#### 5.4.2.1历代TPU芯片
以下表格是不同TPU芯片型号的具体参数和规格，我们的TPU系列会主要围绕v1, v2, v3, v4这一系统去展开。

| TPU 比较              | TPUv1       | TPUv2       | TPUv3       | Edge TPU v1 | Pixel Neural Core | TPUv4i      | TPUv4       | Google Tensor |
|------------------------|-------------|-------------|-------------|-------------|-------------------|-------------|-------------|---------------|
| 推出日期              | 2016年     | 2017年     | 2018年     | 2018年     | 2019年           | 2020年     | 2021年     | 2021年       |
| 制程技术              | 28nm        | 16nm        | 16nm        | -           | -                 | 7nm         | 7nm         | -             |
| 芯片大小 (mm²)        | 330         | 625         | 700         | -           | -                 | 400         | 780         | -             |
| 芯片内存 (MB)         | 28          | 32          | 32          | -           | -                 | 144         | 288         | -             |
| 时钟速度 (MHz)        | 700         | 700         | 940         | -           | -                 | 1050        | 1050        | -             |
| 内存                  | 8 GiB DDR3   | 16 GiB HBM   | 32 GiB HBM  | -           | -                 | 8GiB DDR     | 32 GiB HBM   | -             |
| 内存带宽 (GB/s)       | 300         | 700         | 900         | -           | -                 | 300         | 1200        | -             |
| 热设计功耗 (W)        | 75          | 280         | 450         | -           | -                 | 175         | 300         | -             |
| TOPS (Tera/Second)    | 92           | 45         | 123         | 4           | -                 | -        | 275         | -             |
| TOPS/W                | 0.31         | 0.16        | 0.56        | 2           | -                 | -        | 1.62        | -             |




#### 5.4.2.2历代TPU产品
在前文中，我们讨论了CPU的不同型号，现在让我们将注意力转向谷歌的TPU产品线。以下表格中除了芯片之外，随着芯片技术的不断进步，谷歌推出了TPU Pod，这是一种由众多TPU单元构成的超大规模计算系统，专为处理大量深度学习和AI领域的并行计算任务而设计。除了具有超强的算力之外，TPU Pod装备了高速的互联网络，保证了TPU设备之间无缝的数据传输以保证强大的数据、模型层的高效拓展性。

|名称       | 时间        | 性能                              | 应用                |
|--------------|-----------|------------|----------------------------------|
|               TPUv1     | 2016年    | 92Tops + 8GB DDR3                | 数据中心推理       |
|               TPUv2     | 2017年    | 180TFlops(浮点计算能力) + 64GB(HBM) | 数据中心训练和推理 |
|               TPUv3     | 2018年    | 420TFlops + 128GB(HBM)           | 数据中心训练和推理 |
|               Edge TPU  | 2018年    | 可处理高吞吐量的稀疏数据        | IoT 设备           |
|               TPUv2 Pod | 2019年    | 11.5万亿次运算/秒，4TB ( HBM )   | 数据中心训练和推理 |
|              TPUv3 Pod | 2019年    | >100万亿次运算/秒，32TB ( HBM )  | 数据中心训练和推理 |
|               TPUv4     | 2021年    | -                                | 数据中心训练和推理 |
|               TPUv4 Pod | 2022年    | -                                | 数据中心训练和推理 |


随着时间的推移，谷歌不仅在大型数据中心部署了先进技术，还洞察到将这些技术应用于消费电子产品，尤其是智能手机市场的巨大潜力。于是在2017年，谷歌在Pixel 2 和 Pixel 3上便搭载了自家的Pixel Visual Core，成为了谷歌针对消费类产品的首个定制图像芯片。之后，谷歌基于Edge TPU的框架研发了Pixel Visual Core的继任，Pixel Neural Core，在2019年10月发布的Pixel 4上首次搭载。之后，谷歌在 Pixel 产品线上对于TPU的依赖也一直延续到了今天。

在这个AI爆发的大时代，谷歌在移动端的AI也掷下豪赌，对于最新发布的Tensor G3，Google Silicon 的高级总监 Monika Gupta说到：“我们的合作与Tensor一直不仅仅局限于追求速度和性能这样的传统评价标准。我们的目标是推动移动计算体验的进步。在最新的Tensor G3芯片中，我们对每个关键的系统组件都进行了升级，以便更好地支持设备上的生成式人工智能技术。这包括最新型号的ARM中央处理器、性能更强的图形处理器、全新的图像信号处理器和图像数字信号处理器，以及我们最新研发的，专门为运行Google的人工智能模型而量身打造的TPU。”[<sup>[3]</sup>](#ref3)

[等待补图]

### 5.4.3 TPU的演进

#### 5.4.3.1 TPU v1概览
<!-- --：确定性模型（Deterministic Execution Model） -->

第一代TPU主要服务于8比特的矩阵计算，由CPU通过PCIe 3.0总线驱动CISC指令。它采用28nm工艺制造，频率为700MHz，热设计功耗为40瓦。它具有28MiB的芯片内存和4MiB 32位累加器，用于存储256x256系统阵列的8位乘法器的结果。TPU封装内还有8GiB双通道2133MHz DDR3 SDRAM,带宽为34GB/s。指令能够将数据传输至/离开主机、执行矩阵乘法或卷积运算，并应用激活函数。

受限于时代，初代TPU主要针对2015年左右最火的深度学习网络进行优化，主要分为以下三类：

- MLP 多层感知机（**M**ulti**L**ayer **P**erceptron）
- CNN 卷积神经网络（**C**onvolutional **N**eural **N**etwork）
- RNN 递归神经网络（**R**ecurrent **N**eural **N**etwork）& LSTM 长短期记忆（**L**ong **S**hort-**T**erm **M**emory）
  
而在这三类中，由于RNN和LSTM的高复杂度，初代TPU很难去处理这类神经网络。
#### 5.4.3.2 TPU v1的优化
为了强化TPU的矩阵计算性能，谷歌的工程师针对其进行了若干特殊设计和优化，以提高处理深度学习计算工作负载的效率。以下是谷歌为了加强TPU在矩阵计算方面性能所做的三种主要努力和特殊设计：

**特性一：低精度**
神经网络在进行推理时，并不总是需要32位浮点数（FP32）或16位浮点数（FP16）以上的计算精度。TPU通过引入一种称为**量化**的技术，可以将神经网络模型的权重和激活值从FP32或FP16转换为8位整数（Int8），从而实现模型的压缩。这种转换使得8位整数能够近似表示在预设最小值和最大值之间的任意数值，优化了模型的存储和计算效率。

在下图从FP32量化到Int8的过程中，虽然单个数据点无法维持FP32的超高精确度，整体数据分布却能保持大致准确。尽管连续数值被压缩到较小的离散范围可能引起一定精度损失，但得益于神经网络的泛化能力，在推理场景，特别是分类任务中，量化后的神经网络模型能够在保持接近原始FP32/FP16精度水平的同时，实现更快的推理速度和更低的资源消耗。

![FP32INT8](https://storage.googleapis.com/gweb-cloudblog-publish/images/tpu-148g2u.max-1500x1500.png)

**特性二：脉动阵列 & MXU**

在TPU中有一个关键组件叫做 MXU（Matrix Multiply Unit，矩阵乘法单元）。与传统的CPU和GPU架构相比，MXU专为高效处理大规模的Int8矩阵加乘法运算而设计了独特的**脉动阵列**（Systolic Array）架构。

CPU旨在执行各种计算任务，因此具备通用性。CPU通过在寄存器中存储数据，并通过程序指令控制算术逻辑单元（ALU）读取哪些寄存器、执行何种操作（如加法、乘法或逻辑运算）以及将结果存储到哪个寄存器。程序由一系列的读取/操作/写入指令构成。这些支持通用性的特性（包括寄存器、ALU以及程序控制）在功耗和芯片面积上付出了较高的代价。但是对于MXU来说，它只需要用ALU大批量处理矩阵的加乘运算，而在矩阵的加乘运算生成输出时会多次复用输入数据。因此在某些情况下，TPU只需要读取每个输入值一次，就可以在不存储回寄存器的情况下将数据复用于许多不同的操作。[<sup>[2]</sup>](#ref2)

下面这个图中，左边的图例描述了CPU中的程序逻辑，数据在经过ALU计算前后都会经由寄存器处理，而右图描述了TPU内部数据在ALU之间更快地流动且复用的过程。[<sup>[2]</sup>](#ref2)



![脉动阵列](https://storage.googleapis.com/gweb-cloudblog-publish/images/tpu-17u39j.max-500x500.PNG)

下图的两个动图是脉动阵列的原理图示，我们可以看到，输入的数据在和权重矩阵相乘的流动中十分有节奏感，就像是心脏泵血一样，这就是为什么脉动阵列要这样命名（注：Systolic一词专指“心脏收缩的”）


用脉动阵列做输入向量和权重矩阵的矩阵乘法           |  用脉动阵列做输入矩阵和权重矩阵的矩阵乘法    
:-------------------------:|:-------------------------:
![](https://storage.googleapis.com/gweb-cloudblog-publish/original_images/Systolic_Array_for_Neural_Network_1pkw3.GIF)  |  ![](https://storage.googleapis.com/gweb-cloudblog-publish/original_images/Systolic_Array_for_Neural_Network_2g8b7.GIF)


MXU的本质就是一个包含了$256 \times 256 = 65536$个ALU的超大的，每一个时钟周期可以处理65536个INT8加乘运算的脉动阵列。将这个数字和TPU v1的频率700MHZ相乘我们可以得出TPU v1可以每秒钟处理$65536 \times 7 \times 10^8 \approx 4.6 \times 10 ^{12} $个加乘运算。下图中我们可以看到，数据和权重从控制器传入MXU，脉冲阵列中经过计算再产出最终的结果。

![](https://storage.googleapis.com/gweb-cloudblog-publish/images/tpu-131a5h.max-1000x1000.PNG)

### 参考文献

1. [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760)
2. [An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)
3. [Google Tensor G3: The new chip that gives your Pixel an AI upgrade](https://blog.google/products/pixel/google-tensor-g3-pixel-8/)
4. [Wikipedia-Tensor Processing Unit](https://en.wikipedia.org/wiki/Tensor_Processing_Unit#cite_note-TPU_memory-15)
5. 
