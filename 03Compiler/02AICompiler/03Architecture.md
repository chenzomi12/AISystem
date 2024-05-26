# AI 编译器的通用架构

在上一节中将AI编译器的发展大致分为了3个stage，分别为stage1：朴素编译器、stage2：专用编译器以及stage3：通用编译器。

本节作为上一节AI编译器架构的一个延续，着重讨论AI编译器的通用架构。首先将回顾现有Ai编译器架构（以PyTorch作为标杆），随后引出通用AI编译器的架构模型，并进一步介绍其IR中间表达层、前端优化以及后端优化的细节，最后以一图流的形式展示现有AI编译器全栈产品。

## 现有AI编译器架构回顾

现有AI编译器架构即是专用AI编译器的架构：在表达上以PyTorch作为标杆，对静态图做转换，转换到计算图层IR进行优化；性能上希望打开计算图和算子的边界，进行重新组合优化以发挥芯片尽可能多的算力。

现有AI编译器架构图如下图所示。

![现有AI编译器架构图](images/sources/architecture01.png)

此编译器接受的高级语言为Python，编译器前端会对Python代码进行解析，解析器会将高层次的代码转换为一个中间表示（IR），以便进一步处理。这里编译器前端会生成Graph IR传递给Graph Optimizer（图优化器）。

Graph Optimizer接收到Graph IR后，会对解析后的计算图进行优化。优化的目的是减少计算图的冗余部分，提高执行效率。这可能包括算子融合、常量折叠等技术。Graph Optimizer在优化完成后会向Ops Optimizer（操作优化器）传递一个Tensor IR。

Ops Optimizer接收到Tensor IR后，其会针对每个算子进行具体的性能优化，例如重排计算顺序、内存优化等。

所有的中间表达都传递至后端之后，后端会生成不同的硬件代码以及可执行程序。

## AI编译器通用架构

在回顾完现有AI编译器架构后，来看看一个理想化的AI编译器通用架构应该是什么样的。

笔者推荐各位读者了解一篇关于AI编译器的综述，名称为The Deep Learning Compiler: A Comprehensive Survey。其中有一副插图展示了一个通用AI编译器的完整架构，涵盖从模型输入到在不同硬件平台上执行的整个流程。它分为编译器前端（Compiler Frontend）和编译器后端（Compiler Backend）两个主要部分。下面将结合此图对通用AI编译器进行初步分析。

![通用AI编译器架构图](images/sources/architecture02.png)

### 编译器前端（Compiler Frontend）

前端主要负责接收和处理来自不同深度学习框架的模型，并将其转换为通用的中间表示（IR），进行初步优化。

*Input Format of DL Models（输入格式）*：支持多种深度学习框架，如TensorFlow、PyTorch、Caffe2、MXNet、飞桨（PaddlePaddle）和ONNX等。

*Transformation（转换）*：将来自不同框架的模型转换为统一的表示形式。常见的转换方式包括：TVM的Relay、nGraph的Bridge、PyTorch的ATen（TorchScript）或直接翻译等。

*High-level IR / Graph IR（高层次IR/图IR）*：这些IR是设备无关的，主要用于表示计算图。表示方法包括DAG（有向无环图）和基于绑定的张量计算等。实现部分涉及数据表示和操作符支持。

当初步生成Computation Graph(计算图)后，会通过一些方法对计算图进行进一步的优化。

*Computation Graph Optimizations（计算图优化策略）*：构建计算图并进行优化，具体优化包括：Algebraic simplification（代数简化）、Operator fusion（算子融合）、Operation sinking（操作下沉）、CSE（公共子表达式消除）、DCE（死代码消除）、Static memory planning（静态内存优化）、Layout transformation（布局转换）等。

*Methods（计算图进一步优化方法）*：对计算图进行进一步优化，使用各种方法如Pattern matcher（模式匹配）和Graph rewriting（图重写）等。

*Debug Tools（调试工具）*：提供调试工具，如IR转储（文本形式和DAG形式）等。

### 编译器后端











## 本节视频

<html>
<iframe src="https:&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>
