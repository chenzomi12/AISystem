<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# 常量折叠&冗余节点消除

## 前文回顾

在上一节，主要围绕计算图优化的相关方式进行了整体介绍，现在整体再回顾一下图优化方式的具体组成：

1.Basic: 基础优化涵盖了所有保留计算图语义的修改，如：O1常量折叠、O2冗余节点消除和O3有限数量的算子融合。

2.Extended: 扩展优化仅在运行特定后端，如 CPU、CUDA、NPU  后端执行提供程序时适用。其针对硬件进行特殊且复杂的 Kernel 融合策略和方法。

3.Layout & Memory: 布局转换优化，主要是不同 AI 框架，在不同的硬件后端训练又在不同的硬件后端执行，数据的存储和排布格式不同。

对应到推理引擎的计算流程中，如下图所示，在预优化的阶段会进行很多代数相关的优化和简化，接着在优化阶段会更多的结合神经网络的知识进行优化，而在后优化阶段更多的是对一些数据的格式，内存的布局的重排，以及一些很重要核心的重复的算子的 kernel 进行合并。

![工作流程](image/graph/image.png)

特别需要注意的是，并非所有图优化都是基于模板去写的，它更多地应用于推理引擎或者大多数的推理引擎上。在 AI 框架当中则相反，假设以 TVM 为例，它更多的是去发现一些常用的规则去对计算图进行一个融合优化。下面将通过对比说明 AI 框架与推理引擎之间的差异：

| 特性 | AI 框架 | 推理引擎 |
| ----- | ----- | ----- |
| 用途 | 应用创新 | 覆盖主要的场景 |
| 部署 | 计算服务中心/算力平台 | 离线/AOT |
| 时间重要性 | 中 | 高 |

## 计算图优化详解

### Basic Graph Optimizations 基础图优化

基础优化涵盖了所有保留语义的修改，如常量折叠、冗余节点消除和有限数量的节点融合。接下来将围绕以下内容进行原理的详细展开：

1.Constant folding 常量折叠

2.Redundant eliminations 冗余节点消除

3.Operation fusion 算子融合

4.Operation Replace 算子替换

5.Operation Forward 算子前移

## 计算图优化-常量折叠

### Constant Folding 常量折叠

常量折叠：Constant folding，常量折叠，编译器优化技术之一，通过对编译时常量或常量表达式进行计算来简化代码。常量折叠是将计算图中可以预先可以确定输出值的节点替换成常量，并对计算图进行一些结构简化的操作。

传统编译器常量折叠，示例代码如下所示:

```c 语言
int main() {
    const int a = 100;
    int b = a + 100;
    printf(“b is %d\n”, b);
    return 0;
}
```

其中 b 的值只依赖于 a 的值, 由于 a 为常量, 其取值在编译阶段就确定了, 因此也可以在编译阶段计算得到 b 的值, 所以上面的代码经过常量折叠优化后,等价于下面的代码:

```c 语言
int main() {
    printf(“b is %d\n”, 200);
    return 0;
}
```

深度学习编译器中的常量折叠和传统编译器是类似的, 只需将输入变为 Tensor 即可. 比如对于下面的一个网络:

```c 语言
A = const([3, 5], 1.0, fp32)
B = const([3, 5], 0.5, fp32)
C = var([5, 4], fp32)
D = dot(A + B, C)
```

通过常量折叠后, 就可以变为:

```c 语言
TMP = const([3, 5], 1.5, fp32)
C = var([5, 4], fp32)
D = dot(TMP, C)
```

具体方法如下所示：

|  |  |  |
| ----- | ----- | ----- |
| Constant folding | Const 折叠 | 常量折叠如果一个 Op 所有输入都是常量 Const，可以先计算好结果 Const 代替该 Op，而不用每次都在推理阶段都计算一遍 |
| Fold Const To ExpandDims | ExpandDims 折叠 | ExpandDims Op 指定维度的输入是常量 Const，则把这个维度以参数的形式折叠到 ExpandDims 算子中 |
| Fuse Const To Binary | Binary 折叠 | Binary Op 第二个输入是标量 Const，把这个标量以参数形式折叠到 Binary Op 的属性中 |

- Constant folding 常量折叠：如果一个 Op 所有输入都是常量 Const，可以先计算好结果Const 代替该 Op，而不用每次都在推理阶段都计算一遍。

![Constant folding](image/graph/const_folding.png)

- ExpandDims 折叠： ExpandDims Op 指定维度的输入是常量 Const，则把这个维度以参数的形式折叠到 ExpandDims 算子中。

![ExpandDims](image/graph/ExpandDims.png)

- Binary 折叠： Binary Op 第二个输入是标量 Const ，把这个标量以参数的形式折叠到 Binary Op 的属性中。

![Binary](image/graph/Binary.png)

## 计算图优化-冗余节点消除

冗余节点消除：在不改变图形结构的情况下删除所有冗余节点，目前支持 Identity Elimination、Slice Elimination、Unsqueeze Elimination、 Dropout Elimination 优化方式

### Op 本身无意义

有些 Op 本身不参与计算，在推理阶段可以直接去掉对结果没有影响。如下图所示，在转换前后类型相同的 cast，只有一个输入 tensor 的 concat，以及 Seq2Out、Identity、NoOp、Print、Assert、StopGradient、Split 等算子均可以通过一系列的模板删除包括 dropout 算子。

![Op 本身无意义](image/graph/op_without_meaning.png)

具体示例如下图所示：

当图中存在冗余算子时，可能会出现以下三种情况：

1、当前冗余算子的输出对于下一个节点是有意义的：可以直接去除冗余算子，然后将上一个算子的输出和下一个算子的输入相连

2、当前冗余算子的输出对于下一个节点是无意义的：此时可以把它切成两个子图，一个子图就是 input->op1，另一个子图则是 op2->output

3、当前冗余算子的输入对于下一个节点是无意义的：只要这个节点的输入没有意义，轮循删除往上的节点，直到输入有意义为止。

![Op 无意义示例](image/graph/op_mean_example.png)

### Op 参数无意义

有些 Op 本身是有意义，但是设置成某些参数后就变成了无意义了的 Op。典型示例如 cast 算子，其主要是对数据的排布进行转换，当输入的参数等于输出的参数的时候，算子本身则无意义且可删除。还有很多种其他情况下的算子，在删除处理后，实践证明对于模型性能的提升具有极大的帮助。如下图所示：

![Op 参数无意义](image/graph/op_param.png)

详细示例如下所示：

（1）对于 cast 算子，当它的 source 等于 destination 的时候，cast 算子可以删除

（2）对于 ExpandDims 算子，当输出的 shape 跟输入的 shape 是一致时，ExpandDims 算子可以删除

（3）对于 slice/pooling 算子，index_start 等于 0 或者 index_end 等于 channel-1 以及 pooling 算子的窗口为 1x1 的时候，算子均可删除

![Op 参数无意义示例](image/graph/op_param_example.png)

### Op 位置无意义

一些 Op 在计算图中特殊位置会变得多余无意义。

![Op 位置无意义](image/graph/op_position.png.png)

详细示例如下所示：

示例中的 cast 算子，unsqueeze 算子以及无后续输出的 op1 和在 global pooling 之后的 reshape/flatten 算子等，均可以进行冗余算子的消除。

![Op 位置无意义](image/graph/op_position_example.png)

![Op 位置无意义](image/graph/op_position_example.png)

### Op 前后反义

前后两个相邻 Op 进行操作时，语义相反的两个 Op 都可以删除

|  |  |
| ----- | ----- |
| Squeeze ExpandDims Eliminate | Squeeze和ExpandDims这两个Op是反义的,一个压缩维度，一个是拓展维度，当连续的这两个Op指定的axis相等时即可同时删除这两个Op |
| Inverse Cast Eliminate | 当连续的两个内存排布转换Op的参数前后反义，即src1等于dst2,可同时删除这两个 Op |
| Quant Dequant Eliminate | 连续进行量化和反量化，可同时删除这两个 Op |
| Concat Slice Elimination | 合并后又进行同样的拆分，可同时删除这两个 Op |

详细示例如下所示：可参考上述规则，对于存在前后反义算子的情况，进行冗余节点的消除

![Op 前后反义](image.png)

### 公共子图

 在一个深度神经网络中，如果几个子图的类型、参数和输入均相同, 则将他们称做公共子图。 对于公共子图, 只需要计算其中一个子图的值, 其他子图的值可以通过赋值得到。这个过程就称作公共子图消除, 它是一种传统编译器中常用的优化手段, 经过迁移也可以应用到深度学习编译器中。

Common Subexpression Elimination：当模型当中出现了公共子图，如一个输出是另外两个同类型同参数的Op的输入，则可进行删除其中一个Op。

基本思路是通过一个 MAP 表, 记录截止当前, 已处理过的同一种类型的 OP。 对于当前正在处理的 OP, 先查找该 MAP 表, 如果能找到其他和正在处理的 OP 类型相同的 OP, 则对他们进行遍历, 如果其中某个 OP 的输入和参数与当前正在处理的 OP 相同, 则它们为公共子表达式, 结果可以互相替代；如果所有 OP 都不能与当前正在处理的 OP 匹配, 则将当前 OP 复制一份返回。

![公共子图](image.png)

## 计算图优化02回顾

1）小结 

本章节简要围绕计算图优化中常量折叠&冗余节点消除进行了介绍，在了解计算图优化的相关方式的基础上，针对常量折叠和冗余节点消除进行了详细的展开，重点探讨了 cast 算子、ExpandDims 算子、Squeeze 算子以及 Slice 等算子在神经网络中不同搭配组合时，可以进行优化的情况，以达到减少重复计算和冗余计算的目的。

2）视频更新链接：<iframe src="https://www.bilibili.com/video/BV1g84y1L7tF/?vd_source=48d8e5ac90484eed50f6a9e77c0e730e&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
