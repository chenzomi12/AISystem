# AI框架前端的核心模块

AI系统里面，其实大部分开发者并不关心AI框架或者AI框架的前端，因为AI框架作为一个工具，最大的目标就是帮助更多的算法工程师快速实现他们的算法想法；另外一方面是帮助系统工程师，快速对算法进行落地部署和性能优化。

## 课程简介

- 《AI框架基础》主要是对AI框架的作用、发展、编程范式等散点进行汇总分享，让开发者能够知道AI框架与AI框架之间的差异和共同点，目前的AI框架主要的开发和编程方式。

- 其次，AI框架会默认提供自动微分功能，避免用户手动地去对神经网络模型求导，这些复杂的工作交给AI框架就好了，于是有了《自动微分》系列的内容。

- 有了AI框架，但实际上AI框架主要的职责是把深度学习的表达转换称为计算机能够识别的计算图，计算图作为AI框架中核心的数据结构，贯穿AI框架的整个生命周期，于是《计算图》这一章对于AI框架的前端核心模块就显得尤为重要。

- 最后是《分布式并行训练》，其实这一章可以单独成节，不过随着AI的发展，分布式训练、并行策略、AI集群、集群通信已经成为了AI框架的必配武器，所以也把分布式并行相关的内容作为AI框架前端的核心模块。

希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

## 课程目标

AI是最新的电力，大约在一百年前，我们社会的电气化改变了每个主要行业，从交通运输行业到制造业、医疗保健、通讯等方面，我认为如今我们见到了AI明显的令人惊讶的能量，带来了同样巨大的转变。显然，AI的各个分支中，发展的最为迅速的就是深度学习。因此现在，深度学习是在科技世界中广受欢迎的一种技巧。

通过《AI框架的核心模块》这个课程，以及这门课程后面的几门课程，你将获取并且掌握的技能：

| 编号  | 名称                     | 具体内容                                                                                                |
|:---:|:---------------------- |:--------------------------------------------------------------------------------------------------- |
| 1   | [AI框架基础](./01%20Foundation) | 对AI框架的作用、发展、编程范式等散点进行汇总分享，让开发者能够知道AI框架与AI框架之间的差异和共同点，目前的AI框架主要的开发和编程方式。                             |
| 2   | [自动微分](./02%20AutoDiff)     | AI框架会默认提供自动微分功能，避免用户手动地去对神经网络模型求导，这些复杂的工作交给AI框架就好了，于是自动微自然成为分作为AI框架的核心功能。                           |
| 3   | [计算图](./03%20DataFlow)      | 实际上，AI框架主要的职责是把深度学习的表达转换称为计算机能够识别的计算图，计算图作为AI框架中核心的数据结构，贯穿AI框架的大部分整个生命周期，于是计算图对于AI框架的前端核心技术就显得尤为重要。 |
| 4   | [分布式集群](./04%20AICluster)   | 随着AI集群的出现，越来越多的网络模型运行在集群上面，但是AI集群如何管理？如何通信？如何协同工作？AI框架如何支持分布式功能都需要我们去了解，才能更好地利用AI集群算力。              |
| 5   | [分布式算法](./05%20AIAlgo)   | 随着大模型的出现，越来越多的大模型算法涌现，特别是Transformer和MOE结构，引爆了千亿乃至万亿规模的大模型，新的AI算法奇点来了，AI工程师也需要了解最新的动态。              |
| 6   | [分布式并行](./06%20Parallel)    | 可是在AI集群，想要训练起千亿乃至万亿规模的大模型，谈何容易，于是出现了不同类型的分布式并行策略，目的是解决性能墙、内存墙、调优墙等并行问题，使的开发者能够真正让AI算法快速在AI集群上执行。    |
|     |                        |                                                                                                     |

- 在《AI框架基础》第一门课程中，您将了解到AI框架的具体作用，可以提供给开发者一个编写神经网络模型的库和提供丰富的API。以及近几年AI框架快速发展的历史和变迁。在这门课程的结尾，您将了解到不同的编程范式对AI框架的影响和对用户习惯的影响。

- 在《自动微分》中，你将深入了解微分和微分的不同方式，其中自动微分是微分很重要的实现方法之一，对于传统的几种微分方式有其独特的优势，这里面将会深入自动微分的正反向模式，不过在代码具体实现却有千差万别，于是我们将会手把手去用代码实现2种不同模式的自动微分。

- 在《计算图》这一门课程，您将掌握AI框架的核心表示：计算图。这里主要了解到什么是计算图和对计算图如何表示以外，还会了解到计算图跟自动微分的关系，如何表示反向模型和梯度，最后您还可以了解计算图中最难表达的控制流和动静统一的核心技术。

- 最后在《分布式并行》课程中，我们将会分为分布式集群、分布式训练、分布式算法、分布式并行策略4大内容进行展开，每一个内容都较为独立，于是最后我们将会把上面的技术串起来。

## 课程细节

|     |        |                   |                                                                                                                                                      |
| --- | ------ | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 编号  | 名称     | 名称                | 备注                                                                                                                                                   |
| 1   | AI框架基础 | 01 基本介绍           | [silde](./01 Foundation/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1he4y1z7oD/?vd_source=26de035c60e6c7f810371fdfd13d14b6)         |
|     | AI框架基础 | 02 AI框架有什么用       | [silde](./01 Foundation/02.fundamentals.pptx), [video](https://www.bilibili.com/video/BV1fd4y1q7qk/?vd_source=26de035c60e6c7f810371fdfd13d14b6)         |
|     | AI框架基础 | 03 AI框架之争（框架发展）   | [silde](./01 Foundation/03.history.pptx), [video](https://www.bilibili.com/video/BV1C8411x7Kn/?vd_source=26de035c60e6c7f810371fdfd13d14b6)              |
|     | AI框架基础 | 04 编程范式（声明式&命令式）  | [silde](./01 Foundation/04.programing.pptx), [video](https://www.bilibili.com/video/BV1gR4y1o7WT/?vd_source=26de035c60e6c7f810371fdfd13d14b6)           |
|     |        |                   |                                                                                                                                                      |
| 2   | 自动微分   | 01 基本介绍           | [silde](./02 AutoDiff/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1FV4y1T7zp/), [article](https://zhuanlan.zhihu.com/p/518198564)   |
|     | 自动微分   | 02 什么是微分          | [silde](./02 AutoDiff/02.base_concept.pptx), [video](https://www.bilibili.com/video/BV1Ld4y1M7GJ/), [article](https://zhuanlan.zhihu.com/p/518198564)   |
|     | 自动微分   | 03 正反向计算模式        | [silde](./02 AutoDiff/03.grad_mode.pptx), [video](https://www.bilibili.com/video/BV1zD4y117bL/), [article](https://zhuanlan.zhihu.com/p/518296942)      |
|     | 自动微分   | 04 三种实现方法         | [silde](./02 AutoDiff/04.grad_mode.pptx), [video](https://www.bilibili.com/video/BV1BN4y1P76t/), [article](https://zhuanlan.zhihu.com/p/520065656)      |
|     | 自动微分   | 05 手把手实现正向微分框架    | [silde](./02 AutoDiff/05.forward_mode.ipynb), [video](https://www.bilibili.com/video/BV1Ne4y1p7WU/), [article](https://zhuanlan.zhihu.com/p/520451681)  |
|     | 自动微分   | 06 亲自实现一个PyTorch  | [silde](./02 AutoDiff/06.reversed_mode.ipynb), [video](https://www.bilibili.com/video/BV1ae4y1z7E6/), [article](https://zhuanlan.zhihu.com/p/547865589) |
|     | 自动微分   | 07 自动微分的挑战&未来     | [silde](./02 AutoDiff/07.challenge.pptx), [video](https://www.bilibili.com/video/BV17e4y1z73W/)                                                         |
|     |        |                   |                                                                                                                                                      |
| 3   | 计算图    | 01 基本介绍           | [silde](./03%20DataFlow/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1cG411E7gV/)                                                      |
|     | 计算图    | 02 什么是计算图         | [silde](./03%20DataFlow/02.computation_graph.pptx), [video](https://www.bilibili.com/video/BV1rR4y197HM/)                                                 |
|     | 计算图    | 03 计算图跟自动微分关系     | [silde](./03%20DataFlow/03.atuodiff.pptx), [video](https://www.bilibili.com/video/BV1S24y197FU/)                                                          |
|     | 计算图    | 04 图优化与图执行调度      | [silde](./03%20DataFlow/04.dispatch.pptx),[video](https://www.bilibili.com/video/BV1hD4y1k7Ty/)                                                           |
|     | 计算图    | 05 计算图的控制流机制实现    | [silde](./03%20DataFlow/05.control_flow.pptx),[video](https://www.bilibili.com/video/BV17P41177Pk/)                                                       |
|     | 计算图    | 06 计算图未来将会走向何方？   | [silde](./03%20DataFlow/06.future.pptx),[video](https://www.bilibili.com/video/BV1hm4y1A7Nv/)                                                             |
|     |        |                   |                                                                                                                                                      |
| 4   | 分布式集群  | 01 基本介绍           | [silde](./04%20AICluster/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1ge411L7mi/)                                                     |
|     | 分布式集群  | 02 AI集群服务器架构      | [silde](./04%20AICluster/02.architecture.pptx), [video](https://www.bilibili.com/video/BV1fg41187rc/)                                                     |
|     | 分布式集群  | 03 AI集群软硬件通信      | [silde](./04%20AICluster/03.communication.pptx), [video](https://www.bilibili.com/video/BV14P4y1S7u4/)                                                    |
|     | 分布式集群  | 04 集合通信原语         | [silde](./04%20AICluster/04.primitive.pptx), [video](https://www.bilibili.com/video/BV1te4y1e7vz/)                                                        |
|     | 分布式算法  | 05 AI框架分布式功能      | [silde](./04%20AICluster/05.system.pptx), [video](https://www.bilibili.com/video/BV1n8411s7f3/)                                                           |
|     |        |                   |                                                                                                                                                      |
| 5   | 分布式算法  | 06 大模型训练的挑战       | [silde](./05%20AIAlgo/06.challenge.pptx), [video](https://www.bilibili.com/video/BV1Y14y1576A/)                                                        |
|     | 分布式算法  | 07 算法：大模型算法结构     | [silde](./05%20AIAlgo/07.algorithm_arch.pptx), [video](https://www.bilibili.com/video/BV1Mt4y1M7SE/)                                                   |
|     | 分布式算法  | 08 算法：亿级规模SOTA大模型 | [silde](./05%20AIAlgo/08.algorithm_sota.pptx), [video](https://www.bilibili.com/video/BV1em4y1F7ay/)                                                   |
|     |        |                   |                                                                                                                                                      |
| 6   | 分布式并行  | 01 基本介绍           | [silde](./06%20Parallel/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1ve411w7DL/)                                                      |
|     | 分布式并行  | 02 数据并行           | [silde](./06%20Parallel/02.data_parallel.pptx), [video](https://www.bilibili.com/video/BV1JK411S7gL/)                                                     |
|     | 分布式并行  | 03 模型并行之张量并行      | [silde](./06%20Parallel/03.tensor_parallel.pptx), [video](https://www.bilibili.com/video/BV1vt4y1K7wT/)                                                   |
|     | 分布式并行  | 04 MindSpore张量并行  | [silde](./06%20Parallel/04.mindspore_parallel.pptx), [video](https://www.bilibili.com/video/BV1vt4y1K7wT/)                                                |
|     | 分布式并行  | 05 模型并行之流水并行      | [silde](./06%20Parallel/05.pipeline_parallel.pptx), [video](https://www.bilibili.com/video/BV1WD4y1t7Ba/)                                                 |
|     | 分布式并行  | 06 混合并行           | [silde](./06%20Parallel/06.hybrid_parallel.pptx), [video](https://www.bilibili.com/video/BV1gD4y1t7Ut/)                                                   |
|     | 分布式汇总  | 07 分布式训练总结        | [silde](./06%20Parallel/07.summary.pptx), [video](https://www.bilibili.com/video/BV1av4y1S7DQ/)                                                           |
|     |        |                   |                                                                                                                                                      |

## 目标学员

1. 对人工智能、深度学习感兴趣的学员

2. 渴望学习当今最热门最前沿技术的人 

3. 想储备深度学习技能的学员

4. AI框架开发工程师
