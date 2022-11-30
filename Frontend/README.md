# 前端核心模块

AI系统里面，其实很多人并不关系AI框架或者AI框架的前端，因为AI框架作为一个工具，最大的能力就是帮助更多的算法工程师快速实现他们的想法，另外一方面是帮助工程化工程师，快速对算法进行落地部署和API化。但是对于系统工程师来说，上述的两个方向就是其系统优化的目标，而前端核心模块应该是贴近用户的。

AI框架系统提供自动微分功能，避免用户手动地去对神经网络模型求导，这些复杂的工作交给AI框架就好了，于是有了《自动微分》系列的内容。其次的，《AI框架基础》主要是对AI框架的作用、发展、编程范式等散点进行汇总，让系统工程师能够知道AI框架与AI框架之间的差异和共同点。其次，有了AI框架，但实际上AI框架都是对计算图进行表达，计算图作为AI框架中核心的数据结构，贯穿整个生命周期，于是乎《计算图》这一章对于前端核心模块就显得尤为重要。最后是《分布式并行训练》，其实这一章可以单独成节，不过随着AI的发展，分布式、并行、AI集群已经成为了框架的必配品，所以最终决定把并行相关的内容作为AI框架前端的核心模块。

希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

## 课程部分

|     |        |                   |                                                                                                                                                      |
| --- | ------ | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 编号  | 名称     | 名称                | 备注                                                                                                                                                   |
| 1   | 自动微分   | 01 基本介绍           | [silde](./AutoDiff/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1FV4y1T7zp/), [article](https://zhuanlan.zhihu.com/p/518198564)   |
|     | 自动微分   | 02 什么是微分          | [silde](./AutoDiff/02.base_concept.pptx), [video](https://www.bilibili.com/video/BV1Ld4y1M7GJ/), [article](https://zhuanlan.zhihu.com/p/518198564)   |
|     | 自动微分   | 03 正反向计算模式        | [silde](./AutoDiff/03.grad_mode.pptx), [video](https://www.bilibili.com/video/BV1zD4y117bL/), [article](https://zhuanlan.zhihu.com/p/518296942)      |
|     | 自动微分   | 04 三种实现方法         | [silde](./AutoDiff/04.grad_mode.pptx), [video](https://www.bilibili.com/video/BV1BN4y1P76t/), [article](https://zhuanlan.zhihu.com/p/520065656)      |
|     | 自动微分   | 05 手把手实现正向微分框架    | [silde](./AutoDiff/05.forward_mode.ipynb), [video](https://www.bilibili.com/video/BV1Ne4y1p7WU/), [article](https://zhuanlan.zhihu.com/p/520451681)  |
|     | 自动微分   | 06 亲自实现一个PyTorch  | [silde](./AutoDiff/06.reversed_mode.ipynb), [video](https://www.bilibili.com/video/BV1ae4y1z7E6/), [article](https://zhuanlan.zhihu.com/p/547865589) |
|     | 自动微分   | 07 自动微分的挑战&未来     | [silde](./AutoDiff/07.challenge.pptx), [video](https://www.bilibili.com/video/BV17e4y1z73W/)                                                         |
|     |        |                   |                                                                                                                                                      |
| 2   | AI框架基础 | 01 基本介绍           | [silde](./Foundation/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1he4y1z7oD/?vd_source=26de035c60e6c7f810371fdfd13d14b6)         |
|     | AI框架基础 | 02 AI框架有什么用       | [silde](./Foundation/02.fundamentals.pptx), [video](https://www.bilibili.com/video/BV1fd4y1q7qk/?vd_source=26de035c60e6c7f810371fdfd13d14b6)         |
|     | AI框架基础 | 03 AI框架之争（框架发展）   | [silde](./Foundation/03.history.pptx), [video](https://www.bilibili.com/video/BV1C8411x7Kn/?vd_source=26de035c60e6c7f810371fdfd13d14b6)              |
|     | AI框架基础 | 04 编程范式（声明式&命令式）  | [silde](./Foundation/04.programing.pptx), [video](https://www.bilibili.com/video/BV1gR4y1o7WT/?vd_source=26de035c60e6c7f810371fdfd13d14b6)           |
|     |        |                   |                                                                                                                                                      |
| 3   | 计算图    | 01 基本介绍           | [silde](./DataFlow/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1cG411E7gV/)                                                      |
|     | 计算图    | 02 什么是计算图         | [silde](./DataFlow/02.computation_graph.pptx), [video](https://www.bilibili.com/video/BV1rR4y197HM/)                                                 |
|     | 计算图    | 03 计算图跟自动微分关系     | [silde](./DataFlow/03.atuodiff.pptx), [video](https://www.bilibili.com/video/BV1S24y197FU/)                                                          |
|     | 计算图    | 04 图优化与图执行调度      | [silde](./DataFlow/04.dispatch.pptx),[video](https://www.bilibili.com/video/BV1hD4y1k7Ty/)                                                           |
|     | 计算图    | 05 计算图的控制流机制实现    | [silde](./DataFlow/05.control_flow.pptx),[video](https://www.bilibili.com/video/BV17P41177Pk/)                                                       |
|     | 计算图    | 06 计算图未来将会走向何方？   | [silde](./DataFlow/06.future.pptx),[video](https://www.bilibili.com/video/BV1hm4y1A7Nv/)                                                             |
|     |        |                   |                                                                                                                                                      |
| 4   | 分布式训练  | 01 基本介绍           | [silde](./Frontend/Distribution/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1ge411L7mi/)                                         |
|     | 分布式集群  | 02 AI集群服务器架构      | [silde](./Frontend/Distribution/04.architecture.pptx), [video](https://www.bilibili.com/video/BV1fg41187rc/)                                         |
|     | 分布式集群  | 03 AI集群软硬件通信      | [silde](./Frontend/Distribution/05.1.communication.pptx), [video](https://www.bilibili.com/video/BV14P4y1S7u4/)                                      |
|     | 分布式集群  | 04 集合通信原语         | [silde](./Frontend/Distribution/05.2.primitive.pptx), [video](https://www.bilibili.com/video/BV1te4y1e7vz/)                                          |
|     | 分布式算法  | 05 大模型训练的挑战       | [silde](./Frontend/Distribution/02.challenge.pptx), [video](https://www.bilibili.com/video/BV1n8411s7f3/)                                            |
|     | 分布式算法  | 06 AI框架分布式功能      | [silde](./Frontend/Distribution/03.system.pptx), [video](https://www.bilibili.com/video/BV1Y14y1576A/)                                               |
|     | 分布式算法  | 07 算法：大模型算法结构     | [silde](./Frontend/Distribution/06.algorithm_arch.pptx), [video](https://www.bilibili.com/video/BV1Mt4y1M7SE/)                                       |
|     | 分布式算法  | 08 算法：亿级规模SOTA大模型 | [silde](./Frontend/Distribution/06.algorithm_arch.pptx), [video](https://www.bilibili.com/video/BV1em4y1F7ay/)                                       |
|     | 分布式并行  | 09 并行策略：数据并行      | [silde](./Frontend/Distribution/07.1.data_parallel.pptx), [video](https://www.bilibili.com/video/BV1JK411S7gL/)                                      |
|     | 分布式并行  | 10 模型并行之张量并行      | [silde](./Frontend/Distribution/07.2.model_parallel.pptx), [video](https://www.bilibili.com/video/BV1vt4y1K7wT/)                                     |
|     | 分布式并行  | 11 MindSpore张量并行  | [silde](./Frontend/Distribution/07.2.model_parallel.pptx), [video](https://www.bilibili.com/video/BV1vt4y1K7wT/)                                     |
|     | 分布式并行  | 12 模型并行之流水并行      | [silde](./Frontend/Distribution/07.3.pipeline_parallel.pptx), [video](https://www.bilibili.com/video/BV1WD4y1t7Ba/)                                  |
|     | 分布式并行  | 13 混合并行           | [silde](./Frontend/Distribution/08.hybrid_parallel.pptx), [video](https://www.bilibili.com/video/BV1gD4y1t7Ut/)                                      |
|     | 分布式汇总  | 14 分布式训练总结        | [silde](./Frontend/Distribution/10.summary.pptx), [video](https://www.bilibili.com/video/BV1av4y1S7DQ/)                                              |

## 项目背景

近年来人工智能特别是深度学习技术得到了飞速发展，这背后离不开计算机硬件和软件系统的不断进步。在可见的未来，人工智能技术的发展仍将依赖于计算机系统和人工智能相结合的共同创新模式。需要注意的是，计算机系统现在正以更大的规模和更高的复杂性来赋能于人工智能，这背后不仅需要更多的系统上的创新，更需要系统性的思维和方法论。与此同时，人工智能也反过来为设计复杂系统提供支持。

我们注意到，现在的大部分人工智能相关的课程，特别是深度学习和机器学习相关课程主要集中在相关理论、算法或者应用，与系统相关的课程并不多见。我们希望人工智能系统这门课能让人工智能相关教育变得更加全面和深入，以共同促进人工智能与系统在开源方面的共同学习和讨论。

（原谅我复制粘贴微软[AI-System](https://github.com/microsoft/AI-System)的介绍，人家写得很好啦；另外推荐一个很好学习参考项目，公司跟英国麦络老师（爱丁堡大学）合作的[机器学习系统：设计和实现](https://github.com/openmlsys/openmlsys-zh)。）
