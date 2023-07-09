<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# == 二、AI框架核心模块 ==

AI系统里面，其实大部分开发者并不关心AI框架或者AI框架的前端，因为AI框架作为一个工具，最大的目标就是帮助更多的算法工程师快速实现他们的算法想法；另外一方面是帮助系统工程师，快速对算法进行落地部署和性能优化。

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

## 课程简介

- 《AI框架基础》主要是对AI框架的作用、发展、编程范式等散点进行汇总分享，让开发者能够知道AI框架与AI框架之间的差异和共同点，目前的AI框架主要的开发和编程方式。

- 《自动微分》AI框架会默认提供自动微分功能，避免用户手动地去对神经网络模型求导，这些复杂的工作交给AI框架就好了，于是自动微自然成为分作为AI框架的核心功能。

- 《计算图》实际上，AI框架主要的职责是把深度学习的表达转换称为计算机能够识别的计算图，计算图作为AI框架中核心的数据结构，贯穿AI框架的大部分整个生命周期，于是计算图对于AI框架的前端核心技术就显得尤为重要。

- 《分布式集群》随着AI集群的出现，越来越多的网络模型运行在集群上面，但是AI集群如何管理？如何通信？如何协同工作？AI框架如何支持分布式功能都需要我们去了解，才能更好地利用AI集群算力。

- 《分布式算法》随着大模型的出现，越来越多的大模型算法涌现，特别是Transformer和MOE结构，引爆了千亿乃至万亿规模的大模型，新的AI算法奇点来了，AI工程师也需要了解最新的动态。

- 《分布式并行》可是在AI集群，想要训练起千亿乃至万亿规模的大模型，谈何容易，于是出现了不同类型的分布式并行策略，目的是解决性能墙、内存墙、调优墙等并行问题，使的开发者能够真正让AI算法快速在AI集群上执行。

## 课程细节

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

### AI框架基础

《AI框架基础》主要是对AI框架的作用、发展、编程范式等散点进行汇总分享，让开发者能够知道AI框架与AI框架之间的差异和共同点，目前的AI框架主要的开发和编程方式。

| 小节 | 链接|
|:--:|:--:|
| 01 基本介绍| [文章](../021FW_Foundation/01.introduction.md), [PPT](../021FW_Foundation/01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1he4y1z7oD), [字幕](../021FW_Foundation/srt/01.srt) |
| 02 AI框架的作用| [文章](../021FW_Foundation/02.fundamentals.md), [PPT](../021FW_Foundation/02.fundamentals.pdf), [视频](https://www.bilibili.com/video/BV1fd4y1q7qk), [字幕](../021FW_Foundation/srt/02.srt) |
| 03 AI框架之争（框架发展）| [文章](../021FW_Foundation/03.history.md), [PPT](../021FW_Foundation/03.history.pdf), [视频](https://www.bilibili.com/video/BV1C8411x7Kn), [字幕](../021FW_Foundation/srt/03.srt) |
| 04 编程范式（声明式&命令式） | [文章](../021FW_Foundation/04.programing.md), [PPT](../021FW_Foundation/04.programing.pdf), [视频](https://www.bilibili.com/video/BV1gR4y1o7WT), [字幕](../021FW_Foundation/srt/04.srt) |

### 自动微分

《自动微分》AI框架会默认提供自动微分功能，避免用户手动地去对神经网络模型求导，这些复杂的工作交给AI框架就好了，于是自动微自然成为分作为AI框架的核心功能。

| 分类 | 名称 | 内容 | 
|:-:|:-:|:-:|
| 2 | 自动微分 | 01 基本介绍 | [PPT](./02 AutoDiff/01.introduction.pptx), [视频](https://www.bilibili.com/video/BV1FV4y1T7zp/) |
| 自动微分 | 02 什么是微分| [PPT](./02 AutoDiff/02.base_concept.pptx), [视频](https://www.bilibili.com/video/BV1Ld4y1M7GJ/) |
| 自动微分 | 03 正反向计算模式| [PPT](./02 AutoDiff/03.grad_mode.pptx), [视频](https://www.bilibili.com/video/BV1zD4y117bL/) |
| 自动微分 | 04 三种实现方法 | [PPT](./02 AutoDiff/04.grad_mode.pptx), [视频](https://www.bilibili.com/video/BV1BN4y1P76t/) |
| 自动微分 | 05 手把手实现正向微分框架| [PPT](./02 AutoDiff/05.forward_mode.ipynb), [视频](https://www.bilibili.com/video/BV1Ne4y1p7WU/) |
| 自动微分 | 06 亲自实现一个PyTorch| [PPT](./02 AutoDiff/06.reversed_mode.ipynb), [视频](https://www.bilibili.com/video/BV1ae4y1z7E6/) |
| 自动微分 | 07 自动微分的挑战&未来 | [PPT](./02 AutoDiff/07.challenge.pptx), [视频](https://www.bilibili.com/video/BV17e4y1z73W/) |

### 计算图

《计算图》实际上，AI框架主要的职责是把深度学习的表达转换称为计算机能够识别的计算图，计算图作为AI框架中核心的数据结构，贯穿AI框架的大部分整个生命周期，于是计算图对于AI框架的前端核心技术就显得尤为重要。

| 分类 | 名称 | 内容 | 
|:-:|:-:|:-:|
| 计算图| 01 基本介绍 | [PPT](./03_DataFlow/01.introduction.pptx), [视频](https://www.bilibili.com/video/BV1cG411E7gV/) |
| 计算图| 02 什么是计算图 | [PPT](./03_DataFlow/02.computation_graph.pptx), [视频](https://www.bilibili.com/video/BV1rR4y197HM/) |
| 计算图| 03 计算图跟自动微分关系 | [PPT](./03_DataFlow/03.atuodiff.pptx), [视频](https://www.bilibili.com/video/BV1S24y197FU/) |
| 计算图| 04 图优化与图执行调度| [PPT](./03_DataFlow/04.dispatch.pptx), [视频](https://www.bilibili.com/video/BV1hD4y1k7Ty/) |
| 计算图| 05 计算图的控制流机制实现| [PPT](./03_DataFlow/05.control_flow.pptx), [视频](https://www.bilibili.com/video/BV17P41177Pk/) |
| 计算图| 06 计算图未来将会走向何方？ | [PPT](./03_DataFlow/06.future.pptx), [视频](https://www.bilibili.com/video/BV1hm4y1A7Nv/) |

### 分布式集群

《分布式集群》随着AI集群的出现，越来越多的网络模型运行在集群上面，但是AI集群如何管理？如何通信？如何协同工作？AI框架如何支持分布式功能都需要我们去了解，才能更好地利用AI集群算力。

| 分类 | 名称 | 内容 | 
|:-:|:-:|:-:|
| 分布式集群| 01 基本介绍 | [PPT](./04_AICluster/01.introduction.pptx), [视频](https://www.bilibili.com/video/BV1ge411L7mi/) |
| 分布式集群| 02 AI集群服务器架构| [PPT](./04_AICluster/02.architecture.pptx), [视频](https://www.bilibili.com/video/BV1fg41187rc/) |
| 分布式集群| 03 AI集群软硬件通信| [PPT](./04_AICluster/03.communication.pptx), [视频](https://www.bilibili.com/video/BV14P4y1S7u4/) |
| 分布式集群| 04 集合通信原语 | [PPT](./04_AICluster/04.primitive.pptx), [视频](https://www.bilibili.com/video/BV1te4y1e7vz/) |
| 分布式算法| 05 AI框架分布式功能| [PPT](./04_AICluster/05.system.pptx), [视频](https://www.bilibili.com/video/BV1n8411s7f3/) |

### 分布式算法

《分布式算法》随着大模型的出现，越来越多的大模型算法涌现，特别是Transformer和MOE结构，引爆了千亿乃至万亿规模的大模型，新的AI算法奇点来了，AI工程师也需要了解最新的动态。

| 分类 | 名称 | 内容 | 
|:-:|:-:|:-:|
| 分布式算法| 06 大模型训练的挑战 | [PPT](./05_AIAlgo/06.challenge.pptx), [视频](https://www.bilibili.com/video/BV1Y14y1576A/) |
| 分布式算法| 07 算法：大模型算法结构 | [PPT](./05_AIAlgo/07.algorithm_arch.pptx), [视频](https://www.bilibili.com/video/BV1Mt4y1M7SE/) |
| 分布式算法| 08 算法：亿级规模SOTA大模型 | [PPT](./05_AIAlgo/08.algorithm_sota.pptx), [视频](https://www.bilibili.com/video/BV1em4y1F7ay/) |

### 分布式并行

《分布式并行》可是在AI集群，想要训练起千亿乃至万亿规模的大模型，谈何容易，于是出现了不同类型的分布式并行策略，目的是解决性能墙、内存墙、调优墙等并行问题，使的开发者能够真正让AI算法快速在AI集群上执行。

| 分类 | 名称 | 内容 | 
|:-:|:-:|:-:|
| 分布式并行| 01 基本介绍 | [PPT](./06_Parallel/01.introduction.pptx), [视频](https://www.bilibili.com/video/BV1ve411w7DL/) |
| 分布式并行| 02 数据并行 | [PPT](./06_Parallel/02.data_parallel.pptx), [视频](https://www.bilibili.com/video/BV1JK411S7gL/) |
| 分布式并行| 03 模型并行之张量并行| [PPT](./06_Parallel/03.tensor_parallel.pptx), [视频](https://www.bilibili.com/video/BV1vt4y1K7wT/) |
| 分布式并行| 04 MindSpore张量并行| [PPT](./06_Parallel/04.mindspore_parallel.pptx), [视频](https://www.bilibili.com/video/BV1vt4y1K7wT/) |
| 分布式并行| 05 模型并行之流水并行| [PPT](./06_Parallel/05.pipeline_parallel.pptx), [视频](https://www.bilibili.com/video/BV1WD4y1t7Ba/) |
| 分布式并行| 06 混合并行 | [PPT](./06_Parallel/06.hybrid_parallel.pptx), [视频](https://www.bilibili.com/video/BV1gD4y1t7Ut/) |
| 分布式汇总| 07 分布式训练总结| [PPT](./06_Parallel/07.summary.pptx), [视频](https://www.bilibili.com/video/BV1av4y1S7DQ/) |
