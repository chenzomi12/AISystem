<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# ==== 六、大模型训练

- 待更新...

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

## 课程简介

- 待更新...

## 课程细节

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

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
