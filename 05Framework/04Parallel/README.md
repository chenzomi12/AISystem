<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# 分布式并行

什么是大模型？大模型模型参数量实在太大，需要分布式并行训练能力一起来加速训练过程。分布式并行是在大规模 AI 集群上工作的，想要加速就需要软硬件协同，不仅仅要解决通信拓扑的问题、集群组网的问题，还要了解上层 MOE、Transform 等新兴算法。通过对算法的剖析，提出模型并行、数据并行、优化器并行等新的并行模式和通信同步模式，来加速分布式训练的过程。最小的单机执行单元里面，还要针对大模型进行混合精度、梯度累积等算法，进一步压榨集群的算力！

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

**内容大纲**

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/AISystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 分布式并行 | 01 基本介绍| [PPT](./01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1ve411w7DL/) |
| 分布式并行 | 02 数据并行| [PPT](./02.data_parallel.pdf), [视频](https://www.bilibili.com/video/BV1JK411S7gL/) |
| 分布式并行 | 03 模型并行之张量并行 | [PPT](./03.tensor_parallel.pdf), [视频](https://www.bilibili.com/video/BV1vt4y1K7wT/) |
| 分布式并行 | 04 MindSpore 张量并行 | [PPT](./04.mindspore_parallel.pdf), [视频](https://www.bilibili.com/video/BV1vt4y1K7wT/) |
| 分布式并行 | 05 模型并行之流水并行 | [PPT](./05.pipeline_parallel.pdf), [视频](https://www.bilibili.com/video/BV1WD4y1t7Ba/) |
| 分布式并行 | 06 混合并行| [PPT](./06.hybrid_parallel.pdf), [视频](https://www.bilibili.com/video/BV1gD4y1t7Ut/) |
| 分布式汇总 | 07 分布式训练总结 | [PPT](./07.summary.pdf), [视频](https://www.bilibili.com/video/BV1av4y1S7DQ/) |

```toc
:maxdepth: 2

01.introduction
02.data_parallel
03.tensor_parallel
04.mindspore_parallel
05.pipeline_parallel
06.hybrid_parallel
07.summary
```
