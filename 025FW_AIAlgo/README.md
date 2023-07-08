# 分布式训练

什么是大模型？大模型模型参数量实在太大，需要分布式并行训练能力一起来加速训练过程。分布式并行是在大规模AI集群上工作的，想要加速就需要软硬件协同，不仅仅要解决通信拓扑的问题、集群组网的问题，还要了解上层MOE、Transform等新兴算法。通过对算法的剖析，提出模型并行、数据并行、优化器并行等新的并行模式和通信同步模式，来加速分布式训练的过程。最小的单机执行单元里面，还要针对大模型进行混合精度、梯度累积等算法，进一步压榨集群的算力！

我在这里抛砖引玉，希望您可以一起参与到这个开源项目中，跟更多的您一起探讨学习！

## 内容大纲

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 分布式集群 | 01 基本介绍 | [silde](./01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1ge411L7mi/) |
| 分布式集群 | 02 AI集群服务器架构| [silde](./02.architecture.pdf), [视频](https://www.bilibili.com/video/BV1fg41187rc/) |
| 分布式集群 | 03 AI集群软硬件通信| [silde](./03.communication.pdf), [视频](https://www.bilibili.com/video/BV14P4y1S7u4/)|
| 分布式集群 | 04 集合通信原语 | [silde](./04.primitive.pdf), [视频](https://www.bilibili.com/video/BV1te4y1e7vz/)|
| 分布式算法 | 05 AI框架分布式功能| [silde](./05.system.pdf), [视频](https://www.bilibili.com/video/BV1n8411s7f3/) |
