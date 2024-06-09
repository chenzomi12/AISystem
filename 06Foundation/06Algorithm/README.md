<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# 分布式算法

什么是大模型？大模型模型参数量实在太大，需要分布式并行训练能力一起来加速训练过程。分布式并行是在大规模 AI 集群上工作的，想要加速就需要软硬件协同，不仅仅要解决通信拓扑的问题、集群组网的问题，还要了解上层 MOE、Transform 等新兴算法。通过对算法的剖析，提出模型并行、数据并行、优化器并行等新的并行模式和通信同步模式，来加速分布式训练的过程。最小的单机执行单元里面，还要针对大模型进行混合精度、梯度累积等算法，进一步压榨集群的算力！

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

**内容大纲**

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/AISystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 分布式集群 | 01 基本介绍 | [PPT](./01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1ge411L7mi/) |
| 分布式集群 | 02 AI 集群服务器架构| [PPT](./02.architecture.pdf), [视频](https://www.bilibili.com/video/BV1fg41187rc/) |
| 分布式集群 | 03 AI 集群软硬件通信| [PPT](./03.communication.pdf), [视频](https://www.bilibili.com/video/BV14P4y1S7u4/) |
| 分布式集群 | 04 集合通信原语 | [PPT](./04.primitive.pdf), [视频](https://www.bilibili.com/video/BV1te4y1e7vz/) |
| 分布式算法 | 05 AI 框架分布式功能| [PPT](./05.system.pdf), [视频](https://www.bilibili.com/video/BV1n8411s7f3/) |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AISys](https://chenzomi12.github.io/) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT 开源在[github](https://github.com/chenzomi12/AISystem)，欢迎取用！！！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
>
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交代码 PR 到开源社区哦！
>
> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！
