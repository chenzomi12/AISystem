<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# 分布式并行

分布式训练可以将模型训练任务分配到多个计算节点上,从而加速训练过程并处理更大的数据集。模型是一个有机的整体，简单增加机器数量并不能提升算力，需要有并行策略和通信设计，才能实现高效的并行训练。本节将会重点打开业界主流的分布式并行框架 DeepSpeed、Megatron-LM 的核心多维并行的特性来进行原理介绍。

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

**内容大纲**

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/AISystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 分布式并行 | 01 分布式并行框架介绍 | [PPT](./01Introduction.pdf), [视频](https://www.bilibili.com/video/BV1op421C7wp/) |
| 分布式并行 | 02 DeepSpeed 介绍 | [PPT](./02DeepSpeed.pdf), [视频](https://www.bilibili.com/video/BV1tH4y1J7bm/) |
| 分布式并行 | 03 ZeRO 优化器并行原理 | [PPT](./03DSZero.pdf), [视频](https://www.bilibili.com/video/BV1fb421t7KN/) |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AISys](https://chenzomi12.github.io/) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT 开源在[github](https://github.com/chenzomi12/AISystem)，欢迎取用！！！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
>
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交代码 PR 到开源社区哦！
>
> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！
