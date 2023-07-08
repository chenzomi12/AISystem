<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# 计算图

为了高效地训练一个复杂神经网络，框架需要解决诸多问题， 例如：如何实现自动求导，如何利用编译期分析对神经网络计算进行化简、合并、变换，如何规划基本计算单元在加速器上的执行，如何将基本处理单元派发（dispatch）到特定的高效后端实现，如何进行内存预分配和管理等。用统一的方式解决这些问题都驱使着框架设计者思考如何为各类神经网络计算提供统一的描述，从而使得在运行神经网络计算之前，能够对整个计算过程尽可能进行推断，在编译期自动为用户程序补全反向计算，规划执行，最大程度地降低运行时开销。

目前主流的深度学习框架都选择使用计算图来抽象神经网络计算，《计算图》实际上，AI框架主要的职责是把深度学习的表达转换称为计算机能够识别的计算图，计算图作为AI框架中核心的数据结构，贯穿AI框架的大部分整个生命周期，于是计算图对于AI框架的前端核心技术就显得尤为重要。

我在这里抛砖引玉，希望您可以一起参与到这个开源项目中，跟更多的您一起探讨学习！

## 内容大纲

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 计算图 | 01 基本介绍 | [PPT](./01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1cG411E7gV/) |
| 计算图 | 02 什么是计算图 | [PPT](./02.computation_graph.pdf), [视频](https://www.bilibili.com/video/BV1rR4y197HM/) |
| 计算图 | 03 计算图跟自动微分关系 | [PPT](./03.atuodiff.pdf), [视频](https://www.bilibili.com/video/BV1S24y197FU/) |
| 计算图 | 04 图优化与图执行调度| [PPT](./04.dispatch.pdf), [视频](https://www.bilibili.com/video/BV1hD4y1k7Ty/) |
| 计算图 | 05 计算图的控制流机制实现| [PPT](./05.control_flow.pdf), [视频](https://www.bilibili.com/video/BV17P41177Pk/) |
| 计算图 | 06 计算图未来将会走向何方？ | [PPT](./06.future.pdf), [视频](https://www.bilibili.com/video/BV1hm4y1A7Nv/) |

```toc
:maxdepth: 2

01.introduction
02.computation_graph
03.atuodiff
04.dispatch
05.control_flow
06.future
```