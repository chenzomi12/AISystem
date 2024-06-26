<!--Copyright © 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# 计算图

为了高效地训练一个复杂神经网络，框架需要解决诸多问题，例如：如何实现自动求导，如何利用编译期分析对神经网络计算进行化简、合并、变换，如何规划基本计算单元在加速器上的执行，如何将基本处理单元派发（dispatch）到特定的高效后端实现，如何进行内存预分配和管理等。用统一的方式解决这些问题都驱使着框架设计者思考如何为各类神经网络计算提供统一的描述，从而使得在运行神经网络计算之前，能够对整个计算过程尽可能进行推断，在编译期自动为用户程序补全反向计算，规划执行，最大程度地降低运行时开销。

目前主流的 AI 框架都选择使用计算图来抽象神经网络计算，《计算图》实际上，AI 框架主要的职责是把深度学习的表达转换称为计算机能够识别的计算图，计算图作为 AI 框架中核心的数据结构，贯穿 AI 框架的大部分整个生命周期，于是计算图对于 AI 框架的前端核心技术就显得尤为重要。

## 内容大纲

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/AISystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 小节 | 链接|
|:--:|:--:|
| 01 基本介绍 | [文章](./01Introduction.md), [PPT](./01Introduction.pdf), [视频](https://www.bilibili.com/video/BV1cG411E7gV/) |
| 02 什么是计算图 | [文章](./02Computegraph.md), [PPT](./02Computegraph.pdf), [视频](https://www.bilibili.com/video/BV1rR4y197HM/) |
| 03 与自动微分关系 | [文章](./03Atuodiff.md), [PPT](./03Atuodiff.pdf), [视频](https://www.bilibili.com/video/BV1S24y197FU/)|
| 04 图优化与图执行调度| [文章](./04Dispatch.md), [PPT](./04Dispatch.pdf), [视频](https://www.bilibili.com/video/BV1hD4y1k7Ty/) |
| 05 计算图控制流实现| [文章](./05ControlFlow.md), [PPT](./05ControlFlow.pdf), [视频](https://www.bilibili.com/video/BV17P41177Pk/)|
| 06 计算图实现动静统一| [文章](./06StaticGraph.md), [PPT](./06StaticGraph.pdf), [视频](https://www.bilibili.com/video/BV17P41177Pk/)|
| 07 计算图的挑战与未来 |[文章](./07Future.md), [PPT](./07Future.pdf), [视频](https://www.bilibili.com/video/BV1hm4y1A7Nv/) |

## 备注

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT 开源在[github](https://github.com/chenzomi12/AISystem)，欢迎取用！！！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
> 
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交代码 PR 到开源社区哦！
>
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交 PR 到开源社区哦！
>
> 请大家尊重开源和 ZOMI 的努力，引用 PPT 的内容请规范转载标明出处哦！