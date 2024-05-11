<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# 五、AI 框架核心模块

AI 系统里面，其实大部分开发者并不关心 AI 框架或者 AI 框架的前端，因为 AI 框架作为一个工具，最大的目标就是帮助更多的算法工程师快速实现他们的算法想法；另外一方面是帮助系统工程师，快速对算法进行落地部署和性能优化。

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

## 课程简介

- **《AI 框架基础》**主要是对 AI 框架的作用、发展、编程范式等散点进行汇总分享，让开发者能够知道 AI 框架与 AI 框架之间的差异和共同点，目前的 AI 框架主要的开发和编程方式。

- **《自动微分》**AI 框架会默认提供自动微分功能，避免用户手动地去对神经网络模型求导，这些复杂的工作交给 AI 框架就好了，于是自动微自然成为分作为 AI 框架的核心功能。

- **《计算图》**实际上，AI 框架主要的职责是把深度学习的表达转换称为计算机能够识别的计算图，计算图作为 AI 框架中核心的数据结构，贯穿 AI 框架的大部分整个生命周期，于是计算图对于 AI 框架的前端核心技术就显得尤为重要。

## 课程细节

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/AISystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

### 课程脑图

![AI 系统全栈](./images/Architecture.png)

### [AI 框架基础](./01Foundation/)

《AI 框架基础》主要是对 AI 框架的作用、发展、编程范式等散点进行汇总分享，让开发者能够知道 AI 框架与 AI 框架之间的差异和共同点，目前的 AI 框架主要的开发和编程方式。

| 小节 | 链接|
|:--:|:--:|
| 01 基本介绍| [文章](./01Foundation/01.introduction.md), [PPT](./01Foundation/01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1he4y1z7oD), [字幕](./01Foundation/srt/01.srt) |
| 02 AI 框架的作用| [文章](./01Foundation/02.fundamentals.md), [PPT](./01Foundation/02.fundamentals.pdf), [视频](https://www.bilibili.com/video/BV1fd4y1q7qk), [字幕](./01Foundation/srt/02.srt) |
| 03 AI 框架之争（框架发展）| [文章](./01Foundation/03.history.md), [PPT](./01Foundation/03.history.pdf), [视频](https://www.bilibili.com/video/BV1C8411x7Kn), [字幕](./01Foundation/srt/03.srt) |
| 04 编程范式（声明式&命令式） | [文章](./01Foundation/04.programing.md), [PPT](./01Foundation/04.programing.pdf), [视频](https://www.bilibili.com/video/BV1gR4y1o7WT), [字幕](./01Foundation/srt/04.srt) |

### [自动微分](./02AutoDiff/)

《自动微分》AI 框架会默认提供自动微分功能，避免用户手动地去对神经网络模型求导，这些复杂的工作交给 AI 框架就好了，于是自动微自然成为分作为 AI 框架的核心功能。

| 小节 | 链接|
|:--:|:--:|
| 01 基本介绍| [文章](./02AutoDiff/01.introduction.md), [PPT](./02AutoDiff/01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1FV4y1T7zp/), [字幕](./02AutoDiff/srt/01.srt) |
| 02 什么是微分 | [文章](./02AutoDiff/02.base_concept.md), [PPT](./02AutoDiff/02.base_concept.pdf), [视频](https://www.bilibili.com/video/BV1Ld4y1M7GJ/), [字幕](./02AutoDiff/srt/02.srt) |
| 03 正反向计算模式 | [文章](./02AutoDiff/03.grad_mode.md), [PPT](./02AutoDiff/03.grad_mode.pdf), [视频](https://www.bilibili.com/video/BV1zD4y117bL/), [字幕](./02AutoDiff/srt/03.srt) |
| 04 三种实现方法| [文章](./02AutoDiff/04.implement.md), [PPT](./02AutoDiff/04.implement.pdf), [视频](https://www.bilibili.com/video/BV1BN4y1P76t/), [字幕](./02AutoDiff/srt/04.srt) |
| 05 手把手实现正向微分框架 | [文章](./02AutoDiff/05.forward_mode.md), [视频](https://www.bilibili.com/video/BV1Ne4y1p7WU/), [字幕](./02AutoDiff/srt/05.srt) |
| 06 亲自实现一个 PyTorch | [文章](./02AutoDiff/06.reversed_mode.md), [视频](https://www.bilibili.com/video/BV1ae4y1z7E6/), [字幕](./02AutoDiff/srt/06.srt) |
| 07 自动微分的挑战&未来| [文章](./02AutoDiff/07.challenge.md), [PPT](./02AutoDiff/07.challenge.pdf), [视频](https://www.bilibili.com/video/BV17e4y1z73W/), [字幕](./02AutoDiff/srt/07.srt) |


### [计算图](./03DataFlow/)

《计算图》实际上，AI 框架主要的职责是把深度学习的表达转换称为计算机能够识别的计算图，计算图作为 AI 框架中核心的数据结构，贯穿 AI 框架的大部分整个生命周期，于是计算图对于 AI 框架的前端核心技术就显得尤为重要。

| 小节 | 链接|
|:--:|:--:|
| 01 基本介绍 | [文章](./03DataFlow/01.introduction.md), [PPT](./03DataFlow/01.introduction.pptx), [视频](https://www.bilibili.com/video/BV1cG411E7gV/), [字幕](./03DataFlow/srt/01.srt) |
| 02 什么是计算图 | [文章](./03DataFlow/02.computegraph.md), [PPT](./03DataFlow/02.computegraph.pptx), [视频](https://www.bilibili.com/video/BV1rR4y197HM/), [字幕](./03DataFlow/srt/02.srt) |
| 03 与自动微分关系 | [文章](./03DataFlow/03.atuodiff.md), [PPT](./03DataFlow/03.atuodiff.pptx), [视频](https://www.bilibili.com/video/BV1S24y197FU/), [字幕](./03DataFlow/srt/03.srt) |
| 04 图优化与图执行调度| [文章](./03DataFlow/04.dispatch.md), [PPT](./03DataFlow/04.dispatch.pptx), [视频](https://www.bilibili.com/video/BV1hD4y1k7Ty/), [字幕](./03DataFlow/srt/04.srt) |
| 05 计算图控制流实现| [文章](./03DataFlow/05.control_flow.md), [PPT](。./023FW_DataFlow/05.control_flow.pptx), [视频](https://www.bilibili.com/video/BV17P41177Pk/), [字幕](./03DataFlow/srt/05.srt) |
| 06 计算图实现动静统一| [文章](./03DataFlow/06.static_graph.md), [PPT](./03DataFlow/06.static_graph.pdf), [视频](https://www.bilibili.com/video/BV17P41177Pk/), [字幕](。./srt/06.srt) |
| 07 计算图的挑战与未来 |[文章](./03DataFlow/07.future.md), [PPT](./03DataFlow/07.future.pdf), [视频](https://www.bilibili.com/video/BV1hm4y1A7Nv/), [字幕](.。/srt/07.srt) |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AISys](https://chenzomi12.github.io/) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT 开源在[github](https://github.com/chenzomi12/AISystem)，欢迎取用！！！

> 非常希望您也参与到这个开源项目中，B 站给 ZOMI 留言哦！
>
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交代码 PR 到开源社区哦！
>
> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！
