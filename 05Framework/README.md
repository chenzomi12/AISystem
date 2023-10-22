<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# === 五、AI框架核心模块 ===

AI系统里面，其实大部分开发者并不关心AI框架或者AI框架的前端，因为AI框架作为一个工具，最大的目标就是帮助更多的算法工程师快速实现他们的算法想法；另外一方面是帮助系统工程师，快速对算法进行落地部署和性能优化。

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

## 课程简介

- **《AI框架基础》**主要是对AI框架的作用、发展、编程范式等散点进行汇总分享，让开发者能够知道AI框架与AI框架之间的差异和共同点，目前的AI框架主要的开发和编程方式。

- **《自动微分》**AI框架会默认提供自动微分功能，避免用户手动地去对神经网络模型求导，这些复杂的工作交给AI框架就好了，于是自动微自然成为分作为AI框架的核心功能。

- **《计算图》**实际上，AI框架主要的职责是把深度学习的表达转换称为计算机能够识别的计算图，计算图作为AI框架中核心的数据结构，贯穿AI框架的大部分整个生命周期，于是计算图对于AI框架的前端核心技术就显得尤为重要。

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

| 小节 | 链接|
|:--:|:--:|
| 01 基本介绍| [文章](../022FW_AutoDiff/01.introduction.md), [PPT](../022FW_AutoDiff/01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1FV4y1T7zp/), [字幕](../022FW_AutoDiff/srt/01.srt) |
| 02 什么是微分 | [文章](../022FW_AutoDiff/02.base_concept.md), [PPT](../022FW_AutoDiff/02.base_concept.pdf), [视频](https://www.bilibili.com/video/BV1Ld4y1M7GJ/), [字幕](../022FW_AutoDiff/srt/02.srt) |
| 03 正反向计算模式 | [文章](../022FW_AutoDiff/03.grad_mode.md), [PPT](../022FW_AutoDiff/03.grad_mode.pdf), [视频](https://www.bilibili.com/video/BV1zD4y117bL/), [字幕](../022FW_AutoDiff/srt/03.srt) |
| 04 三种实现方法| [文章](../022FW_AutoDiff/04.implement.md), [PPT](../022FW_AutoDiff/04.implement.pdf), [视频](https://www.bilibili.com/video/BV1BN4y1P76t/), [字幕](../022FW_AutoDiff/srt/04.srt) |
| 05 手把手实现正向微分框架 | [文章](../022FW_AutoDiff/05.forward_mode.md), [视频](https://www.bilibili.com/video/BV1Ne4y1p7WU/), [字幕](../022FW_AutoDiff/srt/05.srt) |
| 06 亲自实现一个PyTorch | [文章](../022FW_AutoDiff/06.reversed_mode.md), [视频](https://www.bilibili.com/video/BV1ae4y1z7E6/), [字幕](../022FW_AutoDiff/srt/06.srt) |
| 07 自动微分的挑战&未来| [文章](../022FW_AutoDiff/07.challenge.md), [PPT](../022FW_AutoDiff/07.challenge.pdf), [视频](https://www.bilibili.com/video/BV17e4y1z73W/), [字幕](../022FW_AutoDiff/srt/07.srt) |


### 计算图

《计算图》实际上，AI框架主要的职责是把深度学习的表达转换称为计算机能够识别的计算图，计算图作为AI框架中核心的数据结构，贯穿AI框架的大部分整个生命周期，于是计算图对于AI框架的前端核心技术就显得尤为重要。

| 小节 | 链接|
|:--:|:--:|
| 01 基本介绍 | [文章](../023FW_DataFlow/01.introduction.md), [PPT](../023FW_DataFlow/01.introduction.pptx), [视频](https://www.bilibili.com/video/BV1cG411E7gV/), [字幕](../023FW_DataFlow/srt/01.srt) |
| 02 什么是计算图 | [文章](../023FW_DataFlow/02.computegraph.md), [PPT](../023FW_DataFlow/02.computegraph.pptx), [视频](https://www.bilibili.com/video/BV1rR4y197HM/), [字幕](../023FW_DataFlow/srt/02.srt) |
| 03 与自动微分关系 | [文章](../023FW_DataFlow/03.atuodiff.md), [PPT](../023FW_DataFlow/03.atuodiff.pptx), [视频](https://www.bilibili.com/video/BV1S24y197FU/), [字幕](../023FW_DataFlow/srt/03.srt) |
| 04 图优化与图执行调度| [文章](../023FW_DataFlow/04.dispatch.md), [PPT](../023FW_DataFlow/04.dispatch.pptx), [视频](https://www.bilibili.com/video/BV1hD4y1k7Ty/), [字幕](../023FW_DataFlow/srt/04.srt) |
| 05 计算图控制流实现| [文章](../023FW_DataFlow/05.control_flow.md), [PPT](。./023FW_DataFlow/05.control_flow.pptx), [视频](https://www.bilibili.com/video/BV17P41177Pk/), [字幕](../023FW_DataFlow/srt/05.srt) |
| 06 计算图实现动静统一| [文章](../023FW_DataFlow/06.static_graph.md), [PPT](../023FW_DataFlow/06.static_graph.pdf), [视频](https://www.bilibili.com/video/BV17P41177Pk/), [字幕](。./srt/06.srt) |
| 07 计算图的挑战与未来 |[文章](../023FW_DataFlow/07.future.md), [PPT](../023FW_DataFlow/07.future.pdf), [视频](https://www.bilibili.com/video/BV1hm4y1A7Nv/), [字幕](.。/srt/07.srt) |
