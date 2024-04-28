<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# 传统编译器(DOING)

《传统编译器》会粗略地回顾传统编译器中的前端、后端、IR 中间表达等主要的概念，并对目前最常用的 GCC 和 LLVM 的发展历史，GCC 的使用方式和 LLVM 的架构前后端优化划分，两大编译器 GCC 和 LLVM 进行简单的展开，去了解 GCC 的编译流程和编译方式，并回顾 LLVM 的整体架构，了解传统编译器的整体架构和脉络。

随着深度学习的不断发展，AI 模型结构在快速演化，底层计算硬件技术更是层出不穷，对于广大开发者来说不仅要考虑如何在复杂多变的场景下有效的将算力发挥出来，还要应对计算框架的持续迭代。AI 编译器就成了应对以上问题广受关注的技术方向，让用户仅需专注于上层模型开发，降低手工优化性能的人力开发成本，进一步压榨硬件性能空间。我们先了解通用编译器的概念，然后通过编译器的历史回顾知道编译器近几十年的发展，最后开始深入传统编译器的流程和原理。

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

**内容大纲**

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 小节 | 链接|
|:--:|:--:|
| 01 编译器基础概念 | [PPT](./01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1D84y1y73v/), [文章](./01.introduction.md), [字幕](./srt/01.srt) |
|  |  |
| 02 开源编译器的发展 | [PPT](./02.history.pdf), [视频](https://www.bilibili.com/video/BV1sM411C7Vr/), [文章](./02.history.md), [字幕](./srt/02.srt) |
| 03 GCC 编译过程和原理 | [PPT](./03.gcc.pdf), [视频](https://www.bilibili.com/video/BV1LR4y1f7et/), [文章](./03.gcc.md), [字幕](./srt/03.srt) |
| 04 LLVM 设计架构 | [PPT](./04.llvm.pdf), [视频](https://www.bilibili.com/video/BV1CG4y1V7Dn/), [文章](./04.llvm.md), [字幕](./srt/04.srt) |
| 05 LLVM IR 详解 | [PPT](./05.llvm_detail01.pdf), [视频](https://www.bilibili.com/video/BV1LR4y1f7et/), [文章](./05.llvm_detail01.md), [字幕](./srt/05.srt) |
| 06 LLVM 前端和优化层 | [PPT](./06.llvm_detail02.pdf), [视频](https://www.bilibili.com/video/BV1vd4y1t7vS), [文章](./06.llvm_detail02.md), [字幕](./srt/06.srt) |
| 07 LLVM 后端代码生成 | [PPT](./07.llvm_detail03.pdf), [视频](https://www.bilibili.com/video/BV1cd4y1b7ho), [文章](./07.llvm_detail03.md), [字幕](./srt/07.srt) |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AISys](https://chenzomi12.github.io/) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT 开源在[github](https://github.com/chenzomi12/DeepLearningSystem)，欢迎取用！！！

> 非常希望您也参与到这个开源项目中，B 站给 ZOMI 留言哦！
>
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交代码 PR 到开源社区哦！
>
> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！
