<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# 传统编译器

《传统编译器》会粗略地回顾传统编译器中的前端、后端、IR中间表达等主要的概念，并对目前最常用的GCC和LLVM的发展历史，GCC的使用方式和LLVM的架构前后端优化划分，两大编译器GCC和LLVM进行简单的展开，去了解GCC的编译流程和编译方式，并回顾LLVM的整体架构，了解传统编译器的整体架构和脉络。

随着深度学习的不断发展，AI 模型结构在快速演化，底层计算硬件技术更是层出不穷，对于广大开发者来说不仅要考虑如何在复杂多变的场景下有效的将算力发挥出来，还要应对计算框架的持续迭代。深度编译器就成了应对以上问题广受关注的技术方向，让用户仅需专注于上层模型开发，降低手工优化性能的人力开发成本，进一步压榨硬件性能空间。我们先了解通用编译器的概念(1），然后通过(2)知道编译器近几十年的发展，(3)(4)(5)开始深入传统编译器的流程和原理。

我在这里抛砖引玉，希望您可以一起参与到这个开源项目中，跟更多的您一起探讨学习！

## 内容大纲

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 编译器基础 | 01 课程概述| [PPT](./01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1D84y1y73v/) |
| | |
| 传统编译器 | 02 开源编译器的发展| [PPT](./02.history.pdf), [视频](https://www.bilibili.com/video/BV1sM411C7Vr/) |
| 传统编译器 | 03 GCC编译过程和原理| [PPT](./03.gcc.pdf), [视频](https://www.bilibili.com/video/BV1LR4y1f7et/) |
| 传统编译器 | 04 LLVM设计架构| [PPT](./04.llvm.pdf), [视频](https://www.bilibili.com/video/BV1CG4y1V7Dn/) |
| 传统编译器 | 05(上) LLVM IR详解| [PPT](./05.llvm_detail01.pdf), [视频](https://www.bilibili.com/video/BV1LR4y1f7et/) |
| 传统编译器 | 05(中) LLVM前端和优化层 | [PPT](./06.llvm_detail02.pdf), [视频](https://www.bilibili.com/video/BV1vd4y1t7vS) |
| 传统编译器 | 05(下) LLVM后端代码生成 | [PPT](./07.llvm_detail03.pdf), [视频](https://www.bilibili.com/video/BV1cd4y1b7ho) |

```toc
:maxdepth: 2

01.introduction
02.history
03.gcc
04.llvm
05.llvm_detail01
06.llvm_detail02
07.llvm_detail03
```
