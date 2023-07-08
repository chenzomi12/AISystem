<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# 前端优化

随着深度学习的不断发展，AI 模型结构在快速演化，底层计算硬件技术更是层出不穷，对于广大开发者来说不仅要考虑如何在复杂多变的场景下有效的将算力发挥出来，还要应对计算框架的持续迭代。AI编译器就成了应对以上问题广受关注的技术方向，让用户仅需专注于上层模型开发，降低手工优化性能的人力开发成本，进一步压榨硬件性能空间。

AI编译器主要是分为前端优化、后端优化，部分还会有中间优化层，而这里面主要介绍AI编译器的前端优化涉及到的算法和优化Pass。

我在这里抛砖引玉，希望您可以一起参与到这个开源项目中，跟更多的您一起探讨学习！

## 内容大纲

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 前端优化 | 01 内容介绍| [PPT](./01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1ne411w7n2/) |
| 前端优化 | 02 计算图层IR| [PPT](./02.graph_ir.pdf), [视频](https://www.bilibili.com/video/BV1kV4y1w72W/) |
| 前端优化 | 03 算子融合策略| [PPT](./03.op_fusion.pdf), [视频](https://www.bilibili.com/video/BV1P24y1D7RV/) |
| 前端优化 | 04(上) 布局转换原理 | [PPT](./04.layout_trans01.pdf), [视频](https://www.bilibili.com/video/BV1xK411z7Uw/) |
| 前端优化 | 04(下) 布局转换算法 | [PPT](./04.layout_trans02.pdf), [视频](https://www.bilibili.com/video/BV1gd4y1Y7dc/) |
| 前端优化 | 05 内存分配算法| [PPT](./05.memory.pdf), [视频](https://www.bilibili.com/video/BV1nM411879s/) |
| 前端优化 | 06 常量折叠原理| [PPT](./06.constant_fold.pdf), [视频](https://www.bilibili.com/video/BV1P8411W7dY/) |
| 前端优化 | 07 公共表达式消除 | [PPT](./07.cse.pdf), [视频](https://www.bilibili.com/video/BV1rv4y1Q7tp/) |
| 前端优化 | 08 死代码消除 | [PPT](./08.dce.pdf), [视频](https://www.bilibili.com/video/BV1hD4y1h7nh/) |
| 前端优化 | 09 代数简化| [PPT](./09.algebraic.pdf), [视频](https://www.bilibili.com/video/BV1g24y1Q7qC/) |
| 前端优化 | 10 优化Pass排序| [PPT](./10.summary.pdf), [视频](https://www.bilibili.com/video/BV1L14y1P7ku/) |

```toc
:maxdepth: 2

01.introduction
02.graph_ir
03.op_fusion
04.layout_trans01
04.layout_trans02
05.memory
06.constant_fold
07.cse
08.dce
09.algebraic
10.summary
```
