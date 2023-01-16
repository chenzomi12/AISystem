# AI编译器 -- 前端优化

随着深度学习的不断发展，AI 模型结构在快速演化，底层计算硬件技术更是层出不穷，对于广大开发者来说不仅要考虑如何在复杂多变的场景下有效的将算力发挥出来，还要应对计算框架的持续迭代。AI编译器就成了应对以上问题广受关注的技术方向，让用户仅需专注于上层模型开发，降低手工优化性能的人力开发成本，进一步压榨硬件性能空间。

AI编译器主要是分为前端优化、后端优化，部分还会有中间优化层，而这里面主要介绍AI编译器的前端优化涉及到的算法和优化Pass。

## 内容大纲

| 名称   | 名称           | 备注                                                                                      |
| ---- | ------------ | --------------------------------------------------------------------------------------- |
|      |              |                                                                                         |
| 前端优化 | 01 内容介绍      | [silde](./01.introduction.pdf), [video](https://www.bilibili.com/video/BV1ne411w7n2/)   |
| 前端优化 | 02 计算图层IR    | [silde](./02.graph_ir.pdf), [video](https://www.bilibili.com/video/BV1kV4y1w72W/)       |
| 前端优化 | 03 算子融合策略    | [silde](./03.op_fusion.pdf), [video](https://www.bilibili.com/video/BV1P24y1D7RV/)      |
| 前端优化 | 04(上) 布局转换原理 | [silde](./04.layout_trans01.pdf), [video](https://www.bilibili.com/video/BV1xK411z7Uw/) |
| 前端优化 | 04(下) 布局转换算法 | [silde](./04.layout_trans02.pdf), [video](https://www.bilibili.com/video/BV1gd4y1Y7dc/) |
| 前端优化 | 05 内存分配算法    | [silde](./05.memory.pdf), [video](https://www.bilibili.com/video/BV1nM411879s/)         |
| 前端优化 | 06 常量折叠原理    | [silde](./06.constant_fold.pdf), [video](https://www.bilibili.com/video/BV1P8411W7dY/)  |
| 前端优化 | 07 公共表达式消除   | [silde](./07.cse.pdf), [video](https://www.bilibili.com/video/BV1rv4y1Q7tp/)            |
| 前端优化 | 08 死代码消除     | [silde](./08.dce.pdf), [video](https://www.bilibili.com/video/BV1hD4y1h7nh/)            |
| 前端优化 | 09 代数简化      | [silde](./09.algebraic.pdf), [video](https://www.bilibili.com/video/BV1g24y1Q7qC/)      |
| 前端优化 | 10 优化Pass排序  | [silde](./10.summary.pdf), [video](https://www.bilibili.com/video/BV1L14y1P7ku/)        |
