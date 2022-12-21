# AI编译器之传统编译器

随着深度学习的不断发展，AI 模型结构在快速演化，底层计算硬件技术更是层出不穷，对于广大开发者来说不仅要考虑如何在复杂多变的场景下有效的将算力发挥出来，还要应对计算框架的持续迭代。深度编译器就成了应对以上问题广受关注的技术方向，让用户仅需专注于上层模型开发，降低手工优化性能的人力开发成本，进一步压榨硬件性能空间。我们先了解通用编译器的概念(1），然后通过(2)知道编译器近几十年的发展，(3)(4)(5)开始深入传统编译器的流程和原理。最后从(6)深入了解下近年来连Chris都投身的AI编译器。

## 内容大纲

| 编号  | 名称    | 名称               | 备注                                                                                      |
| --- | ----- | ---------------- | --------------------------------------------------------------------------------------- |
|     |       |                  |                                                                                         |
|     | 编译器基础 | 01 课程概述          | [silde](./01.introduction.pptx), [video](https://www.bilibili.com/video/BV1D84y1y73v/)  |
|     |       |                  |                                                                                         |
| 1   | 传统编译器 | 02 开源编译器的发展      | [silde](./02.history.pptx), [video](https://www.bilibili.com/video/BV1sM411C7Vr/)       |
|     | 传统编译器 | 03 GCC编译过程和原理    | [silde](./03.gcc.pptx), [video](https://www.bilibili.com/video/BV1LR4y1f7et/)           |
|     | 传统编译器 | 04 LLVM设计架构      | [silde](./04.llvm.pptx), [video](https://www.bilibili.com/video/BV1CG4y1V7Dn/)          |
|     | 传统编译器 | 05(上) LLVM IR详解  | [silde](./05.llvm_detail01.pptx), [video](https://www.bilibili.com/video/BV1LR4y1f7et/) |
|     | 传统编译器 | 05(中) LLVM前端和优化层 | [silde](./06.llvm_detail02.pptx), [video](https://www.bilibili.com/video/BV1vd4y1t7vS)  |
|     | 传统编译器 | 05(下) LLVM后端代码生成 | [silde](./07.llvm_detail03.pptx), [video](https://www.bilibili.com/video/BV1cd4y1b7ho)  |
|     |       |                  |                                                                                         |
