# AI编译器原理介绍

随着深度学习的应用场景的不断泛化，深度学习计算任务也需要部署在不同的计算设备和硬件架构上；同时，实际部署或训练场景对性能往往也有着更为激进的要求，例如针对硬件特点定制计算代码。

这些需求在通用的AI框架中已经难已得到满足。由于深度学习计算任务在现有的AI框架中往往以DSL（Domain Specific Language）的方式进行编程和表达，这本身使得深度学习计算任务的优化和执行天然符合传统计算机语言的编译和优化过程。因此，深度学习的编译与优化就是将当前的深度学习计算任务通过一层或多层中间表达进行翻译和优化，最终转化成目标硬件上的可执行代码的过程。本章将围绕现有AI编译器中的编译和优化工作的内容展开介绍。

## 课程简介

- 《传统编译器》会粗略地回顾传统编译器中的前端、后端、IR中间表达等主要的概念，并对目前主流的两大编译器GCC和LLVM进行简单的展开，去了解GCC的编译流程和编译方式，并回顾LLVM的整体架构。

- 《AI 编译器》是本节的概览重点，去了解本章的主要内容 AI 编译器的整体架构，包括他的发展阶段，目前主要的组成模块，整体的技术演进方向等概念性的内容，因为近年来AI编译器发展迅猛，可以横向去了解AI编译器整体技术。

- 《前端优化》前端优化作为 AI编译器 的整体架构主要模块，主要优化的对象是计算图，而计算图是通过AI框架产生的，值得注意的是并不是所有的AI框架都会生成计算图，有了计算图就可以结合深度学习的原理知识进行图的优化。

- 《后端优化》后端优化作为AI编译器跟硬件之间的相连接的模块，更多的是算子或者Kernel进行优化，而优化之前需要把计算图转换称为调度树等IR格式，然后针对每一个算子/Kernel进行循环优化、指令优化和内存优化等技术。

- 《多面体技术》多面体不属于新的技术，反而是传统编译器的一种优化手段，得益于深度学习中的主要特征（循环、张量），因此多面体技术可以发挥更大的作用，对循环展开、内存映射等优化工作。

- 《PyTorch图模式》会以实际的AI框架 PyTorch 2.0为主线，去把其主打特性 Dynamo 和 AOTAutograd 进行展开，并回顾 PyTorch 对图模式的尝试，了解现今最热门的AI框架如何进行编译器优化的。

希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

## 课程目标

通过《AI编译器原理介绍》这个课程，以及这门课程后面的几门课程，你将获取并且掌握的技能：

| 编号  | 名称                       | 具体内容                                                                                |
|:---:|:------------------------ |:----------------------------------------------------------------------------------- |
| 1   | [传统编译器](./Tradition/)    | 目前最常用的GCC和LLVM的发展历史，GCC的使用方式和LLVM的架构前后端优化划分，去了解传统编译器的整体架构和脉络。                       |
| 2   | [AI 编译器](./AICompiler/)  | AI 编译器发展时间并不长，从TF第一代开始到目前进入第二代，AI编译器整体架构基本上已经清晰，可是也会越到很多挑战和技术难点。                    |
| 3   | [前端优化](./Frontend/)      | AI 编译器的前端优化包括图算融合、数据排布、内存优化等跟深度学习相结合的优化手段，同时把传统编译器关于代数优化的技术引入到计算图中。                 |
| 4   | [后端优化](./Backend/)       | AI 编译器为了更好地跟硬件打交道，充分赋能硬件，需要后端优化来支持，于是后端针对调度树或者底层IR回进行循环优化、指令优化和内存优化等优化手段。           |
| 5   | [多面体技术](./AICluster)     | 多面体表示技术作为统一化的程序变换表示技术, 可以通过迭代域、仿射调度、访存函数等操作对算子或者Kernel进行循环优化和内存映射优化，作为AI编译器的前言研究方向。 |
| 6   | [PyTorch图模式](./PyTorch/) | 在充分了解AI编译器后，来深度剖析PyTorch2.0关于图模式的Dynamo是如何实现的，如何对PyTorch的后端执行进行加速。                  |
|     |                          |                                                                                     |

## 课程细节

|     |             |                             |                                                                                                  |
| --- | ----------- | --------------------------- | ------------------------------------------------------------------------------------------------ |
| 编号  | 名称          | 具体内容                        | 备注                                                                                               |
|     | 编译器基础       | 01 课程概述                     | [silde](./Tradition/01.introduction.pdf), [video](https://www.bilibili.com/video/BV1D84y1y73v/)  |
|     |             |                             |                                                                                                  |
| 1   | 传统编译器       | 02 开源编译器的发展                 | [silde](./Tradition/02.history.pdf), [video](https://www.bilibili.com/video/BV1sM411C7Vr/)       |
|     | 传统编译器       | 03 GCC编译过程和原理               | [silde](./Tradition/03.gcc.pdf), [video](https://www.bilibili.com/video/BV1LR4y1f7et/)           |
|     | 传统编译器       | 04 LLVM设计架构                 | [silde](./Tradition/04.llvm.pdf), [video](https://www.bilibili.com/video/BV1CG4y1V7Dn/)          |
|     | 传统编译器       | 05(上) LLVM IR详解             | [silde](./Tradition/05.llvm_detail01.pdf), [video](https://www.bilibili.com/video/BV1LR4y1f7et/) |
|     | 传统编译器       | 05(中) LLVM前端和优化层            | [silde](./Tradition/06.llvm_detail02.pdf), [video](https://www.bilibili.com/video/BV1vd4y1t7vS)  |
|     | 传统编译器       | 05(下) LLVM后端代码生成            | [silde](./Tradition/07.llvm_detail03.pdf), [video](https://www.bilibili.com/video/BV1cd4y1b7ho)  |
|     |             |                             |                                                                                                  |
| 2   | AI 编译器      | 01 为什么需要AI编译器               | [silde](./AICompiler/01.appear.pdf), [video](https://www.bilibili.com/video/BV1pM41167KP)        |
|     | AI 编译器      | 02 AI编译器的发展阶段               | [silde](./AICompiler/02.stage.pdf), [video](https://www.bilibili.com/video/BV1QK411R7iy/)        |
|     | AI 编译器      | 03 AI编译器的通用架构               | [silde](./AICompiler/03.architecture.pdf), [video](https://www.bilibili.com/video/BV1qD4y1Y73e/) |
|     | AI 编译器      | 04 AI编译器的挑战与思考              | [silde](./AICompiler/04.future.pdf),  [video](https://www.bilibili.com/video/BV1Hv4y1R7uc/)      |
|     |             |                             |                                                                                                  |
| 3   | 前端优化        | 01 内容介绍                     | [silde](./Frontend/01.introduction.pdf), [video](https://www.bilibili.com/video/BV1ne411w7n2/)   |
|     | 前端优化        | 02 计算图层IR                   | [silde](./Frontend/02.graph_ir.pdf), [video](https://www.bilibili.com/video/BV1kV4y1w72W/)       |
|     | 前端优化        | 03 算子融合策略                   | [silde](./Frontend/03.op_fusion.pdf), [video](https://www.bilibili.com/video/BV1P24y1D7RV/)      |
|     | 前端优化        | 04(上) 布局转换原理                | [silde](./Frontend/04.layout_trans01.pdf), [video](https://www.bilibili.com/video/BV1xK411z7Uw/) |
|     | 前端优化        | 04(下) 布局转换算法                | [silde](./Frontend/04.layout_trans02.pdf), [video](https://www.bilibili.com/video/BV1gd4y1Y7dc/) |
|     | 前端优化        | 05 内存分配算法                   | [silde](./Frontend/05.memory.pdf), [video]()                                                     |
|     | 前端优化        | 06 常量折叠原理                   | [silde](./Frontend/06.constant_fold.pdf), [video](https://www.bilibili.com/video/BV1P8411W7dY/)  |
|     | 前端优化        | 07 公共表达式消除                  | [silde](./Frontend/07.cse.pdf), [video](https://www.bilibili.com/video/BV1rv4y1Q7tp/)            |
|     | 前端优化        | 08 死代码消除                    | [silde](./Frontend/08.dce.pdf), [video](https://www.bilibili.com/video/BV1hD4y1h7nh/)            |
|     | 前端优化        | 09 代数简化原理                   | [silde](./Frontend/09.algebraic.pdf), [video](https://www.bilibili.com/video/BV1g24y1Q7qC/)      |
|     | 前端优化        | 10 优化Pass排序                 | [silde](./Frontend/10.summary.pdf), [video](https://www.bilibili.com/video/BV1L14y1P7ku/)        |
|     |             |                             |                                                                                                  |
| 4   | 后端优化        | 01 AI编译器后端优化介绍              | [silde](./Backend/01.introduction.pdf), [video](https://www.bilibili.com/video/BV17D4y177bP/)    |
|     | 后端优化        | 02 算子分为计算与调度                | [silde](./Backend/02.ops_compute.pdf), [video](https://www.bilibili.com/video/BV1K84y1x7Be/)     |
|     | 后端优化        | 03 算子优化手工方式                 | [silde](./Backend/03.optimization.pdf), [video](https://www.bilibili.com/video/BV1ZA411X7WZ/)    |
|     | 后端优化        | 04 算子循环优化                   | [silde](./Backend/04.loop_opt.pdf), [video](https://www.bilibili.com/video/BV17D4y177bP/)        |
|     | 后端优化        | 05 指令和内存优化                  | [silde](./Backend/05.other_opt.pdf), [video](https://www.bilibili.com/video/BV11d4y1a7J6/)       |
|     | 后端优化        | 06 Auto-Tuning原理            | [silde](./Backend/06.auto_tuning.pdf), [video](https://www.bilibili.com/video/BV1uA411D7JF/)     |
|     |             |                             |                                                                                                  |
| 5   | PyTorch2.0  | 01 PyTorch2.0 特性串讲          | [silde](./PyTorch/01.introduction.pdf), [video](https://www.bilibili.com/video/BV1p84y1675B/)    |
| 5.1 | TorchDynamo | 02 TorchScript 静态图尝试        | [silde](./PyTorch/02.torchscript.pdf), [video](https://www.bilibili.com/video/BV1JV4y1P7gB/)     |
|     | TorchDynamo | 03 Torch FX 与 LazyTensor 特性 | [silde](./PyTorch/03.torchfx_lazy.pdf), [video](https://www.bilibili.com/video/BV1944y1m7fU/)    |
|     | TorchDynamo | 04 TorchDynamo 来啦           | [silde](./PyTorch/04.torchdynamo.pdf),  [video](https://www.bilibili.com/video/BV1Hv4y1R7uc/)    |
| 5.2 | AOTAutograd | 05 AOTAutograd 原理           | [silde](./PyTorch/05.aotatuograd.pdf),  [video](https://www.bilibili.com/video/BV1Me4y1V7Ke/)    |
|     | AOTAutograd | 06 Dispatch 机制              |                                                                                                  |

## 目标学员

1. 对人工智能、深度学习感兴趣的学员

2. 渴望学习当今最热门最前沿AI技术的学员

3. 想储备深度学习技能的学员

4. AI系统开发工程师
