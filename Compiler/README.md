# Deep Learning System

深度学习系统（AI系统）

这个开源项目英文名字叫做 Deep Learning System 或者 AI System，中文名字叫做 深度学习系统 或者 AI系统。

主要是跟大家一起探讨和学习人工智能、深度学习的计算机系统设计，而整个系统是围绕着我在工作当中所积累、梳理、构建关于华为昇腾的内容。当然这里不是打广告，而是希望跟所有关注开源项目的好朋友一起探讨研究，共同促进学习讨论。

## 内容大纲

第二部分进进阶篇介绍AI框架**<u>底层编译技术</u>**，将站在系统设计的角度，思考在设计现代机器学习系统中需要考虑的编译器问题，特别是中间表达乃至后端优化。

### 课程部分

**[二. 底层编译技术](./Compiler/)**

|     |             |                             |                                                                                                                  |
| --- | ----------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 编号  | 名称          | 具体内容                        | 备注                                                                                                               |
|     | 编译器基础       | 01 课程概述                     | [silde](./Compiler/01.Base/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1D84y1y73v/)          |
|     |             |                             |                                                                                                                  |
| 1   | 传统编译器       | 02 开源编译器的发展                 | [silde](./Compiler/01.Base/02.history.pptx), [video](https://www.bilibili.com/video/BV1sM411C7Vr/)               |
|     | 传统编译器       | 03 GCC编译过程和原理               | [silde](./Compiler/01.Base/03.gcc.pptx), [video](https://www.bilibili.com/video/BV1LR4y1f7et/)                   |
|     | 传统编译器       | 04 LLVM设计架构                 | [silde](./Compiler/01.Base/04.llvm.pptx), [video](https://www.bilibili.com/video/BV1CG4y1V7Dn/)                  |
|     | 传统编译器       | 05(上) LLVM IR详解             | [silde](./Compiler/01.Base/05.llvm_detail01.pptx), [video](https://www.bilibili.com/video/BV1LR4y1f7et/)         |
|     | 传统编译器       | 05(中) LLVM前端和优化层            | [silde](./Compiler/01.Base/06.llvm_detail02.pptx), [video](https://www.bilibili.com/video/BV1vd4y1t7vS)          |
|     | 传统编译器       | 05(下) LLVM后端代码生成            | [silde](./Compiler/01.Base/07.llvm_detail03.pptx), [video](https://www.bilibili.com/video/BV1cd4y1b7ho)          |
|     |             |                             |                                                                                                                  |
| 2   | AI 编译器      | 01 为什么需要AI编译器               | [silde](./Compiler/02.AICompiler/01.appear.pptx), [video](https://www.bilibili.com/video/BV1pM41167KP)           |
|     | AI 编译器      | 02 AI编译器的发展阶段               | [silde](./Compiler/02.AICompiler/02.stage.pptx), [video](https://www.bilibili.com/video/BV1QK411R7iy/)           |
|     | AI 编译器      | 03 AI编译器的通用架构               | [silde](./Compiler/02.AICompiler/03.architecture.pptx), [video](https://www.bilibili.com/video/BV1qD4y1Y73e/)    |
|     | AI 编译器      | 04 AI编译器的挑战与思考              | [silde](./Compiler/02.AICompiler/04.future.pptx),  [video](https://www.bilibili.com/video/BV1Hv4y1R7uc/)         |
|     |             |                             |                                                                                                                  |
| 3   | PyTorch2.0  | 01 PyTorch2.0 特性串讲          | [silde](./Compiler/03.PyTorch/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1p84y1675B/)       |
| 3.1 | TorchDynamo | 02 TorchScript 静态图尝试        | [silde](./Compiler/03.PyTorch/02.torchscript.pptx), [video](https://www.bilibili.com/video/BV1JV4y1P7gB/)        |
|     | TorchDynamo | 03 Torch FX 与 LazyTensor 特性 | [silde](./Compiler/03.PyTorch/03.torchfx_lazy.pptx), [video](https://www.bilibili.com/video/BV1944y1m7fU/)       |
|     | TorchDynamo | 04 TorchDynamo 来啦           | [silde](./Compiler/03.PyTorch/04.torchdynamo.pptx),  [video](https://www.bilibili.com/video/BV1Hv4y1R7uc/)       |
|     |             |                             |                                                                                                                  |
| 4   | 前端优化        | 01 内容介绍                     | [silde](./Compiler/04.FrontendOpt/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1ne411w7n2/)   |
|     | 前端优化        | 02 计算图层IR                   | [silde](./Compiler/04.FrontendOpt/02.graph_ir.pptx), [video](https://www.bilibili.com/video/BV1kV4y1w72W/)       |
|     | 前端优化        | 03 算子融合策略                   | [silde](./Compiler/04.FrontendOpt/03.op_fusion.pptx), [video](https://www.bilibili.com/video/BV1P24y1D7RV/)      |
|     | 前端优化        | 04(上) 布局转换原理                | [silde](./Compiler/04.FrontendOpt/04.layout_trans01.pptx), [video](https://www.bilibili.com/video/BV1xK411z7Uw/) |
|     | 前端优化        | 04(下) 布局转换算法                | [silde](./Compiler/04.FrontendOpt/04.layout_trans02.pptx), [video](https://www.bilibili.com/video/BV1gd4y1Y7dc/) |
|     | 前端优化        | 05 内存分配算法                   | [silde](./Compiler/04.FrontendOpt/05.memory.pptx), [video]()                                                     |
|     | 前端优化        | 06 常量折叠原理                   | [silde](./Compiler/04.FrontendOpt/06.constant_fold.pptx), [video](https://www.bilibili.com/video/BV1P8411W7dY/)  |
|     | 前端优化        | 07 公共子表达式消除                 | [silde](./Compiler/04.FrontendOpt/07.cse.pptx), [video](https://www.bilibili.com/video/BV1rv4y1Q7tp/)            |
|     | 前端优化        | 08 死代码消除                    | [silde](./Compiler/04.FrontendOpt/08.dce.pptx), [video](https://www.bilibili.com/video/BV1hD4y1h7nh/)            |
|     | 前端优化        | 09 代数简化                     | [silde](./Compiler/04.FrontendOpt/09.algebraic.pptx), [video]()                                                  |
|     |             |                             |                                                                                                                  |
更新ing...

## 项目背景

近年来人工智能特别是深度学习技术得到了飞速发展，这背后离不开计算机硬件和软件系统的不断进步。在可见的未来，人工智能技术的发展仍将依赖于计算机系统和人工智能相结合的共同创新模式。需要注意的是，计算机系统现在正以更大的规模和更高的复杂性来赋能于人工智能，这背后不仅需要更多的系统上的创新，更需要系统性的思维和方法论。与此同时，人工智能也反过来为设计复杂系统提供支持。

我们注意到，现在的大部分人工智能相关的课程，特别是深度学习和机器学习相关课程主要集中在相关理论、算法或者应用，与系统相关的课程并不多见。我们希望人工智能系统这门课能让人工智能相关教育变得更加全面和深入，以共同促进人工智能与系统在开源方面的共同学习和讨论。

（原谅我复制粘贴微软[AI-System](https://github.com/microsoft/AI-System)的介绍，人家写得很好啦；另外推荐一个很好学习参考项目，公司跟英国麦络老师（爱丁堡大学）合作的[机器学习系统：设计和实现](https://github.com/openmlsys/openmlsys-zh)。）
