# Deep Learning System

这个开源项目英文名字叫做 **Deep Learning System** 或者 **AI System(AISys)**、**AI Infra**、**ML System(MLSys)**，中文名字叫做 **深度学习系统** 或者 **AI系统**。

主要是跟大家一起探讨和学习人工智能、深度学习的计算机系统设计，而整个系统是围绕着 ZOMI 在工作当中所积累、梳理、构建 AI 系统全栈的内容。希望跟所有关注 AI 开源项目的好朋友一起探讨研究，共同促进学习讨论。

> 欢迎大家使用的过程中发现bug或者勘误直接提交PR到开源社区哦！
> 
> 请大家尊重开源和ZOMI付出，引用PPT的内容请规范转载标明出处哦！

## 项目背景

近年来人工智能特别是深度学习技术得到了飞速发展，这背后离不开计算机硬件和软件系统的不断进步。在可见的未来，人工智能技术的发展仍将依赖于计算机系统和人工智能相结合的共同创新模式。需要注意的是，计算机系统现在正以更大的规模和更高的复杂性来赋能于人工智能，这背后不仅需要更多的系统上的创新，更需要系统性的思维和方法论。与此同时，人工智能也反过来为设计复杂系统提供支持。

我们注意到，现在的大部分人工智能相关的课程，特别是深度学习和机器学习相关课程主要集中在相关理论、算法或者应用，与系统相关的课程并不多见。我们希望人工智能系统这门课能让人工智能相关教育变得更加全面和深入，以共同促进人工智能与系统在开源方面的共同学习和讨论。

## 课程内容大纲

课程主要包括以下五大模块：

第一部分，AI基础知识和AI系统的全栈概述的<u><strong>AI系统概述</strong></u>，以及深度学习系统的系统性设计和方法论，主要是整体了解AI训练和推理全栈的体系结构内容。

第二部分，基础篇介绍AI框架的<u><strong>AI框架核心技术</strong></u>，首先介绍任何一个AI框架都离不开的自动微分，通过自动微分功能后就会产生表示神经网络的图和算子，然后介绍AI框架前端的优化，还有最近很火的大模型分布式训练在AI框架中的关键技术。

第三部分，进进阶篇介绍AI框架<u><strong>AI编译器原理</strong></u>，将站在系统设计的角度，思考在设计现代机器学习系统中需要考虑的编译器问题，特别是中间表达乃至后端优化。

第四部分，是很实际的<u><strong>推理系统</strong></u>，讲了太多原理身体太虚容易消化不良，还是得回归到业务本质，让行业、企业能够真正应用起来，而推理系统涉及一些核心算法和注意的事情也分享下。

第五部分，硬核篇介绍<u><strong>AI芯片</strong></u>，这里就很硬核了，从芯片基础到AI芯片的范围都会涉及，芯片设计需要考虑上面AI框架的前端、后端编译，而不是停留在天天喊着吊打英伟达，被现实打趴。

## 课程设立目的

本课程主要为本科生高年级、硕博研究生、AI系统从业者设计，帮助大家：

1. 完整的了解支持AI的计算机系统架构，并通过实际问题和案例，来了解AI完整生命周期下的系统设计。

2. 介绍前沿的工程系统和AI相结合的研究工作，包括AI for Systems和Systems for AI（即AISys），以帮助大家更好的寻找有意义的研究课题。

3. 从AI系统研究的角度出发设计全栈课程。通过了解主流和最新框架、平台和工具来了解AI系统，以提高解决实际问题能力。

**先修课程:** C++/Python，计算机体系结构，算法导论，人工智能基础

### 课程详细内容

### 一. AI系统概述

1.0 AI系统概述前言 [link](./第1章-AI系统概述/1.0-AI系统概述前言.md)

1.1 AI历史、现状与发展 [link](./第1章-AI系统概述/1.1-AI历史、现状与发展.md)

1.2 算法、体系结构与算力的进步 [link](./第1章-AI系统概述/1.2-算法、体系结构与算力的进步.md)

1.3 AI系统组成与AI产业生态 [link](./第1章-AI系统概述/1.3-AI系统组成与AI产业生态.md)

1.4 AI背后系统问题 [link](./第1章-AI系统概述/1.4-AI背后系统问题.md)

1.5 影响AI系统的理论与原则 [link](./第1章-AI系统概述/1.5-影响AI系统的理论与原则.md)

### 二、AI框架核心技术

#### 2.1 AI框架基础概念

1. AI框架基本介绍

2. AI框架的作用

3. AI框架的发展

4. AI框架编程范式

#### 2.2 自动微分原理

1. 基本介绍  

2. 什么是微分 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/02.base_concept.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Ld4y1M7GJ/),[article](https://zhuanlan.zhihu.com/p/518198564)

3. 正反向计算模式 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/03.grad_mode.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1zD4y117bL/),[article](https://zhuanlan.zhihu.com/p/518296942)

4. 三种实现方法 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/04.grad_mode.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1BN4y1P76t/),[article](https://zhuanlan.zhihu.com/p/520065656)

5. 手把手实现正向微分框架 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/05.forward_mode.ipynb),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Ne4y1p7WU/),[article](https://zhuanlan.zhihu.com/p/520451681)

6. 亲自实现一个PyTorch [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/06.reversed_mode.ipynb),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1ae4y1z7E6/),[article](https://zhuanlan.zhihu.com/p/547865589)

7. 自动微分的挑战&未来 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/07.challenge.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV17e4y1z73W/)

#### 计算图——赵含霖

14. 基本介绍 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/01.introduction.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1cG411E7gV/)

15. 什么是计算图 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/02.computation_graph.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1rR4y197HM/)

16. 计算图跟自动微分关系 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/03.atuodiff.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1S24y197FU/)

17. 图优化与图执行调度 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/04.dispatch.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1hD4y1k7Ty/)

18. 计算图的控制流机制实现 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/05.control_flow.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV17P41177Pk/)

19. 计算图未来将会走向何方？[silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/06.future.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1hm4y1A7Nv/)

20. 分布式集群——管一鸣

21. 基本介绍 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/01.introduction.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1ge411L7mi/)

22. AI集群服务器架构 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/02.architecture.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1fg41187rc/)

23. AI集群软硬件通信 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/03.communication.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV14P4y1S7u4/)

24. 集合通信原语 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/04.primitive.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1te4y1e7vz/)

25. AI框架分布式功能 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/05.system.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1n8411s7f3/)

26. 分布式算法——赵含霖

27. 大模型训练的挑战 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/06.challenge.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Y14y1576A/)

28. 算法：大模型算法结构 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/07.algorithm_arch.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Mt4y1M7SE/)

29. 算法：亿级规模SOTA大模型 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/08.algorithm_sota.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1em4y1F7ay/)

30. 分布式并行——粟君杰

31. 基本介绍 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/01.introduction.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1ve411w7DL/)

32. 数据并行 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/02.data_parallel.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1JK411S7gL/)

33. 模型并行之张量并行 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/03.tensor_parallel.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1vt4y1K7wT/)

34. MindSpore张量并行 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/04.mindspore_parallel.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1vt4y1K7wT/)

35. 模型并行之流水并行 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/05.pipeline_parallel.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1WD4y1t7Ba/)

36. 混合并行 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/06.hybrid_parallel.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1gD4y1t7Ut/)

37. 分布式训练总结 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/07.summary.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1av4y1S7DQ/)