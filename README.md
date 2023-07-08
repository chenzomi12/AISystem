# Deep Learning System & AI System

[![Continuous Integration](https://github.com/d2l-ai/d2l-en/actions/workflows/ci.yml/badge.svg)](https://github.com/d2l-ai/d2l-en/actions/workflows/ci.yml)
[![Build Docker Image](https://github.com/d2l-ai/d2l-en/actions/workflows/build-docker.yml/badge.svg)](https://github.com/d2l-ai/d2l-en/actions/workflows/build-docker.yml)

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AISys](https://chenzomi12.github.io/) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@zomi6222/videos)，PPT开源在[github](https://github.com/chenzomi12/DeepLearningSystem)，欢迎取用！！！

> 非常希望您也参与到这个开源项目中，联系方式[Github Issues] 区留言哦
>
> 欢迎大家使用的过程中发现bug或者勘误直接提交PR到开源社区哦

## 项目背景

这个开源项目英文名字叫做 **Deep Learning System** 或者 **AI System(AISys)**、**AI Infra**、**ML System(MLSys)**，中文名字叫做 **深度学习系统** 或者 **AI系统**。

本开源项目主要是跟大家一起探讨和学习人工智能、深度学习的计算机系统设计，而整个系统是围绕着 ZOMI 在华为昇腾工作当中所积累、梳理、构建 AI 系统全栈的内容。希望跟所有关注 AI 开源项目的好朋友一起探讨研究，共同促进学习讨论。

近年来人工智能特别是深度学习技术得到了飞速发展，这背后离不开计算机硬件和软件系统的不断进步。在可见的未来，人工智能技术的发展仍将依赖于计算机系统和人工智能相结合的共同创新模式。需要注意的是，计算机系统现在正以更大的规模和更高的复杂性来赋能于人工智能，这背后不仅需要更多的系统上的创新，更需要系统性的思维和方法论。与此同时，人工智能也反过来为设计复杂系统提供支持。

我们注意到，现在的大部分人工智能相关的课程，特别是深度学习和机器学习相关课程主要集中在相关理论、算法或者应用，与系统相关的课程并不多见。我们希望人工智能系统这门课能让人工智能相关教育变得更加全面和深入，以共同促进人工智能与系统在开源方面的共同学习和讨论。（原谅我复制粘贴微软[AI-System](https://github.com/microsoft/AI-System)的介绍哈，人家写得很好啦）

## 课程内容大纲

课程主要包括以下五大模块：

第一部分，AI基础知识和AI系统的全栈概述的<u>**AI系统概述**</u>，以及深度学习系统的系统性设计和方法论，主要是整体了解AI训练和推理全栈的体系结构内容。

第二部分，基础篇介绍AI框架的<u>**AI框架核心技术**</u>，首先介绍任何一个AI框架都离不开的自动微分，通过自动微分功能后就会产生表示神经网络的图和算子，然后介绍AI框架前端的优化，还有最近很火的大模型分布式训练在AI框架中的关键技术。

第三部分，进进阶篇介绍AI框架<u>**AI编译器原理**</u>，将站在系统设计的角度，思考在设计现代机器学习系统中需要考虑的编译器问题，特别是中间表达乃至后端优化。

第四部分，是很实际的<u>**推理系统**</u>，讲了太多原理身体太虚容易消化不良，还是得回归到业务本质，让行业、企业能够真正应用起来，而推理系统涉及一些核心算法和注意的事情也分享下。

第五部分，硬核篇介绍<u>**AI芯片**</u>，这里就很硬核了，从芯片基础到AI芯片的范围都会涉及，芯片设计需要考虑上面AI框架的前端、后端编译，而不是停留在天天喊着吊打英伟达，被现实打趴。

## 课程设立目的

本课程主要为本科生高年级、硕博研究生、AI系统从业者设计，帮助大家：

1. 完整的了解支持AI的计算机系统架构，并通过实际问题和案例，来了解AI完整生命周期下的系统设计。

2. 介绍前沿的工程系统和AI相结合的研究工作，包括AI for Systems和Systems for AI（即AISys），以帮助大家更好的寻找有意义的研究课题。

3. 从AI系统研究的角度出发设计全栈课程。通过了解主流和最新框架、平台和工具来了解AI系统，以提高解决实际问题能力。

**先修课程:** C++/Python，计算机体系结构，算法导论，人工智能基础

### 课程部分

**[二. AI框架核心技术](./020Framework/)**

| 编号  | 名称                                  | 具体内容                        |
|:---:|:----------------------------------- |:--------------------------- |
| 1   | [AI框架基础](./021FW_Foundation/) | AI框架的作用、发展、编程范式             |
| 2   | [自动微分](./022FW_AutoDiff/)     | 自动微分的实现方式和原理                |
| 3   | [计算图](./023FW_DataFlow/)      | 计算图的概念，图优化、图执行、控制流表达        |
| 4   | [分布式集群](./024FW_AICluster/)    | AI集群服务器架构、软硬件通信方式           |
| 5   | [分布式算法](./025FW_AIAlgo/)       | 大模型挑战，Transformer和MOE结构的大模型 |
| 6   | [分布式并行](./026FW_Parallel/)     | 数据并行、模型并行、混合并行的原理和策略        |
|     |                                     |                             |

**[三. AI编译原理](./030Compiler/)**

|        |                                     |                                 |
|:------:|:----------------------------------- |:------------------------------- |
| **编号** | **名称**                              | **具体内容**                        |
| 1      | [传统编译器](./031CM_Tradition/)    | 传统编译器GCC与LLVM，LLVM详细架构          |
| 2      | [AI 编译器](./032CM_AICompiler/)  | AI编译器发展与架构定义，未来挑战与思考            |
| 3      | [前端优化](./033CM_Frontend/)      | AI编译器的前端优化(算子融合、内存优化等)          |
| 4      | [后端优化](./034CM_Backend/)       | AI编译器的后端优化(Kernel优化、AutoTuning) |
| 5      | 多面体                                 | 待更ing...                        |
| 6      | [PyTorch2.0](./036CM_PyTorch/) | PyTorch2.0最重要的新特性：编译技术栈         |
|        |                                     |                                 |

**[四. AI推理系统](./040Inference/)**

|        |                                    |                            |
|:------:|:---------------------------------- |:-------------------------- |
| **编号** | **名称**                             | **具体内容**                   |
| 1      | [推理系统](./041INF_Inference/)  | 推理系统整体介绍，推理引擎架构梳理          |
| 2      | [轻量网络](./042INF_Mobilenet/)  | 轻量化主干网络，MobileNet等SOTA模型介绍 |
| 3      | [模型压缩](./043INF_Slim/)       | 模型压缩4件套，量化、蒸馏、剪枝和二值化       |
| 4      | [转换&优化](./044INF_Converter/) | AI框架训练后模型进行转换，并对计算图优化      |
| 5      | [Kernel优化](./045INF_Kernel/) | Kernel层、算子层优化，对算子、内存、调度优化  |

**[五. AI芯片架构](./050Hardware/)**

|        |                                      |                              |
|:------:|:------------------------------------ |:---------------------------- |
| **编号** | **名称**                               | **具体内容**                     |
| 1      | [AI 计算体系](./051HW_Foundation/) | 神经网络等AI技术的计算模式和计算体系架构        |
| 2      | [AI 芯片基础](./052HW_ChipBase/)   | CPU、GPU、NPU等芯片体系架构基础原理       |
| 3      | [图形处理器 GPU](./053HW_GPUBase/)  | GPU的基本原理，英伟达GPU的架构发展         |
| 4      | [英伟达GPU详解](./054HW_GPUDetail/) | 英伟达GPU的TensorCore、NVLink深度剖析 |
| 5      | [AI专用处理器](./055HW_NPU/)         | 华为、谷歌、特斯拉等专用AI处理器核心原理        |

## 贡献者

<!-- readme: collaborators,contributors -start -->

<a href="https://github.com/chenzomi12/DeepLearningSystem/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=chenzomi12/DeepLearningSystem" />
</a>

## 备注

> 这个仓已经到达疯狂的8G啦（ZOMI把所有的制作过程、高清图片都原封不动提供），如果你要git clone会非常的慢，因此建议到  [Releases · chenzomi12/DeepLearningSystem](https://github.com/chenzomi12/DeepLearningSystem/releases) 来下载你需要的内容。

<!-- readme: collaborators,contributors -end -->