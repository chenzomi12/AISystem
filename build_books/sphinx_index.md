---
title: AISystem & AIInfra 
---

::::{grid}
:reverse:
:gutter: 2 1 1 1
:margin: 4 4 1 1

:::{grid-item}
:columns: 4

```{image} ./_static/logo-square.svg
:width: 150px
```
:::
::::

本开源项目主要是跟大家一起探讨和学习人工智能、深度学习的系统设计，而整个系统是围绕着在 NVIDIA、ASCEND 等芯片厂商构建算力层面，所用到的、积累、梳理得到 AI 系统全栈的内容。希望跟所有关注 AI 开源项目的好朋友一起探讨研究，共同促进学习讨论。

![AI系统全栈架构图](images/01Introduction/03Architecture03.png)

# 课程内容大纲

课程主要包括以下六大模块：

第一部分，AI 基础知识和 AI 系统的全栈概述的[<u>**AI 系统概述**</u>](./01Introduction/README.md)，以及深度学习系统的系统性设计和方法论，主要是整体了解 AI 训练和推理全栈的体系结构内容。

第二部分，硬核篇介绍[<u>**AI 芯片概况**</u>](./02Hardware/README.md)，这里就很硬核了，从芯片基础到 AI 芯片的范围都会涉及，芯片设计需要考虑上面 AI 框架的前端、后端编译，而不是停留在天天喊着吊打英伟达，被现实打趴。

第三部分，进阶篇介绍[<u>**AI 编译器原理**</u>](./03Compiler/README.md)，将站在系统设计的角度，思考在设计现代机器学习系统中需要考虑的编译器问题，特别是中间表达乃至后端优化。

第四部分，实际应用[<u>**推理系统与引擎**</u>](./04Inference/README.md)，讲了太多原理身体太虚容易消化不良，还是得回归到业务本质，让行业、企业能够真正应用起来，而推理系统涉及一些核心算法和注意的事情也分享下。

第五部分，介绍[<u>**AI 框架核心技术**</u>](./05Framework/README.md)，首先介绍任何一个 AI 框架都离不开的自动微分，通过自动微分功能后就会产生表示神经网络的图和算子，然后介绍 AI 框架前端的优化，还有最近很火的大模型分布式训练在 AI 框架中的关键技术。

第六部分，汇总篇介绍<u>**大模型与 AI 系统**</u>，大模型是基于 AI 集群的全栈软硬件性能优化，通过最小的每一块 AI 芯片组成的 AI 集群，编译器使能到上层的 AI 框架，训练过程需要分布式并行、集群通信等算法支持，而且在大模型领域最近持续演进如智能体等新技术。

:::{大模型与 AI 系统} 大模型与到AI系统因为内容过多，引起整个 AI 产业和周边的网络、存储、通讯、机房建设风火水电等，在 AI 系统上也加入了更多的集合通信、分布式加速库、AI Agent等内容，因此独立一个大内容后续再详细展开。 :::

# 课程设立目的

本课程主要为本科生高年级、硕博研究生、AI 系统从业者设计，帮助大家：

1. 完整了解 AI 的计算机系统架构，并通过实际问题和案例，来了解 AI 完整生命周期下的系统设计。

2. 介绍前沿系统架构和 AI 相结合的研究工作，了解主流框架、平台和工具来了解 AI 系统。

**先修课程:** C++/Python，计算机体系结构，人工智能基础

# 课程目录内容

<!-- ## 一. AI 系统概述 -->

```{toctree}
:maxdepth: 1
:caption: === 一. AI 系统概述 ===

01Introduction/README
```

<!-- ## 二. AI 硬件体系结构 -->

```{toctree}
:maxdepth: 1
:caption: === 二. AI 硬件体系结构 ===

02Hardware/README
02Hardware01Foundation/README
02Hardware02ChipBase/README
02Hardware03GPUBase/README
02Hardware04NVIDIA/README
02Hardware05Abroad/README
02Hardware06Domestic/README
02Hardware07Thought/README
```

<!-- ## 三. AI 编译器 -->

```{toctree}
:maxdepth: 1
:caption: === 三. AI 编译器 ===

03Compiler/README
03Compiler01Tradition/README
03Compiler02AICompiler/README
03Compiler03Frontend/README
03Compiler04Backend/README
```

<!-- ## 四. 推理系统&引擎 -->

```{toctree}
:maxdepth: 1
:caption: === 四. 推理系统&引擎 ===

04Inference/README
04Inference01Inference/README
04Inference02Mobilenet/README
04Inference03Slim/README
04Inference04Converter/README
04Inference05Optimize/README
04Inference06Kernel/README
```

<!-- ## 五. AI 框架核心模块 -->

```{toctree}
:maxdepth: 1
:caption: === 五. AI 框架核心模块 ===

05Framework/README
05Framework01Foundation/README
05Framework02AutoDiff/README
05Framework03DataFlow/README
05Framework04Parallel/README
```

<!-- ## 附录内容 -->

```{toctree}
:caption: === 附录内容 ===
:maxdepth: 1

00Others/README
```

## 备注

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT 开源在[github](https://github.com/chenzomi12/AISystem)，欢迎取用！！！

> 非常希望您也参与到这个开源项目中，B 站给 ZOMI 留言哦！
> 
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交代码 PR 到开源社区哦！
>
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交 PR 到开源社区哦！
>
> 请大家尊重开源和 ZOMI 的努力，引用 PPT 的内容请规范转载标明出处哦！