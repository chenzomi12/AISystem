# AI芯片：核心原理

AI芯片这里就很硬核了，希望可以坚持到最后啦，从芯片的基础到AI芯片的范围都会涉及，芯片设计需要考虑上面AI框架的前端、后端编译，而不是停留在天天喊着吊打英伟达，被现实打趴。

## 课程简介

- 《AI 计算体系》深入深度学习计算模式，从而理解“计算”需要什么。通过AI芯片关键指标，了解AI芯片要更好的支持“计算”，需要关注那些重点工作。最后通过深度学习的计算核心“矩阵乘”来看对“计算”的实际需求和情况，为了提升计算性能、降低功耗和满足训练推理不同场景应用，对“计算”引入 TF32/BF16 等复杂多样的比特位宽。

- 《AI 芯片基础》简单从CPU开始看通用逻辑架构（冯诺依曼架构）开始，通过打开计算的本质（数据与时延）从而引出对于并行计算GPU作用和解决的业务场景，到目前最火的AI芯片NPU。最后迈入超异构并行CPU、GPU、NPU并存的计算系统架构黄金十年。

- 《GPU 原理详解》主要是深入地讲解GPU的工作原理，其最重要的指标是计算吞吐和存储和传输带宽，并对英伟达的GPU的十年5代架构进行梳理。

- 《Tensor Core 与 NVlink》英伟达架构里面专门为AI而生的 Tensor Core 和 NVLink 对AI加速尤为重要，因此重点对 Tensor Core 和 NVLink 进行深入剖析其发展、演进和架构。

希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

## 课程目标

通过《AI芯片：核心原理》这个课程，以及这门课程后面的几门课程，你将获取并且掌握的技能：

|        |                             |                               |
|:------:|:--------------------------- |:----------------------------- |
| **编号** | **名称**                      | **具体内容**                      |
| 1      | [AI 计算体系](./01%20Foundation/)    | 神经网络等AI技术的计算模式和计算体系架构         |
| 2      | [AI 芯片基础](./02%20ChipBase/)      | CPU、GPU、NPU等芯片基础原理，与体系架构黄金10年 |
| 3      | [通用图形处理器 GPU](./03%20GPUBase/)   | GPU的基本原理，英伟达GPU的架构发展          |
| 4      | [英伟达GPU的AI详解](./04%20GPUDetail/) | 英伟达GPU的TensorCore、NVLink深度剖析  |
| 5      | [AI专用处理器 NPU](./05%20NPU/)       | 华为、谷歌、特斯拉等专用AI处理器核心原理         |

## 课程细节

> *建议优先下载或者使用PDF版本，PPT版本会因为字体缺失等原因导致版本很丑哦~*

|     |          |                    |                                                                                                     |     |
| --- | -------- | ------------------ | --------------------------------------------------------------------------------------------------- | --- |
| 编号  | 名称       | 内容                 | 资源                                                                                                  | 备注  |
| 1   | AI 计算体系  | 01 课程内容            | [slide](./01%20Foundation/01.introduction), [video](https://www.bilibili.com/video/BV1DX4y1D7PC/)        |     |
| 1   | AI 计算体系  | 02 AI计算模式(上)       | [slide](./01%20Foundation/02.constraints.pdf), [video](https://www.bilibili.com/video/BV17x4y1T7Cn/)     |     |
| 1   | AI 计算体系  | 03 AI计算模式(下)       | [slide](./01%20Foundation/03.mobile_parallel.pdf), [video](https://www.bilibili.com/video/BV1754y1M78X/) |     |
| 1   | AI 计算体系  | 04 关键设计指标          | [slide](./01%20Foundation/04.metrics.pdf), [video](https://www.bilibili.com/video/BV1qL411o7S9/)         |     |
| 1   | AI 计算体系  | 05 核心计算：矩阵乘        | [slide](./01%20Foundation/05.matrix.pdf), [video](https://www.bilibili.com/video/BV1ak4y1h7mp/)          |     |
| 1   | AI 计算体系  | 06 数据单位：bits       | [slide](./01%20Foundation/06.bit_width.pdf), [video](https://www.bilibili.com/video/BV1WT411k724/)       |     |
| 1   | AI 计算体系  | 07 AI计算体系总结        | [slide](./01%20Foundation/07.summary.pdf), [video](https://www.bilibili.com/video/BV1j54y1T7ii/)         |     |
|     |          |                    |                                                                                                     |     |
| 2   | AI 芯片基础  | 01 CPU 基础          | [slide](./02%20ChipBase/01.cpu_base.pdf), [video](https://www.bilibili.com/video/BV1tv4y1V72f/)          |     |
| 2   | AI 芯片基础  | 02 CPU 指令集架构       | [slide](./02%20ChipBase/02.cpu_isa.pdf), [video](https://www.bilibili.com/video/BV1ro4y1W7xN/)           |     |
| 2   | AI 芯片基础  | 03 CPU 计算本质        | [slide](./02%20ChipBase/03.cpu_data.pdf), [video](https://www.bilibili.com/video/BV17X4y1k7eF/)          |     |
| 2   | AI 芯片基础  | 04 CPU 计算时延        | [slide](./02%20ChipBase/04.cpu_latency.pdf), [video](https://www.bilibili.com/video/BV1Qk4y1i7GT/)       |     |
| 2   | AI 芯片基础  | 05 GPU 基础          | [slide](./02%20ChipBase/05.gpu.pdf), [video](https://www.bilibili.com/video/BV1sM411T72Q/)               |     |
| 2   | AI 芯片基础  | 06 NPU 基础          | [slide](./02%20ChipBase/06.npu.pptx), [video](https://www.bilibili.com/video/BV1Rk4y1e77n/)              |     |
| 2   | AI 芯片基础  | 07 超异构计算           | [slide](./02%20ChipBase/07.future.pdf), [video](https://www.bilibili.com/video/BV1YM4y117VK)             |     |
|     |          |                    |                                                                                                     |     |
| 3   | GPU 原理详解 | 01 GPU工作原理         | [slide](./03%20GPUBase/01.works.pdf), [video](https://www.bilibili.com/video/BV1bm4y1m7Ki/)              |     |
| 3   | GPU 原理详解 | 02 GPU适用于AI        | [slide](./03%20GPUBase/02.principle.pdf), [video](https://www.bilibili.com/video/BV1Ms4y1N7RL/)          |     |
| 3   | GPU 原理详解 | 03 GPU架构与CUDA关系    | [slide](./03%20GPUBase/03.base_concept.pdf), [video](https://www.bilibili.com/video/BV1Kk4y1Y7op/)       |     |
| 3   | GPU 原理详解 | 04 GPU架构回顾第一篇      | [slide](./03%20GPUBase/04.fermi.pdf), [video](https://www.bilibili.com/video/BV1x24y1F7kY/)              |     |
| 3   | GPU 原理详解 | 05 GPU架构回顾第二篇      | [slide](./03%20GPUBase/05.turing.pdf), [video](https://www.bilibili.com/video/BV1mm4y1C7fg/)             |     |
|     |          |                    |                                                                                                     |     |
| 4   | GPU 原理详解 | 01 TensorCore原理(上) | [slide](./04%20GPUDetail/01.basic_tc.pdf), [video](https://www.bilibili.com/video/BV1aL411a71w/)         |     |
| 4   | GPU 原理详解 | 02 TensorCore架构(中) | [slide](./04%20GPUDetail/02.history_tc.pdf), [video](https://www.bilibili.com/video/BV1pL41187FH/)       |     |
| 4   | GPU 原理详解 | 03 TensorCore剖析(下) | [slide](./04%20GPUDetail/03.deep_tc.pdf), [video](https://www.bilibili.com/video/BV1oh4y1J7B4/)          |     |

## 目标学员

1. 对人工智能、深度学习感兴趣的学员

2. 渴望学习当今最热门最前沿AI技术的学员

3. 想储备深度学习技能的学员

4. AI系统开发工程师