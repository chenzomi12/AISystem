<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# == 五、AI芯片核心原理 ==

AI芯片这里就很硬核了，从芯片的基础到AI芯片的范围都会涉及，芯片设计需要考虑上面AI框架的前端、后端编译，而不是停留在天天喊着吊打英伟达，被现实打趴。

> 欢迎大家使用的过程中发现bug或者勘误直接提交PR到开源社区哦！

> 请大家尊重开源和ZOMI的努力，引用PPT的内容请规范转载标明出处哦！

## 课程简介

- 《AI 计算体系》深入深度学习计算模式，从而理解“计算”需要什么。通过AI芯片关键指标，了解AI芯片要更好的支持“计算”，需要关注那些重点工作。最后通过深度学习的计算核心“矩阵乘”来看对“计算”的实际需求和情况，为了提升计算性能、降低功耗和满足训练推理不同场景应用，对“计算”引入 TF32/BF16 等复杂多样的比特位宽。

- 《AI 芯片基础》简单从CPU开始看通用逻辑架构（冯诺依曼架构）开始，通过打开计算的本质（数据与时延）从而引出对于并行计算GPU作用和解决的业务场景，到目前最火的AI芯片NPU。最后迈入超异构并行CPU、GPU、NPU并存的计算系统架构黄金十年。

- 《GPU 原理详解》主要是深入地讲解GPU的工作原理，其最重要的指标是计算吞吐和存储和传输带宽，并对英伟达的GPU的十年5代架构进行梳理。此外，《NVIDIA GPU详解》英伟达架构里面专门为AI而生的 Tensor Core 和 NVLink 对AI加速尤为重要，因此重点对 Tensor Core 和 NVLink 进行深入剖析其发展、演进和架构。

- 《国外AI芯片》更新中ING...

- 《国内AI芯片》更新中ING...

希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

## 课程细节

> *建议优先下载或者使用PDF版本，PPT版本会因为字体缺失等原因导致版本很丑哦~*

### AI 计算体系

- 《AI 计算体系》深入深度学习计算模式，从而理解“计算”需要什么。通过AI芯片关键指标，了解AI芯片要更好的支持“计算”，需要关注那些重点工作。最后通过深度学习的计算核心“矩阵乘”来看对“计算”的实际需求和情况，为了提升计算性能、降低功耗和满足训练推理不同场景应用，对“计算”引入 TF32/BF16 等复杂多样的比特位宽。

| 分类   | 名称               | 内容                                                                                                   |
|:-:|:-:|:-:|
| AI 计算体系  | 01 课程内容            | [video](https://www.bilibili.com/video/BV1DX4y1D7PC/)        |
| AI 计算体系  | 02 AI计算模式(上)       |[video](https://www.bilibili.com/video/BV17x4y1T7Cn/)     |
| AI 计算体系  | 03 AI计算模式(下)       |[video](https://www.bilibili.com/video/BV1754y1M78X/) |
| AI 计算体系  | 04 关键设计指标          |[video](https://www.bilibili.com/video/BV1qL411o7S9/)         |
| AI 计算体系  | 05 核心计算：矩阵乘        |[video](https://www.bilibili.com/video/BV1ak4y1h7mp/)          |
| AI 计算体系  | 06 数据单位：bits       |[video](https://www.bilibili.com/video/BV1WT411k724/)       |
| AI 计算体系  | 07 AI计算体系总结        |[video](https://www.bilibili.com/video/BV1j54y1T7ii/)         |

### AI 芯片基础

- 《AI 芯片基础》简单从CPU开始看通用逻辑架构（冯诺依曼架构）开始，通过打开计算的本质（数据与时延）从而引出对于并行计算GPU作用和解决的业务场景，到目前最火的AI芯片NPU。最后迈入超异构并行CPU、GPU、NPU并存的计算系统架构黄金十年。

| 分类   | 名称               | 内容                                                                                                   |
|:-:|:-:|:-:|
| AI 芯片基础  | 01 CPU 基础          |[video](https://www.bilibili.com/video/BV1tv4y1V72f/)          |
| AI 芯片基础  | 02 CPU 指令集架构       |[video](https://www.bilibili.com/video/BV1ro4y1W7xN/)           |
| AI 芯片基础  | 03 CPU 计算本质        |[video](https://www.bilibili.com/video/BV17X4y1k7eF/)          |
| AI 芯片基础  | 04 CPU 计算时延        |[video](https://www.bilibili.com/video/BV1Qk4y1i7GT/)       |
| AI 芯片基础  | 05 GPU 基础          |[video](https://www.bilibili.com/video/BV1sM411T72Q/)               |
| AI 芯片基础  | 06 NPU 基础          | [video](https://www.bilibili.com/video/BV1Rk4y1e77n/)              |
| AI 芯片基础  | 07 超异构计算           |[video](https://www.bilibili.com/video/BV1YM4y117VK)             |

### GPU 原理详解

- 《GPU 原理详解》主要是深入地讲解GPU的工作原理，其最重要的指标是计算吞吐和存储和传输带宽，并对英伟达的GPU的十年5代架构进行梳理。英伟达架构里面专门为AI而生的 Tensor Core 和 NVLink 对AI加速尤为重要，因此重点对 Tensor Core 和 NVLink 进行深入剖析其发展、演进和架构。

| 分类   | 名称               | 内容                                                                                                   |
|:-:|:-:|:-:|
| GPU 原理详解 | 01 GPU工作原理         |[video](https://www.bilibili.com/video/BV1bm4y1m7Ki/)              |
| GPU 原理详解 | 02 GPU适用于AI        |[video](https://www.bilibili.com/video/BV1Ms4y1N7RL/)          |
| GPU 原理详解 | 03 GPU架构与CUDA关系    |[video](https://www.bilibili.com/video/BV1Kk4y1Y7op/)       |
| GPU 原理详解 | 04 GPU架构回顾第一篇      |[video](https://www.bilibili.com/video/BV1x24y1F7kY/)              |
| GPU 原理详解 | 05 GPU架构回顾第二篇      |[video](https://www.bilibili.com/video/BV1mm4y1C7fg/)             |
| NVIDIA GPU详解 | 01 TensorCore原理(上) |[video](https://www.bilibili.com/video/BV1aL411a71w/)         |
| NVIDIA GPU详解 | 02 TensorCore架构(中) |[video](https://www.bilibili.com/video/BV1pL41187FH/)       |
| NVIDIA GPU详解 | 03 TensorCore剖析(下) |[video](https://www.bilibili.com/video/BV1oh4y1J7B4/)          |
| NVIDIA GPU详解 | 04 分布式通信与NVLink    |[video](https://www.bilibili.com/video/BV1cV4y1r7Rz/)     |
| NVIDIA GPU详解 | 05 NVLink原理剖析      |[video](https://www.bilibili.com/video/BV1uP411X7Dr/)      |
| NVIDIA GPU详解 | 05 NVSwitch原理剖析    |[video](https://www.bilibili.com/video/BV1uM4y1n7qd/)    |

### 国外AI芯片

更新中ING...

| 分类   | 名称               | 内容                                                                                                   |
|:-:|:-:|:-:|
| 国外AI芯片 | 01 特斯拉DOJO架构       |[video](https://www.bilibili.com/video/BV1Ro4y1M7n8/)              |
| 国外AI芯片 | 02 特斯拉DOJO Core原理  |[video](https://www.bilibili.com/video/BV17o4y1N7Yn/)            |
| 国外AI芯片 | 03 特斯拉DOJO存算系统     |[video](https://www.bilibili.com/video/BV1Ez4y1e7zo/)            |

### 国内AI芯片

更新中ING...

| 分类   | 名称               | 内容                                                                                                   |
|:-:|:-:|:-:|
| 国内AI芯片 | 01 壁仞产品解读          |[video](https://www.bilibili.com/video/BV1QW4y1S75Y/)           |
| 国内AI芯片 | 02 壁仞BR100架构       |[video](https://www.bilibili.com/video/BV1G14y1275T/)          |
| 国内AI芯片 | 03 燧原产品与DTU架构       |         |
