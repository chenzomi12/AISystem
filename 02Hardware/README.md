<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# 二、AI芯片体系结构

AI硬件体系结构主要是指AI芯片，这里就很硬核了，从芯片的基础到AI芯片的范围都会涉及，芯片设计需要考虑上面AI框架的前端、后端编译，而不是停留在天天喊着吊打英伟达，被现实打趴。

> 欢迎大家使用的过程中发现bug或者勘误直接提交PR到开源社区哦！

> 请大家尊重开源和ZOMI的努力，引用PPT的内容请规范转载标明出处哦！

## 课程简介

- [**《AI 计算体系》**](./01Foundation/)：深入深度学习计算模式，从而理解“计算”需要什么。通过AI芯片关键指标，了解AI芯片要更好的支持“计算”，需要关注那些重点工作。最后通过深度学习的计算核心“矩阵乘”来看对“计算”的实际需求和情况，为了提升计算性能、降低功耗和满足训练推理不同场景应用，对“计算”引入 TF32/BF16 等复杂多样的比特位宽。

- [**《AI 芯片基础》**](./02ChipBase/)：简单从CPU开始看通用逻辑架构（冯诺依曼架构）开始，通过打开计算的本质（数据与时延）从而引出对于并行计算GPU作用和解决的业务场景，到目前最火的AI芯片NPU。最后迈入超异构并行CPU、GPU、NPU并存的计算系统架构黄金十年。

- [**《图形处理器 GPU》**](./03GPUBase/)：主要是深入地讲解GPU的工作原理，其最重要的指标是计算吞吐和存储和传输带宽，并对英伟达的GPU的十年5代架构进行梳理。此外，《NVIDIA GPU详解》英伟达架构里面专门为AI而生的 Tensor Core 和 NVLink 对AI加速尤为重要，因此重点对 Tensor Core 和 NVLink 进行深入剖析其发展、演进和架构。

- [**《英伟达 GPU 详解》**](./04NVIDIA/): 英伟达架构里面专门为AI而生的 Tensor Core 和 NVLink 对AI加速尤为重要，因此重点对 Tensor Core 和 NVLink 进行深入剖析其发展、演进和架构。

- [**《国外 AI 芯片》**](./05Abroad/)：深入地剖析国外 Google TPU 和特斯拉 DOJO 相关 AI 芯片的架构，以TPU为主主要使用了数据流（Data FLow）的方式的脉动阵列来加速矩阵的运算，而特斯拉则使用了近存计算（Near Memory）两种不同的产品形态。

- [**《国内 AI 芯片》**](./06Domestic/)：深入地解读国内 AI 初创芯片厂商如国内第一AI芯片上市公司寒武纪、国内造GPU声势最大的壁仞科技、腾讯重头的燧原科技等科技公司的 AI 芯片架构。

- [**《AI 芯片黄金十年》**](./07Thought/)：基于 AI 芯片的 SIMD 硬件结构和 SIMT 的硬件结构原理，分析其上层的编程模型 SPMD 与 CUDA 之间的关系，去了解做好 AI 芯片其实跟软件的关联性也有着密切的关系，并对 AI 芯片近10年的发展进行一个总结和思考。

希望这个系列能够给朋友们带来一些帮助，也希望ZOMI能够继续坚持完成所有内容哈！欢迎您也参与到这个开源项目的贡献！

## 课程细节

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

### [AI 计算体系概述](./01Foundation/)

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| AI 计算体系 | 01 课程内容| [slide](./01Foundation/01.introduction), [video](https://www.bilibili.com/video/BV1DX4y1D7PC/) |
| AI 计算体系 | 02 AI计算模式(上) | [slide](./01Foundation/02.constraints.pdf), [video](https://www.bilibili.com/video/BV17x4y1T7Cn/) |
| AI 计算体系 | 03 AI计算模式(下) | [slide](./01Foundation/03.mobile_parallel.pdf), [video](https://www.bilibili.com/video/BV1754y1M78X/) |
| AI 计算体系 | 04 关键设计指标| [slide](./01Foundation/04.metrics.pdf), [video](https://www.bilibili.com/video/BV1qL411o7S9/) |
| AI 计算体系 | 05 核心计算：矩阵乘| [slide](./01Foundation/05.matrix.pdf), [video](https://www.bilibili.com/video/BV1ak4y1h7mp/) |
| AI 计算体系 | 06 数据单位：比特位 | [slide](./01Foundation/06.bit_width.pdf), [video](https://www.bilibili.com/video/BV1WT411k724/) |
| AI 计算体系 | 07 AI计算体系总结| [slide](./01Foundation/07.summary.pdf), [video](https://www.bilibili.com/video/BV1j54y1T7ii/) |

### [AI 芯片基础](./02ChipBase/)

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| AI 芯片基础 | 01 CPU 基础| [slide](./02ChipBase/01.cpu_base.pdf), [video](https://www.bilibili.com/video/BV1tv4y1V72f/)|
| AI 芯片基础 | 02 CPU 指令集架构 | [slide](./02ChipBase/02.cpu_isa.pdf), [video](https://www.bilibili.com/video/BV1ro4y1W7xN/) |
| AI 芯片基础 | 03 CPU 计算本质| [slide](./02ChipBase/03.cpu_data.pdf), [video](https://www.bilibili.com/video/BV17X4y1k7eF/)|
| AI 芯片基础 | 04 CPU 计算时延| [slide](./02ChipBase/04.cpu_latency.pdf), [video](https://www.bilibili.com/video/BV1Qk4y1i7GT/) |
| AI 芯片基础 | 05 GPU 基础| [slide](./02ChipBase/05.gpu.pdf), [video](https://www.bilibili.com/video/BV1sM411T72Q/) |
| AI 芯片基础 | 06 NPU 基础| [slide](./02ChipBase/06.npu.pptx), [video](https://www.bilibili.com/video/BV1Rk4y1e77n/)|
| AI 芯片基础 | 07 超异构计算 | [slide](./02ChipBase/07.future.pdf), [video](https://www.bilibili.com/video/BV1YM4y117VK) |

### [图形处理器 GPU](./03GPUBase/)

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 图形处理器 GPU | 01 GPU工作原理| [slide](./03_GPUBase/01.works.pdf), [video](https://www.bilibili.com/video/BV1bm4y1m7Ki/)|
| 图形处理器 GPU | 02 GPU适用于AI | [slide](./03_GPUBase/02.principle.pdf), [video](https://www.bilibili.com/video/BV1Ms4y1N7RL/)|
| 图形处理器 GPU | 03 GPU架构与CUDA关系 | [slide](./03_GPUBase/03.base_concept.pdf), [video](https://www.bilibili.com/video/BV1Kk4y1Y7op/) |
| 图形处理器 GPU | 04 GPU架构回顾第一篇 | [slide](./03_GPUBase/04.fermi.pdf), [video](https://www.bilibili.com/video/BV1x24y1F7kY/)|
| 图形处理器 GPU | 05 GPU架构回顾第二篇 | [slide](./03_GPUBase/05.turing.pdf), [video](https://www.bilibili.com/video/BV1mm4y1C7fg/) |

### [英伟达 GPU 详解](./04NVIDIA/)

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| GPU 原理详解 | 01 TensorCore原理(上) | [slide](./04NVIDIA/01.basic_tc.pdf), [video](https://www.bilibili.com/video/BV1aL411a71w/)|
| GPU 原理详解 | 02 TensorCore架构(中) | [slide](./04NVIDIA/02.history_tc.pdf), [video](https://www.bilibili.com/video/BV1pL41187FH/)|
| GPU 原理详解 | 03 TensorCore剖析(下) | [slide](./04NVIDIA/03.deep_tc.pdf), [video](https://www.bilibili.com/video/BV1oh4y1J7B4/) |
| GPU 原理详解 | 04 分布式通信与NVLink| [slide](./04NVIDIA/04.basic_nvlink.pdf), [video](https://www.bilibili.com/video/BV1cV4y1r7Rz/)|
| GPU 原理详解 | 05 NVLink原理剖析| [slide](./04NVIDIA/05.deep_nvlink.pdf), [video](https://www.bilibili.com/video/BV1uP411X7Dr/) |
| GPU 原理详解 | 05 NVSwitch原理剖析| [slide](./04NVIDIA/06.deep_nvswitch.pdf), [video](https://www.bilibili.com/video/BV1uM4y1n7qd/) |

### [国外 AI 芯片](./05Abroad/)

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 国外 AI 芯片 | 01 特斯拉 DOJO 架构 | [slide](./05Abroad/01.DOJO_Arch.pdf), [video](https://www.bilibili.com/video/BV1Ro4y1M7n8/) |
| 国外 AI 芯片 | 02 特斯拉 DOJO 细节 | [slide](./05Abroad/02.DOJO_Detail.pdf), [video](https://www.bilibili.com/video/BV17o4y1N7Yn/) |
| 国外 AI 芯片 | 03 特斯拉 DOJO 存算系统 | [slide](./05Abroad/03.DOJO_System.pdf), [video](https://www.bilibili.com/video/BV1Ez4y1e7zo/) |
| 国外 AI 芯片 | 04 谷歌 TPU 芯片发展 | [slide](./05Abroad/04.TPU_Introl.pdf), [video](https://www.bilibili.com/video/BV1Dp4y1V7PF/) |
| 国外 AI 芯片 | 05 谷歌 TPU1 脉动阵列 | [slide](./05Abroad/05.TPU1.pdf), [video](https://www.bilibili.com/video/BV12P411W7YC/) |
| 国外 AI 芯片 | 06 谷歌 TPU2 推理到训练 | [slide](./05Abroad/06.TPU2.pdf), [video](https://www.bilibili.com/video/BV1x84y1f7Ex/) |
| 国外 AI 芯片 | 07 谷歌 TPU3 POD超节点 | [slide](./05Abroad/07.TPU3.pdf), [video](https://www.bilibili.com/video/BV1Pm4y1g7MG/) |
| 国外 AI 芯片 | 08 谷歌 TPU4 AI集群 | [slide](./05Abroad/08.TPU4.pdf), [video](https://www.bilibili.com/video/BV1QH4y1X77U) |
| 国外 AI 芯片 | 08 谷歌 OCS 光交换机  | [slide](./05Abroad/08.TPU4.pdf), [video](https://www.bilibili.com/video/BV1yc411o7cQ) |

### [国内 AI 芯片](./06Domestic/)

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 国外 AI 芯片 | 01 壁仞BR100产品介绍 | [slide](./06Domestic/01.BR100_System.pdf), [video](https://www.bilibili.com/video/BV1QW4y1S75Y)|
| 国外 AI 芯片 | 02 壁仞BR100芯片架构 | [slide](./06Domestic/02.BR100_Detail.pdf), [video](https://www.bilibili.com/video/BV1G14y1275T)|
| 国外 AI 芯片 | 03 燧原科技AI芯片 | [slide](./06Domestic/03.SUIYUAN_DTU.pdf), [video](https://www.bilibili.com/video/BV15W4y1Z7Hj)|
| 国外 AI 芯片 | 04 寒武纪AI芯片第一股 | [slide](./06Domestic/04.cambricon_Product.pdf), [video](https://www.bilibili.com/video/BV1Y8411m7Cd)|
| 国外 AI 芯片 | 05 寒武纪AI芯片架构剖析（上） | [slide](./06Domestic/05.cambricon_Arch.pdf), [video](https://www.bilibili.com/video/BV1op4y157Qf)|
| 国外 AI 芯片 | 06 寒武纪AI芯片架构剖析（下） | [slide](./06Domestic/06.cambricon_Arch.pdf), [video](https://www.bilibili.com/video/BV1TV411j7Yx)|

### [AI 芯片黄金十年](./07Thought/)

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| AI 芯片黄金十年 | 01 芯片的编程体系 | [slide](./01.Introduction.pdf), [video](https://www.bilibili.com/video/BV13u4y197Lw)|
| AI 芯片黄金十年 | 02 SIMD和SIMT跟AI芯片关系 | [slide](./02.SIMT&SIMD.pdf), [video](https://www.bilibili.com/video/BV1Kr4y1d7eW)|
| AI 芯片黄金十年 | 03 CUDA/SIMD/SIMT/DSA关系 | [slide](./03.SPMT.pdf), [video](https://www.bilibili.com/video/BV1WC4y1w79T)|
| AI 芯片黄金十年 | 04 CUDA跟SIMT硬件关系 | [slide](./04.NVSIMT.pdf), [video](https://www.bilibili.com/video/BV16c41117vp)|
| AI 芯片黄金十年 | 05 从CUDA和NVIDIA中借鉴 | [slide](./05.DSA.pdf), [video](https://www.bilibili.com/video/BV1j94y1N7qh)|
| AI 芯片黄金十年 | 06 AI芯片的思考 | [slide](./06.AIChip.pdf), [video](https://www.bilibili.com/video/BV1te411y7UC/)|

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AISys](https://chenzomi12.github.io/) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT开源在[github](https://github.com/chenzomi12/DeepLearningSystem)，欢迎取用！！！

> 非常希望您也参与到这个开源项目中，B站给ZOMI留言哦！
>
> 欢迎大家使用的过程中发现bug或者勘误直接提交代码PR到开源社区哦！
>
> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！
