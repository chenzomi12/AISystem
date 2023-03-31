# AI芯片：核心原理

XXXX待补充！AI芯片这里就很硬核了，希望可以坚持到最后啦，从芯片的基础到AI芯片的范围都会涉及，芯片设计需要考虑上面AI框架的前端、后端编译，而不是停留在天天喊着吊打英伟达，被现实打趴。

## 课程简介

- 《AI 计算体系》深入深度学习计算模式，从而理解“计算”需要什么。通过AI芯片关键指标，了解AI芯片要更好的支持“计算”，需要关注那些重点工作。最后通过深度学习的计算核心“矩阵乘”来看对“计算”的实际需求和情况，为了提升计算性能、降低功耗和满足训练推理不同场景应用，对“计算”引入 TF32/BF16 等复杂多样的比特位宽。

- 

- 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

## 课程目标

通过《AI芯片：核心原理》这个课程，以及这门课程后面的几门课程，你将获取并且掌握的技能：

|        |                          |                               |
|:------:|:------------------------ |:----------------------------- |
| **编号** | **名称**                   | **具体内容**                      |
| 1      | [AI 计算体系](./Foundation/) | 神经网络等AI技术的计算模式和计算体系架构         |
| 2      | [AI 芯片基础](./ChipBase/)   | CPU、GPU、NPU等芯片基础原理，与体系架构黄金10年 |
| 3      | [通用图形处理器 GPU](./GPU/)    | 模型压缩4件套，量化、蒸馏、剪枝和二值化          |
| 4      | [AI专用处理器 NPU](./NPU)     | 华为、谷歌、特斯拉等专用AI处理器核心原理         |

## 课程细节

> *建议优先下载或者使用PDF版本，PPT版本会因为字体缺失等原因导致版本很丑哦~*

|     |         |              |                                                                                                     |     |
| --- | ------- | ------------ | --------------------------------------------------------------------------------------------------- | --- |
| 编号  | 名称      | 内容           | 资源                                                                                                  | 备注  |
| 1   | AI 计算体系 | 01 课程内容      | [slide](./Foundation/01.introduction), [video](https://www.bilibili.com/video/BV1DX4y1D7PC/)        |     |
|     | AI 计算体系 | 02 AI计算模式(上) | [slide](./Foundation/02.constraints.pdf), [video](https://www.bilibili.com/video/BV17x4y1T7Cn/)     |     |
|     | AI 计算体系 | 03 AI计算模式(下) | [slide](./Foundation/03.mobile_parallel.pdf), [video](https://www.bilibili.com/video/BV1754y1M78X/) |     |
|     | AI 计算体系 | 04 关键设计指标    | [slide](./Foundation/04.metrics.pdf), [video](https://www.bilibili.com/video/BV1qL411o7S9/)         |     |
|     | AI 计算体系 | 05 核心计算：矩阵乘  | [slide](./Foundation/05.matrix.pdf), [video](https://www.bilibili.com/video/BV1ak4y1h7mp/)                     |     |
|     | AI 计算体系 | 06 数据单位：bits | [slide](./Foundation/06.bit_width.pdf), [video](https://www.bilibili.com/video/BV1WT411k724/)                  |     |
|     | AI 计算体系 | 07 AI计算体系总结  | [slide](./Foundation/07.summary.pdf), [video](https://www.bilibili.com/video/BV1j54y1T7ii/)                    |     |
|     |         |              |                                                                                                     |     |
|     | AI 芯片基础 | 01 CPU 基础  | [slide](./ChipBase/01.cpu_base.pdf), [video](https://www.bilibili.com/video/BV1tv4y1V72f/)                    |     |
|     | AI 芯片基础 | 02 ISA 指令架构  | [slide](./ChipBase/02.cpu_isa.pdf), [video](https://www.bilibili.com/video/BV1ro4y1W7xN/)                    |     |

## 目标学员

1. 对人工智能、深度学习感兴趣的学员

2. 渴望学习当今最热门最前沿AI技术的学员

3. 想储备深度学习技能的学员

4. AI系统开发工程师