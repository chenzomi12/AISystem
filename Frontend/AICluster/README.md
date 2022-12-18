# 分布式训练

什么是大模型？大模型模型参数量实在太大，需要分布式并行训练能力一起来加速训练过程。分布式并行是在大规模AI集群上工作的，想要加速就需要软硬件协同，不仅仅要解决通信拓扑的问题、集群组网的问题，还要了解上层MOE、Transform等新兴算法。通过对算法的剖析，提出模型并行、数据并行、优化器并行等新的并行模式和通信同步模式，来加速分布式训练的过程。最小的单机执行单元里面，还要针对大模型进行混合精度、梯度累积等算法，进一步压榨集群的算力！

## 内容大纲

| 编号  | 名称    | 名称                | 备注                                                                                                             |
| --- | ----- | ----------------- | -------------------------------------------------------------------------------------------------------------- |
|     |       |                   |                                                                                                      |
| 4   | 分布式集群 | 01 基本介绍           | [silde](./01.introduction.pptx), [video](https://www.bilibili.com/video/BV1ge411L7mi/)   |
|     | 分布式集群 | 02 AI集群服务器架构      | [silde](./02.architecture.pptx), [video](https://www.bilibili.com/video/BV1fg41187rc/)   |
|     | 分布式集群 | 03 AI集群软硬件通信      | [silde](./03.communication.pptx), [video](https://www.bilibili.com/video/BV14P4y1S7u4/)  |
|     | 分布式集群 | 04 集合通信原语         | [silde](./04.primitive.pptx), [video](https://www.bilibili.com/video/BV1te4y1e7vz/)      |
|     | 分布式算法 | 05 AI框架分布式功能      | [silde](./05.system.pptx), [video](https://www.bilibili.com/video/BV1n8411s7f3/)         |
|     |       |                   |                                                                                                                |
| 5   | 分布式算法 | 06 大模型训练的挑战       | [silde](./06.challenge.pptx), [video](https://www.bilibili.com/video/BV1Y14y1576A/)      |
|     | 分布式算法 | 07 算法：大模型算法结构     | [silde](./07.algorithm_arch.pptx), [video](https://www.bilibili.com/video/BV1Mt4y1M7SE/) |
|     | 分布式算法 | 08 算法：亿级规模SOTA大模型 | [silde](./08.algorithm_sota.pptx), [video](https://www.bilibili.com/video/BV1em4y1F7ay/) |
|     |       |                   |                                                                                                                |