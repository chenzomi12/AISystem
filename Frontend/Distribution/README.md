# 分布式训练

什么是大模型？大模型模型参数量实在太大，需要分布式并行训练能力一起来加速训练过程。分布式并行是在大规模AI集群上工作的，想要加速就需要软硬件协同，不仅仅要解决通信拓扑的问题、集群组网的问题，还要了解上层MOE、Transform等新兴算法。通过对算法的剖析，提出模型并行、数据并行、优化器并行等新的并行模式和通信同步模式，来加速分布式训练的过程。最小的单机执行单元里面，还要针对大模型进行混合精度、梯度累积等算法，进一步压榨集群的算力！

## 内容大纲

| 编号  | 名称    | 名称                   | 备注                                                                                                                  |
| --- | ----- | -------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 4   | 分布式训练 | 01 基本介绍              | [silde](./Frontend/Distribution/01.introduction.pptx), [video](https://www.bilibili.com/video/BV1ge411L7mi/)        |
|     | 分布式训练 | 02 大模型训练的挑战          | [silde](./Frontend/Distribution/02.challenge.pptx), [video](https://www.bilibili.com/video/BV1n8411s7f3/)           |
|     | 分布式训练 | 03 AI框架分布式功能         | [silde](./Frontend/Distribution/03.system.pptx), [video](https://www.bilibili.com/video/BV1Y14y1576A/)              |
|     | 分布式训练 | 04 AI集群服务器架构         | [silde](./Frontend/Distribution/04.architecture.pptx), [video](https://www.bilibili.com/video/BV1fg41187rc/)        |
|     | 分布式训练 | 05(上) 通信：AI集群软硬件通信   | [silde](./Frontend/Distribution/05.1.communication.pptx), [video](https://www.bilibili.com/video/BV14P4y1S7u4/)     |
|     | 分布式训练 | 05(下) 通信：集合通信原语      | [silde](./Frontend/Distribution/05.2.primitive.pptx), [video](https://www.bilibili.com/video/BV1te4y1e7vz/)         |
|     | 分布式训练 | 06(上) 算法：大模型算法结构     | [silde](./Frontend/Distribution/06.algorithm_arch.pptx), [video](https://www.bilibili.com/video/BV1Mt4y1M7SE/)      |
|     | 分布式训练 | 06(下) 算法：亿级规模SOTA大模型 | [silde](./Frontend/Distribution/06.algorithm_arch.pptx), [video](https://www.bilibili.com/video/BV1em4y1F7ay/)      |
|     | 分布式训练 | 07(上) 并行策略：数据并行      | [silde](./Frontend/Distribution/07.1.data_parallel.pptx), [video](https://www.bilibili.com/video/BV1JK411S7gL/)     |
|     | 分布式训练 | 07(中) 并行策略：张量并行      | [silde](./Frontend/Distribution/07.2.model_parallel.pptx), [video](https://www.bilibili.com/video/BV1vt4y1K7wT/)    |
|     | 分布式训练 | 07(下) 并行策略：流水并行      | [silde](./Frontend/Distribution/07.3.pipeline_parallel.pptx), [video](https://www.bilibili.com/video/BV1WD4y1t7Ba/) |
|     | 分布式训练 | 08 混合并行              | [silde](./Frontend/Distribution/08.hybrid_parallel.pptx), [video](https://www.bilibili.com/video/BV1gD4y1t7Ut/)     |
|     | 分布式训练 | 09 分布式训练总结           | [silde](./Frontend/Distribution/10.summary.pptx), [video](https://www.bilibili.com/video/BV1av4y1S7DQ/)             |
