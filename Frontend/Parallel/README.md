# 分布式并行

什么是大模型？大模型模型参数量实在太大，需要分布式并行训练能力一起来加速训练过程。分布式并行是在大规模AI集群上工作的，想要加速就需要软硬件协同，不仅仅要解决通信拓扑的问题、集群组网的问题，还要了解上层MOE、Transform等新兴算法。通过对算法的剖析，提出模型并行、数据并行、优化器并行等新的并行模式和通信同步模式，来加速分布式训练的过程。最小的单机执行单元里面，还要针对大模型进行混合精度、梯度累积等算法，进一步压榨集群的算力！

## 内容大纲

| 编号  | 名称    | 名称                   | 备注                                                                                                                  |
| --- | ----- | -------------------- | ------------------------------------------------------------------------------------------------------------------- |
|     |        |                   |                                                                                                                                                               |
| 6   | 分布式并行  | 01 基本介绍           | [silde](./01.introduction.pptx), [video](https://www.bilibili.com/video/BV1ve411w7DL/)   |
|     | 分布式并行  | 02 数据并行           | [silde](./02.data_parallel.pptx), [video](https://www.bilibili.com/video/BV1JK411S7gL/)                                                 |
|     | 分布式并行  | 03 模型并行之张量并行      | [silde](./03.tensor_parallel.pptx), [video](https://www.bilibili.com/video/BV1vt4y1K7wT/)                                               |
|     | 分布式并行  | 04 MindSpore张量并行  | [silde](./04.mindspore_parallel.pptx), [video](https://www.bilibili.com/video/BV1vt4y1K7wT/)                                            |
|     | 分布式并行  | 05 模型并行之流水并行      | [silde](./05.pipeline_parallel.pptx), [video](https://www.bilibili.com/video/BV1WD4y1t7Ba/)                                             |
|     | 分布式并行  | 06 混合并行           | [silde](./06.hybrid_parallel.pptx), [video](https://www.bilibili.com/video/BV1gD4y1t7Ut/)                                               |
|     | 分布式汇总  | 07 分布式训练总结        | [silde](./07.summary.pptx), [video](https://www.bilibili.com/video/BV1av4y1S7DQ/)                                                       |
|     |        |                   |                                                                                                                                                               |