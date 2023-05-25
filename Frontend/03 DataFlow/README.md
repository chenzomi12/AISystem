# Computational Graph 计算图

为了高效地训练一个复杂神经网络，框架需要解决诸多问题， 例如：如何实现自动求导，如何利用编译期分析对神经网络计算进行化简、合并、变换，如何规划基本计算单元在加速器上的执行，如何将基本处理单元派发（dispatch）到特定的高效后端实现，如何进行内存预分配和管理等。用统一的方式解决这些问题都驱使着框架设计者思考如何为各类神经网络计算提供统一的描述，从而使得在运行神经网络计算之前，能够对整个计算过程尽可能进行推断，在编译期自动为用户程序补全反向计算，规划执行，最大程度地降低运行时开销。目前主流的深度学习框架都选择使用计算图来抽象神经网络计算，【计算图】系列展示了基于深度学习框架/AI框架计算图的核心内容。

## 内容大纲

> *建议优先下载或者使用PDF版本，PPT版本会因为字体缺失等原因导致版本很丑哦~*

|     |     |                 |                                                                                             |
| --- | --- | --------------- | ------------------------------------------------------------------------------------------- |
| 编号  | 名称  | 名称              | 备注                                                                                          |
|     | 计算图 | 01 基本介绍         | [silde](./01.introduction.pdf), [video](https://www.bilibili.com/video/BV1cG411E7gV/)      |
|     | 计算图 | 02 什么是计算图       | [silde](./02.computation_graph.pdf), [video](https://www.bilibili.com/video/BV1rR4y197HM/) |
|     | 计算图 | 03 计算图跟自动微分关系   | [silde](./03.atuodiff.pdf), [video](https://www.bilibili.com/video/BV1S24y197FU/)          |
|     | 计算图 | 04 图优化与图执行调度    | [silde](./04.dispatch.pdf),[video](https://www.bilibili.com/video/BV1hD4y1k7Ty/)           |
|     | 计算图 | 05 计算图的控制流机制实现  | [silde](./05.control_flow.pdf),[video](https://www.bilibili.com/video/BV17P41177Pk/)       |
|     | 计算图 | 06 计算图未来将会走向何方？ | [silde](./06.future.pdf),[video](https://www.bilibili.com/video/BV1hm4y1A7Nv/)             |
|     |     |                 |                                                                                             |