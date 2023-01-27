# 推理引擎

训练过程通过设定数据处理方式，并设计合适的网络模型结构以及损失函数和优化算法，在此基础上将数据集以小批量（mini-batch）反复进行前向计算并计算损失，然后反向计算梯度利用特定的优化函数来更新模型，来使得损失函数达到最优的结果。训练过程最重要的就是梯度的计算和反向传播。

而推理就是在训练好的模型结构和参数基础上，做一次前向传播得到模型输出的过程。相对于训练而言，推理不涉及梯度和损失优化。推理的最终目标是将训练好的模型部署生产环境中。真正让 AI 能够运用起来。推理引擎可以将深度学习模型部署到云（Cloud）端或者边缘（Edge）端，并服务用户的请求。模型训练过程好比是传统软件工程中的代码开发的过程，而开发完的代码势必要打包，部署给用户使用，那么推理系统就负责应对模型部署的生命周期中遇到的挑战和问题。

当推理系统将完成训练的模型进行部署，并在服务时还需要考虑设计和提供负载均衡，请求调度，加速优化，多副本和生命周期管理等支持。相比深度学习框架等为训练而设计的系统，推理系统不仅关注低延迟，高吞吐，可靠性等设计目标，同时受到资源，服务等级协议（Service-Level Agreement），功耗等约束。本章将围绕深度学习推理系统的设计，实现与优化内容展开，同时还会在最后介绍部署和 MLOps 等内容。

移动端的推理引擎应该挺多的了，google在2017年推出了TF-Lite，腾讯在2017年推出了ncnn，Apple在2017也推出了CoreML，阿里在2018年推出了MNN，华为2019年推出了MindSpsore-Lite。距今已经过去了快5年的时间，技术上也接近收敛。下面让我们一起打开推理引擎的技术吧！

## 课程部分

> *建议优先下载或者使用PDF版本，PPT版本会因为字体缺失等原因导致版本很丑哦~*

|     |         |                   |                                                                                                 |     |
| --- | ------- | ----------------- | ----------------------------------------------------------------------------------------------- | --- |
| 编号  | 名称      | 内容                | 资源                                                                                              | 备注  |
| 1   | 推理系统    | 01 内容介绍           | [slide](./Inference/01.introduction.pdf), [video](https://www.bilibili.com/video/BV1J8411K7pj/) |     |
|     | 推理系统    | 02 什么是推理系统        | [slide](./Inference/02.constraints.pdf), [video](https://www.bilibili.com/video/BV1nY4y1f7G5/)  |     |
|     | 推理系统    | 03 推理流程全景         | [slide](./Inference/03.workflow.pdf), [video](https://www.bilibili.com/video/BV1M24y1v7rK/)     |     |
|     | 推理系统    | 04 推理系统架构         | [slide](./Inference/04.system.pdf), [video](https://www.bilibili.com/video/BV1Gv4y1i7Tw/)       |     |
|     | 推理系统    | 05(上) 推理引擎架构      | [slide](./Inference/05.inference.pdf), [video](https://www.bilibili.com/video/BV1Mx4y137Er/)    |     |
|     | 推理系统    | 05(下) 推理引擎架构      | [slide](./Inference/06.architecture.pdf), [video](https://www.bilibili.com/video/BV1FG4y1C7Mn/) |     |
|     |         |                   |                                                                                                 |     |
| 2   | 模型小型化   | 01 推理参数了解         | [slide](./Mobilenet/01.introduction.pdf), [video](https://www.bilibili.com/video/BV1KW4y1G75J/) |     |
|     | 模型小型化   | 02(上) CNN模型小型化    | [slide](./Mobilenet/02.cnn.pdf), [video](https://www.bilibili.com/video/BV1Y84y1b7xj/)          |     |
|     | 模型小型化   | 02(下) CNN模型小型化    | [slide](./Mobilenet/02.cnn.pdf), [video](https://www.bilibili.com/video/BV1DK411k7qt/)          |     |
|     | 模型小型化   | 03 Transformer小型化 | [slide](./Mobilenet/03.transform.pdf), [video](https://www.bilibili.com/video/BV19d4y1V7ou/)    |     |
|     |         |                   |                                                                                                 |     |
| 3   | 模型压缩    | 01 基本介绍           | [slide](./Slim/01.introduction.pdf), [video](https://www.bilibili.com/video/BV1384y187tL/)      |     |
|     | 模型压缩    | 02 低比特量化原理        | [slide](./Slim/02.quant.pdf), [video](https://www.bilibili.com/video/BV1VD4y1n7AR/)             |     |
|     | 模型压缩    | 03 感知量化训练 QAT     | [slide](./Slim/03.qat.pdf), [video](https://www.bilibili.com/video/BV1s8411w7b9/)               |     |
|     | 模型压缩    | 04 训练后量化PTQ与部署    | [slide](./Slim/04.ptq.pdf), [video](https://www.bilibili.com/video/BV1HD4y1n7E1/)               |     |
|     | 模型压缩    | 05 模型剪枝           | [slide](./Slim/05.pruning.pdf), [video]()                                                       |     |
|     | 模型压缩    | 06(上) 知识蒸馏原理      | [slide](./Slim/06.distillation.pdf), [video]()                                                  |     |
|     | 模型压缩    | 06(下) 知识蒸馏算法      | [slide](./Slim/06.distillation.pdf), [video]()                                                  |     |
|     |         |                   |                                                                                                 |     |
| 4   | 模型转换与优化 | 01 基本介绍           | [slide](./Converter/01.introduction.pdf), [video]                                               |     |
|     | 模型转换模块  | 02 架构与文件格式      | [slide](./Converter/02.converter_princ.pdf)                                                     |     |
|     | 模型转换模块  | 03 推理引擎IR         | [slide](./Converter/03.converter_ir.pdf)                                                        |     |
|     | 模型转换模块  | 04 流程细节           | [slide](./Converter/04.converter_detail.pdf)                                                    |     |
|     |         |                   |                                                                                                 |     |
|     | 更新中ing  |                   |                                                                                                 |     |
|     |         |                   |                                                                                                 |     |
