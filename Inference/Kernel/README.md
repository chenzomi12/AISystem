# Kernel优化

在上层应用或者 AI 网络模型中，看到的是算子；但是在推理引擎实际执行的是具体的 Kernel，而推理引擎中 CNN 占据了主要是得执行时间，因此其 Kernel 优化尤为重要。

## 课程部分

> *建议优先下载或者使用PDF版本，PPT版本会因为字体缺失等原因导致版本很丑哦~*

| 名称       | 内容            | 资源                                                                                           | 备注  |
| -------- | ------------- | -------------------------------------------------------------------------------------------- | --- |
|          |               |                                                                                              |     |
| Kernel优化 | 01 Kernel优化架构 | [slide](./01.introduction.pdf), [video](https://www.bilibili.com/video/BV1Ze4y1c7Bb/) |     |
| Kernel优化 | 02 卷积操作基础原理   | [slide](./02.conv.pdf),[video](https://www.bilibili.com/video/BV1No4y1e7KX/)          |     |
| Kernel优化 | 03 Im2Col算法   | [slide](./03.im2col.pdf),[video](https://www.bilibili.com/video/BV1Ys4y1o7XW/)        |     |
| Kernel优化 | 04 Winograd算法 | [slide](./04.winograd.pdf),[video](https://www.bilibili.com/video/BV1vv4y1Y7sc/)      |     |
| Kernel优化 | 05 QNNPack算法  | [slide](./05.qnnpack.pdf),[video](https://www.bilibili.com/video/BV1ms4y1o7ki/)       |     |
| Kernel优化 | 06 推理内存布局     | [slide](./06.memory.pdf),  [video](https://www.bilibili.com/video/BV1eX4y1X7mL/)      |     |
| Kernel优化 | 07 nc4hw4内存排布 | [slide](./07.nc4hw4.pdf),                                                   |     |
| Kernel优化 | 08 汇编与循环优化    | [slide](./08.others.pdf),                                                  |     |
