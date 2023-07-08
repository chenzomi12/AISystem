<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# Kernel优化

在上层应用或者 AI 网络模型中，看到的是算子；但是在推理引擎实际执行的是具体的 Kernel，而推理引擎中 CNN 占据了主要是得执行时间，因此其 Kernel 优化尤为重要。

我在这里抛砖引玉，希望您可以一起参与到这个开源项目中，跟更多的您一起探讨学习！

## 内容大纲

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| Kernel优化 | 01 Kernel优化架构 | [slide](./01.introduction.pdf), [video](https://www.bilibili.com/video/BV1Ze4y1c7Bb/) |
| Kernel优化 | 02 卷积操作基础原理 | [slide](./02.conv.pdf),[video](https://www.bilibili.com/video/BV1No4y1e7KX/) |
| Kernel优化 | 03 Im2Col算法 | [slide](./03.im2col.pdf),[video](https://www.bilibili.com/video/BV1Ys4y1o7XW/) |
| Kernel优化 | 04 Winograd算法 | [slide](./04.winograd.pdf),[video](https://www.bilibili.com/video/BV1vv4y1Y7sc/) |
| Kernel优化 | 05 QNNPack算法| [slide](./05.qnnpack.pdf),[video](https://www.bilibili.com/video/BV1ms4y1o7ki/) |
| Kernel优化 | 06 推理内存布局 | [slide](./06.memory.pdf),[video](https://www.bilibili.com/video/BV1eX4y1X7mL/) |
| Kernel优化 | 07 nc4hw4内存排布 | [slide](./07.nc4hw4.pdf) |
| Kernel优化 | 08 汇编与循环优化| [slide](./08.others.pdf) |

```toc
:maxdepth: 2

01.introduction
02.conv
03.im2col
04.winograd
05.qnnpack
06.memory
07.nc4hw4
08.others
```
