<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# Kernel优化

《Kernel优化》在上层应用或者 AI 网络模型中，看到的是算子；但是在推理引擎实际执行的是具体的 Kernel，而推理引擎中 CNN 占据了主要是得执行时间，因此其 Kernel 优化尤为重要。

> 我在这里抛砖引玉，希望跟更多的您一起探讨学习！

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