<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# 前端优化

《前端优化》前端优化作为 AI编译器 的整体架构主要模块，主要优化的对象是计算图，而计算图是通过AI框架产生的，值得注意的是并不是所有的AI框架都会生成计算图，有了计算图就可以结合深度学习的原理知识进行图的优化。前端优化包括图算融合、数据排布、内存优化等跟深度学习相结合的优化手段，同时把传统编译器关于代数优化的技术引入到计算图中。

> 我在这里抛砖引玉，希望跟更多的您一起探讨学习！

```toc
:maxdepth: 2

01.introduction
02.torchscript
03.torchfx_lazy
04.torchdynamo
05.aotatuograd
06.dispatch
```