# Auto Diff 自动微分

02 自动微分原理文章中我们大概初步谈了谈从手动微分到自动微分的过程，03 自动微分正反模式中深入了自动微分的正反向模式具体公式和推导。实际上 02 了解到正反向模式只是自动微分的原理模式，在实际代码实现的过程，04 会通过三种实现方式（基于库、操作符重载、源码转换）来实现。05和06则是具体跟大家一起手把手实现一个类似于PyTorch的自动微分框架。07最后做个小小的总结，一起review自动微分面临易用性、性能的挑战，最后在可微分编程方面畅享了下未来。

## 内容大纲

|||||
|---|---|---|---|
|编号|名称|名称|备注|
|1|自动微分|01 基本介绍|[silde](./01.introduction.pptx), [video](https://www.bilibili.com/video/BV1FV4y1T7zp/), [article](https://zhuanlan.zhihu.com/p/518198564)|
| |自动微分|02 什么是微分|[silde](./02.base_concept.pptx), [video](https://www.bilibili.com/video/BV1Ld4y1M7GJ/), [article](https://zhuanlan.zhihu.com/p/518198564)|
| |自动微分|03 正反向计算模式|[silde](./03.grad_mode.pptx), [video](https://www.bilibili.com/video/BV1zD4y117bL/), [article](https://zhuanlan.zhihu.com/p/518296942)|
| |自动微分|04 三种实现方法|[silde](./04.grad_mode.pptx), [video](https://www.bilibili.com/video/BV1BN4y1P76t/), [article](https://zhuanlan.zhihu.com/p/520065656)|
| |自动微分|05 手把手实现正向微分框架|[silde](./05.forward_mode.ipynb), [video](https://www.bilibili.com/video/BV1Ne4y1p7WU/), [article](https://zhuanlan.zhihu.com/p/520451681)|
| |自动微分|06 亲自实现一个PyTorch|[silde](./06.reversed_mode.ipynb), [video](https://www.bilibili.com/video/BV1ae4y1z7E6/), [article](https://zhuanlan.zhihu.com/p/547865589)|
| |自动微分|07 自动微分的挑战&未来|[silde](./07.challenge.pptx), [video](https://www.bilibili.com/video/BV17e4y1z73W/)|
|||||