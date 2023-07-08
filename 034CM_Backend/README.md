<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# 编译后端优化

《编译后端优化》后端优化作为AI编译器跟硬件之间的相连接的模块，更多的是算子或者Kernel进行优化，而优化之前需要把计算图转换称为调度树等IR格式，然后针对每一个算子/Kernel进行循环优化、指令优化和内存优化等技术。

我在这里抛砖引玉，希望您可以一起参与到这个开源项目中，跟更多的您一起探讨学习！

## 内容大纲

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 名称 | 名称 | 备注|
|:--:|:--:|:--:|
| 后端优化 | 01 AI编译器后端优化介绍 | [PPT](./01.introduction.pdf), [视频](https://www.bilibili.com/video/BV17D4y177bP/), [字幕](./srt/01.srt) |
| 后端优化 | 02 算子分为计算与调度 | [PPT](./02.ops_compute.pdf), [视频](https://www.bilibili.com/video/BV1K84y1x7Be/), [字幕](./srt/02.srt) |
| 后端优化 | 03 算子优化手工方式| [PPT](./03.optimization.pdf), [视频](https://www.bilibili.com/video/BV1ZA411X7WZ/), [字幕](./srt/03.srt) |
| 后端优化 | 04 算子循环优化| [PPT](./04.loop_opt.pdf), [视频](https://www.bilibili.com/video/BV17D4y177bP/), [字幕](./srt/04.srt) |
| 后端优化 | 05 指令和内存优化 | [PPT](./05.other_opt.pdf), [视频](https://www.bilibili.com/video/BV11d4y1a7J6/), [字幕](./srt/05.srt) |
| 后端优化 | 06 Auto-Tuning原理 | [PPT](./06.auto_tuning.pdf), [视频](https://www.bilibili.com/video/BV1uA411D7JF/), [字幕](./srt/05.srt) |

```toc
:maxdepth: 2

01.introduction
02.ops_compute
03.optimization
04.loop_opt
05.other_opt
06.auto_tuning
```
