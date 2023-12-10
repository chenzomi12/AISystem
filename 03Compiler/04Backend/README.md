<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# 后端优化

《编译后端优化》后端优化作为AI编译器跟硬件之间的相连接的模块，更多的是算子或者Kernel进行优化，而优化之前需要把计算图转换称为调度树等IR格式，然后针对每一个算子/Kernel进行循环优化、指令优化和内存优化等技术。

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

**内容大纲**

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 小节 | 链接|
|:--|:--|
| 01 AI编译器后端优化介绍 | [PPT](./01.introduction.pdf), [视频](https://www.bilibili.com/video/BV17D4y177bP/), [文章](./01.introduction.md), [字幕](./srt/01.srt) |
| 02 算子分为计算与调度 | [PPT](./02.ops_compute.pdf), [视频](https://www.bilibili.com/video/BV1K84y1x7Be/), [文章](./02.ops_compute.md), [字幕](./srt/02.srt) |
| 03 算子优化手工方式| [PPT](./03.optimization.pdf), [视频](https://www.bilibili.com/video/BV1ZA411X7WZ/), [文章](./03.optimization.md), [字幕](./srt/03.srt) |
| 04 算子循环优化| [PPT](./04.loop_opt.pdf), [视频](https://www.bilibili.com/video/BV17D4y177bP/), [文章](./04.loop_opt.md), [字幕](./srt/04.srt) |
| 05 指令和内存优化 | [PPT](./05.other_opt.pdf), [视频](https://www.bilibili.com/video/BV11d4y1a7J6/), [文章](./05.other_opt.md), [字幕](./srt/05.srt) |
| 06 Auto-Tuning原理 | [PPT](./06.auto_tuning.pdf), [视频](https://www.bilibili.com/video/BV1uA411D7JF/), [文章](./06.auto_tuning.md), [字幕](./srt/05.srt) |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AISys](https://chenzomi12.github.io/) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT开源在[github](https://github.com/chenzomi12/DeepLearningSystem)，欢迎取用！！！

> 非常希望您也参与到这个开源项目中，B站给ZOMI留言哦！
>
> 欢迎大家使用的过程中发现bug或者勘误直接提交代码PR到开源社区哦！
>
> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！
