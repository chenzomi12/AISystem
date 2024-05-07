<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# 前端优化

随着深度学习的不断发展，AI 模型结构在快速演化，底层计算硬件技术更是层出不穷，对于广大开发者来说不仅要考虑如何在复杂多变的场景下有效的将算力发挥出来，还要应对计算框架的持续迭代。AI 编译器就成了应对以上问题广受关注的技术方向，让用户仅需专注于上层模型开发，降低手工优化性能的人力开发成本，进一步压榨硬件性能空间。

AI 编译器主要是分为前端优化、后端优化，部分还会有中间优化层，而这里面主要介绍 AI 编译器的前端优化涉及到的算法和优化 Pass。

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

**内容大纲**

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/AISystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 小节 | 链接|
|:--:|:--:|
| 01 内容介绍| [PPT](./01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1ne411w7n2/), [文章](./01.introduction.md), [字幕](./srt/01.srt) |
| 02 计算图层 IR| [PPT](./02.graph_ir.pdf), [视频](https://www.bilibili.com/video/BV1kV4y1w72W/), [文章](./02.graph_ir.md), [字幕](./srt/02.srt) |
| 03 算子融合策略| [PPT](./03.op_fusion.pdf), [视频](https://www.bilibili.com/video/BV1P24y1D7RV/), [文章](./03.op_fusion.md), [字幕](./srt/03.srt) |
| 04 布局转换原理 | [PPT](./04.layout_princ.pdf), [视频](https://www.bilibili.com/video/BV1xK411z7Uw/), [文章](./04.layout_princ.md), [字幕](./srt/04.srt) |
| 05 布局转换算法 | [PPT](./05.layout_algo.pdf), [视频](https://www.bilibili.com/video/BV1gd4y1Y7dc/), [文章](./05.layout_algo.md), [字幕](./srt/05.srt) |
| 06 内存分配算法| [PPT](./06.memory.pdf), [视频](https://www.bilibili.com/video/BV1nM411879s/), [文章](./06.memory.md), [字幕](./srt/06.srt) |
| 07 常量折叠原理| [PPT](./07.constant_fold.pdf), [视频](https://www.bilibili.com/video/BV1P8411W7dY/), [文章](./07.constant_fold.md), [字幕](./srt/07.srt) |
| 08 公共表达式消除 | [PPT](./08.cse.pdf), [视频](https://www.bilibili.com/video/BV1rv4y1Q7tp/), [文章](./08.cse.md), [字幕](./srt/08.srt) |
| 09 死代码消除 | [PPT](./09.dce.pdf), [视频](https://www.bilibili.com/video/BV1hD4y1h7nh/), [文章](./09.dce.md), [字幕](./srt/09.srt) |
| 10 代数简化| [PPT](./10.algebraic.pdf), [视频](https://www.bilibili.com/video/BV1g24y1Q7qC/), [文章](./10.algebraic.md), [字幕](./srt/10.srt) |
| 11 优化 Pass 排序| [PPT](./11.summary.pdf), [视频](https://www.bilibili.com/video/BV1L14y1P7ku/), [文章](./11.summary.md), [字幕](./srt/11.srt) |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AISys](https://chenzomi12.github.io/) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT 开源在[github](https://github.com/chenzomi12/AISystem)，欢迎取用！！！

> 非常希望您也参与到这个开源项目中，B 站给 ZOMI 留言哦！
>
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交代码 PR 到开源社区哦！
>
> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！
