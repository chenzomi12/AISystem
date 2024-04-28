<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# Kernel 优化

在上层应用或者 AI 网络模型中，看到的是算子；但是在推理引擎实际执行的是具体的 Kernel，而推理引擎中 CNN 占据了主要是得执行时间，因此其 Kernel 优化尤为重要。

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

**内容大纲**

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/AISystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| Kernel 优化 | 01 Kernel 优化架构 | [slide](./01.introduction.pdf), [video](https://www.bilibili.com/video/BV1Ze4y1c7Bb/) |
| Kernel 优化 | 02 卷积操作基础原理 | [slide](./02.conv.pdf),[video](https://www.bilibili.com/video/BV1No4y1e7KX/) |
| Kernel 优化 | 03 Im2Col 算法 | [slide](./03.im2col.pdf),[video](https://www.bilibili.com/video/BV1Ys4y1o7XW/) |
| Kernel 优化 | 04 Winograd 算法 | [slide](./04.winograd.pdf),[video](https://www.bilibili.com/video/BV1vv4y1Y7sc/) |
| Kernel 优化 | 05 QNNPack 算法| [slide](./05.qnnpack.pdf),[video](https://www.bilibili.com/video/BV1ms4y1o7ki/) |
| Kernel 优化 | 06 推理内存布局 | [slide](./06.memory.pdf),[video](https://www.bilibili.com/video/BV1eX4y1X7mL/) |
| Kernel 优化 | 07 nc4hw4 内存排布 | [slide](./07.nc4hw4.pdf) |
| Kernel 优化 | 08 汇编与循环优化| [slide](./08.others.pdf) |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AISys](https://chenzomi12.github.io/) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT 开源在[github](https://github.com/chenzomi12/AISystem)，欢迎取用！！！

> 非常希望您也参与到这个开源项目中，B 站给 ZOMI 留言哦！
>
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交代码 PR 到开源社区哦！
>
> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！
