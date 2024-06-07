<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# 模型优化(DONE)

《模型优化》在这一节当中分为模型转换和模型优化，在整体架构图中属于离线模型转换模块。一方面，推理引擎需要把不同 AI 框架训练得到的模型进行转换；另外一方面需要对转换后的模型进行图优化等技术。

## 内容大纲

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/AISystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 图优化模块| 01 计算图优化策略| [文章](./01Optimizer.md), [PPT](./05Optimizer.pdff), [视频](https://www.bilibili.com/video/BV1g84y1L7tF/) |
| 图优化模块| 02 常量折叠&冗余节点消除| [文章](./02Basic.md), [PPT](./06Basic.pdf), [视频](https://www.bilibili.com/video/BV1fA411r7hr/) |
| 图优化模块| 03 算子融合/替换/前移 | [文章](./02Basic.md), [PPT](./06Basic.pdf), [视频](https://www.bilibili.com/video/BV1Qj411T7Ef/) |
| 图优化模块| 04 数据布局转换&内存优化| [文章](./03Extend.md), [PPT](./07Extend.pdf), [视频](https://www.bilibili.com/video/BV1Ae4y1N7u7/) |

## 备注

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT 开源在[github](https://github.com/chenzomi12/AISystem)，欢迎取用！！！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
> 
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交代码 PR 到开源社区哦！
>
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交 PR 到开源社区哦！
>
> 请大家尊重开源和 ZOMI 的努力，引用 PPT 的内容请规范转载标明出处哦！