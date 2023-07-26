<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# 模型转换&优化

《模型转换&优化》在这一节当中分为模型转换和模型优化，在整体架构图中属于离线模型转换模块。一方面，推理引擎需要把不同 AI 框架训练得到的模型进行转换；另外一方面需要对转换后的模型进行图优化等技术。

> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！

**内容大纲**

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 模型转换&优化| 01 基本介绍 | [PPT](./01.introduction.pdf), [视频](https://www.bilibili.com/video/BV1724y1z7ep/) |
| 模型转换模块 | 02 架构与文件格式| [PPT](./02.converter_princ.pdf), [视频](https://www.bilibili.com/video/BV13P4y167sr/) |
| 模型转换模块 | 03 自定义计算图IR | [PPT](./03.converter_ir.pdf), [视频](https://www.bilibili.com/video/BV1rx4y177R9/) |
| 模型转换模块 | 04 流程细节 | [PPT](./04.converter_detail.pdf), [视频](https://www.bilibili.com/video/BV13341197zU/) |
| 图优化模块| 05 计算图优化策略| [PPT](./05.optimizer.pdf), [视频](https://www.bilibili.com/video/BV1g84y1L7tF/) |
| 图优化模块| 06 常量折叠&冗余节点消除| [PPT](./06.basic.pdf), [视频](https://www.bilibili.com/video/BV1fA411r7hr/) |
| 图优化模块| 07 算子融合/替换/前移 | [PPT](./06.basic.pdf), [视频](https://www.bilibili.com/video/BV1Qj411T7Ef/) |
| 图优化模块| 08 数据布局转换&内存优化| [PPT](./07.extend.pdf), [视频](https://www.bilibili.com/video/BV1Ae4y1N7u7/) |

```toc
:maxdepth: 2

01.introduction
02.converter_princ
03.converter_ir
04.converter_detail
05.optimizer
06.basic
07.extend
```
