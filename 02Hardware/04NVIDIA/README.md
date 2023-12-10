# NVIDIA GPU详解

- 《NVIDIA GPU原理》英伟达架构里面专门为AI而生的 Tensor Core 和 NVLink 对AI加速尤为重要，因此重点对 Tensor Core 和 NVLink 进行深入剖析其发展、演进和架构。

> 希望这个系列能够给朋友们带来一些帮助，也希望ZOMI能够继续坚持完成所有内容哈！欢迎您也参与到这个开源项目的贡献！

**内容大纲**

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| GPU 原理详解 | 01 TensorCore原理(上) | [slide](./01.basic_tc.pdf), [video](https://www.bilibili.com/video/BV1aL411a71w/)|
| GPU 原理详解 | 02 TensorCore架构(中) | [slide](./02.history_tc.pdf), [video](https://www.bilibili.com/video/BV1pL41187FH/)|
| GPU 原理详解 | 03 TensorCore剖析(下) | [slide](./03.deep_tc.pdf), [video](https://www.bilibili.com/video/BV1oh4y1J7B4/) |
| GPU 原理详解 | 04 分布式通信与NVLink| [slide](./04.basic_nvlink.pdf), [video](https://www.bilibili.com/video/BV1cV4y1r7Rz/)|
| GPU 原理详解 | 05 NVLink原理剖析| [slide](./05.deep_nvlink.pdf), [video](https://www.bilibili.com/video/BV1uP411X7Dr/) |
| GPU 原理详解 | 05 NVSwitch原理剖析| [slide](./06.deep_nvswitch.pdf), [video](https://www.bilibili.com/video/BV1uM4y1n7qd/) |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AISys](https://chenzomi12.github.io/) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT开源在[github](https://github.com/chenzomi12/DeepLearningSystem)，欢迎取用！！！

> 非常希望您也参与到这个开源项目中，B站给ZOMI留言哦！
>
> 欢迎大家使用的过程中发现bug或者勘误直接提交代码PR到开源社区哦！
>
> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！
