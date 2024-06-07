<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# 模型轻量化

轻量化模型，其实也是模型小型化的一种方式。主要思想是针对神经网络模型设计更高效的网络计算方式，从而使神经网络模型的参数量减少的同时，不损失网络精度，并进一步提高模型的执行效率。推理引擎之模型小型化，主要集中介绍模型小型化中需要注意的参数和指标，接着深入了解 CNN 经典的轻量化模型和 Transformer 结构的轻量化模型。

## 内容大纲

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/AISystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 模型小型化 | 01 推理参数了解 | [文章](./01Introduction.md), [PPT](./01Introduction.pdf), [视频](https://www.bilibili.com/video/BV1KW4y1G75J/) |
| 模型小型化 | 02(上) CNN 模型小型化 | [PPT](./02Cnn.pdf), [视频](https://www.bilibili.com/video/BV1Y84y1b7xj/) |
| 模型小型化 | 02(下) CNN 模型小型化 | [PPT](./02Cnn.pdf), [视频](https://www.bilibili.com/video/BV1DK411k7qt/) |
| 模型小型化 | SqueezeNet 系列 | [文章](./02Squeezenet.md) |
| 模型小型化 | ShuffleNet 系列 | [文章](./03Shufflenet.md) |
| 模型小型化 | MobileNet 系列 | [文章](./04Mobilenet.md) |
| 模型小型化 | 03 Transformer 小型化 | [PPT](./03Transform.pdf), [视频](https://www.bilibili.com/video/BV19d4y1V7ou/) |

## 备注

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT 开源在[github](https://github.com/chenzomi12/AISystem)，欢迎取用！！！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
> 
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交代码 PR 到开源社区哦！
>
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交 PR 到开源社区哦！
>
> 请大家尊重开源和 ZOMI 的努力，引用 PPT 的内容请规范转载标明出处哦！

## 参考文献

1.[Khalid Ashraf, Bichen Wu, Forrest N. Iandola, Matthew W. Moskewicz, and Kurt Keutzer. Shallow networks for high-accuracy road object-detection. arXiv:1606.01561, 2016.](https://arxiv.org/abs/1606.01561v1)

2.[Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. SegNet: A deep convolutional encoderdecoder architecture for image segmentation. arxiv:1511.00561, 2015.](https://arxiv.org/pdf/1807.10221v1.pdf)

3.[Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu,
Chiyuan Zhang, and Zheng Zhang. Mxnet: A flexible and efficient machine learning library for
heterogeneous distributed systems. arXiv:1512.01274, 2015a.](https://arxiv.org/abs/1512.01274)

4.[Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, and Trevor
Darrell. Decaf: A deep convolutional activation feature for generic visual recognition.
arXiv:1310.1531, 2013.](https://arxiv.org/abs/1310.1531v1)

5.[Song Han, Jeff Pool, Sharan Narang, Huizi Mao, Shijian Tang, Erich Elsen, Bryan Catanzaro, John Tran, and William J. Dally. Dsd: Regularizing deep neural networks with dense-sparse-dense training flow. arXiv:1607.04381, 2016b](https://arxiv.org/abs/1607.04381v1)

6.[C. Farabet, B. Martini, B. Corda, P. Akselrod, E. Culurciello, and Y. LeCun. Neuflow: A runtime reconfigurable dataflow processor for vision. In Computer Vision and Pattern Recognition Workshops (CVPRW),2011 IEEE Computer Society Conference on, pages109–116, 2011.](https://ieeexplore.ieee.org/document/5981829/)

7.[M. Jaderberg, A. Vedaldi, and A. Zisserman. Speeding up convolutional neural networks with low rank expansions. arXiv preprint arXiv:1405.3866, 2014.](https://arxiv.org/pdf/1405.3866.pdf)

8.[M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi.Xnor-net: Imagenet classification using binary convolutional neural networks. In European Conference on Computer Vision, pages 525–542, 2016.](http://allenai.org/plato/xnornet)

9.[ S. Williams, A. Waterman, and D. Patterson. Roofline:an insightful visual performance model for multicore architectures. Communications of the ACM, 52(4):65–76, 2009.](https://dl.acm.org/doi/10.1145/1498765.1498785)

10.[B. Wu, A. Wan, X. Yue, and K. Keutzer. Squeezeseg: Convolutional neural nets with recurrent crf for real-time road-object segmentation from 3d lidar point cloud. arXiv preprint arXiv:1710.07368, 2017.](https://arxiv.org/abs/1710.07368)

11.[K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition.arXiv preprint arXiv:1409.1556, 2014.](https://arxiv.org/abs/1409.1556)

12.[K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.](https://arxiv.org/abs/1512.03385)

13.[S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine
learning, pages 448–456, 2015.](https://arxiv.org/abs/1502.03167v3)

14.[S. Han, H. Mao, and W. J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. International Conference on Learning Representations(ICLR), 2016.](https://arxiv.org/pdf/1510.00149.pdf)

15.[A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko,W. Wang, T. Weyand, M. Andreetto, and H. Adam.Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.](https://arxiv.org/pdf/1704.04861.pdf)
