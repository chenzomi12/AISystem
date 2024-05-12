<!--Copyright © 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# 训练后量化 PTQ 与部署

## 动态离线量化

动态离线量化(Post Training Quantization Dynamic, PTQ Dynamic)仅将模型中特定算子的权重从FP32类型映射成 INT8/16 类型，主要可以减小模型大小，对特定加载权重费时的模型可以起到一定加速效果。但是对于不同输入值，其缩放因子是动态计算。动态量化的权重是离线转换阶段量化，而激活是在运行阶段才进行量化。因此动态量化是几种量化方法中性能最差的。

不同的精度下的动态量化对模型的影响：

- 权重量化成 INT16 类型，模型精度不受影响，模型大小为原始的 1/2。
- 权重量化成 INT8 类型，模型精度会受到影响，模型大小为原始的 1/4。

![动态离线量化算法流程](04.ptq01.png)

动态离线量化将模型中特定OP的权重从FP32类型量化成INT8等类型，该方式的量化有两种预测方式：

1. 反量化预测方式，即是首先将INT8/16类型的权重反量化成FP32类型，然后再使用FP32浮运算运算进行预测。
2. 量化预测方式，即是预测中动态计算量化OP输入的量化信息，基于量化的输入和权重进行INT8整形运算。

## 静态离线量化

静态离线量化（Post Training Quantization Static, PTQ Static）同时也称为校正量化或者数据集量化，使用少量无标签校准数据。其核心是计算量化比例因子，使用静态量化后的模型进行预测，在此过程中量化模型的缩放因子会根据输入数据的分布进行调整。相比量化训练，静态离线量化不需要重新训练，可以快速得到量化模型。

$$
uint8 = round(float/scale) - offset
$$

静态离线量化的目标是求取量化比例因子，主要通过对称量化、非对称量化方式来求，而找最大值或者阈值的方法又有MinMax、KL散度、ADMM、EQ，MSE等方法。

静态离线量化的步骤如下：

1. 加载预训练的FP32模型，配置用于校准的数据加载器；
2. 读取小批量样本数据，执行模型的前向推理，保存更新待量化算子的量化scale等信息；
3. 将FP32模型转成INT8模型，进行保存。

![静态离线量化流程](04.ptq02.png)

一些常用的计算量化scale的方法：

| 量化方法 | 方法详解                                                     |
| -------- | ------------------------------------------------------------ |
| $abs_{max}$  | 选取所有激活值的绝对值的最大值作为截断值α。此方法的计算最为简单，但是容易受到某些绝对值较大的极端值的影响，适用于几乎不存在极端值的情况。 |
| $KL$       | 使用参数在量化前后的KL散度作为量化损失的衡量指标。此方法是TensorRT所使用的方法。在大多数情况下，使用KL方法校准的表现要优于abs_max方法。 |
| $avg $     | 选取所有样本的激活值的绝对值最大值的平均数作为截断值α。此方法计算较为简单，可以在一定程度上消除不同数据样本的激活值的差异，抵消一些极端值影响，总体上优于abs_max方法。 |

## KL散度校准法

### 原理

KL散度校准法也叫相对熵，其中p表示真实分布，q表示非真实分布或p的近似分布：

$$
𝐷_{𝐾𝐿} (𝑃_f || 𝑄_𝑞)=\sum\limits^{N}_{i=1}𝑃(𝑖)*𝑙𝑜𝑔_2\frac{𝑃_𝑓(i)}{𝑄_𝑞(𝑖)}
$$

相对熵，用来衡量真实分布与非真实分布的差异大小。目的就是改变量化域，实则就是改变真实的分布，并使得修改后得真实分布在量化后与量化前相对熵越小越好。

### 流程和实现

1. 选取validation数据集中一部分具有代表的数据作为校准数据集 Calibration
2. 对于校准数据进行FP32的推理，对于每一层：
     1. 收集activation的分布直方图
     2. 使用不同的threshold来生成一定数量的量化好的分布
     3. 计算量化好的分布与FP32分布的KL divergence，并选取使KL最小的threshold作为saturation的阈值

主要注意的点：
- 需要准备小批量数据（500~1000张图片）校准用的数据集；
- 使用校准数据集在FP32精度的网络下推理，并收集激活值的直方图；
- 不断调整阈值，并计算相对熵，得到最优解

通俗地理解，算法收集激活Act直方图，并生成一组具有不同阈值的8位表示法，选择具有最少kl散度的表示；此时的 kl 散度在参考分布（FP32激活）和量化分布之间（即8位量化激活）之间。

KL散度校准法的伪代码实现：

```python
Input: FP32 histogram H with 2048 bins: bin[0], … , bin[2047]

For i in range(128, 2048):
     reference distribution P = [bin[0], …, bin[i-1]]
     outliers count = sum(bin[i], bin[i+1], …, bin[2047])
     reference distribution P[i-1] += outliers count 
     P /= sum(P)
     candidate distribution Q = quantize [bin[0], …, bin[i-1]] into 128 levels
     expand candidate distribution Q to I bins
     Q /= sum(Q)
     divergence[i] = KL divergence(reference distribution P, candidate distribution Q)
End For

Find index m for which divergence[m] is minimal

threshold = (m+0.5) * (width of a bin)

```

## 端侧量化推理部署

### 推理结构

端侧量化推理的结构方式主要由3种，分别是下图 (a) Fp32输入Fp32输出、(b) Fp32输入int8输出、(c) int8输入int32输出

![端侧量化推理方式](04.ptq04.png)

INT8卷积如下图所示，里面混合里三种不同的模式，因为不同的卷积通过不同的方式进行拼接。使用INT8进行inference时，由于数据是实时的，因此数据需要在线量化，量化的流程如图所示。数据量化涉及Quantize，Dequantize和Requantize等3种操作：

![INT8卷积示意图](04.ptq05.png)

### 量化过程

#### 量化

将float32数据量化为int8。离线转换工具转换的过程之前，根据量化原理的计算出数据量化需要的scale和offset：

#### 反量化

INT8相乘、加之后的结果用INT32格式存储，如果下一Operation需要float32格式数据作为输入，则通过Dequantize反量化操作将INT32数据反量化为float32。Dequantize反量化推导过程如下：

#### 重量化

INT8乘加之后的结果用INT32格式存储，如果下一层需要INT8格式数据作为输入，则通过Requantize重量化操作将INT32数据重量化为INT8。重量化推导过程如下：



## 参考

- 8-bit Inference with TensorRT https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

## 本节视频

<html>
<iframe src="https:&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>
