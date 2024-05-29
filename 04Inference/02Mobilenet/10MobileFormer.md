<!--Copyright © XcodeHw 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# MobileFormer

在本章节中，将介绍一种新的网络-MobileFormer，它实现了Transformer全局特征与CNN局部特征的融合，在较低的成本内，创造一个高效的网络。通过本章节，让大家去了解如何将CNN与Transformer更好的结合起来，同时实现模型的轻量化。

## MobileFormer

**MobileFormer**:一种通过双线桥将MobileNet和Transformer并行的结构。这种方式融合了MobileNet局部性表达能力和Transformer全局表达能力的优点，这个桥能将局部性和全局性双向融合。和现有Transformer不同，Mobile-Former使用很少的tokens(例如6个或者更少)随机初始化学习全局先验，计算量更小。

### 设计思路

#### 并行结构

Mobile-Former将MobileNet和Transformer并行化，并通过双向交叉注意力连接（下见图）。Mobile（指MobileNet）采用图像作为输入（$X\in R^{HW \times 3}$），并应用反向瓶颈块提取局部特征。Former（指Transformers）将可学习的参数（或tokens）作为输入，表示为 $Z\in R^{M\times d}$,其中M和d分别是tokens的数量和维度，这些tokens随机初始化。与视觉Transformer（ViT）不同，其中tokens将局部图像patch线性化，Former的tokens明显较少（M≤6），每个代表图像的全局先验知识。这使得计算成本大大降低。

![MobileFormer](./images/10.mobileformer_01.png)

#### 低成本双线桥

Mobile和Former通过双线桥将局部和全局特征双向融合。这两个方向分别表示为Mobile→Former和Mobile←Former。我们提出了一种轻量级的交叉注意力模型，其中映射（$W^{Q}$,$W^{K}$,$W^{V}$)从Mobile中移除，以节省计算，但在Former中保留。在通道数较少的Mobile瓶颈处计算交叉注意力。具体而言，从局部特征图X到全局tokens Z的轻量级交叉注意力计算如下：

$$
A_{X->Z} = [Attn(\widetilde{z_{i}}W_{i}^{Q},\widetilde{x_{i}},\widetilde{x_{i}})]_{i=1:h}W^{o}\tag{1}
$$

其中局部特征X和全局tokens Z被拆分进入h个头，即$X=[\widetilde{x_{1}}...\widetilde{x_{h}}],Z=[\widetilde{z_{1}}...\widetilde{z_{h}}]$表示多头注意力。第i个头的拆分$\widetilde{z_{1}}\in R^{M \times \frac {d}{h} }$与第i个token$\widetilde{z_{1}}\in R^{d}$不同。$W_{i}^{Q}$是第i个头的查询映射矩阵。 $W^{O}$用于将多个头组合在一起。Attn(Q,K,V)是查询Q、键K和值V的标准注意力函数，即 $softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$
，其中 $[.]_{1:h}$表示将h个元素concat到一起。需要注意的是，键和值的映射矩阵从Mobile中移除，而查询的映射矩阵$W_{i}^{Q}$在Former中保留。类似地 从全局到局部的交叉注意力计算如下：

$$
A_{Z->X} = [Attn(\widetilde{x_{i}},\widetilde{z_{i}}\odot W_{i}^{K},\widetilde{z_{i}}\odot W_{i}^{V})]_{i=1:h}\tag{2}
$$

其中$W_{i}^{K}$和$W_{i}^{V}$分别是Former中键和值的映射矩阵。而查询的映射矩阵从Mobile中移除。

#### Mobile-Former块

Mobile-Former由Mobile-Former块组成。每个块包含四部分：Mobile子块、Former子块以及双向交叉注意力Mobile←Former和Mobile→Former（如下图所示）。

![MobileFormer](./images/10.mobileformer_02.png)

输入和输出：Mobile-Former块有两个输入：(a) 局部特征图$X\in R^{HW\times C}$，为C通道、高度H和宽度W，以及(b) 全局tokens $Z\in R^{M\times d}$，其中M和d是分别是tokens的数量和维度，M和d在所有块中一样。Mobile-Former块输出更新的局部特征图$X$和全局tokens$Z$，用作下一个块的输入。

Mobile子块：如上图所示，Mobile子块将特征图$X$作为输入，并将其输出作为Mobile←Former的输入。这和反向瓶颈块略有不同，其用动态ReLU替换ReLU作为激活函数。不同于原始的动态ReLU，在平均池化特征图上应用两个MLP以生成参数。我们从Former的第一个全局tokens的输出$z'_{1}$应用两个MLP层（上图中的θ）保存平均池化。其中所有块的depth-wise卷积的核大小为3×3。

Former子块：Former子块是一个标准的Transformer块，包括一个多头注意力（MHA）和一个前馈网络（FFN）。在FFN中，膨胀率为2（代替4）。使用post层归一化。Former在Mobile→Former和Mobile←Former之间处理（见上图）。

Mobile→Former：文章提出的轻量级交叉注意力（式1）用于将局部特征X融合到全局特征 tokens Z。与标准注意力相比，映射矩阵的键$W^{K}$和值$W^{V}$（在局部特征X上）被移除以节省计算（见上图）。

Mobile←Former：这里的交叉注意力（式2） 与Mobile→Former的方向相反，其将全局tokens融入本地特征。局部特征是查询，全局tokens是键和值。因此，我们保留键$W^{K}$和值$W^{V}$中的映射矩阵，但移除查询$W^{Q}$的映射矩阵以节省计算，如上图所示。

计算复杂度：Mobile-Former块的四个核心部分具有不同的计算成本。给定输入大小为$HW\timesC$的特征图，以及尺寸为d的M个全局tokens，Mobile占据了大部分的计算量$O(HWC^{2})$。Former和双线桥是重量级的，占据不到总计算成本的20%。具体而言，Former的自注意力和FFN具有复杂度 $O(M^{2}d+Md^{2})$。 Mobile→Former和Mobile←Former共享交叉注意力的复杂度$O(MHWC+MdC)$。

**代码**



```python
class Former(nn.Module):
    '''Post LayerNorm, no Res according to the paper.'''
    def __init__(self, head, d_model, expand_ratio=2):
        super(Former, self).__init__()
        self.d_model = d_model
        self.expand_ratio = expand_ratio
        self.eps = 1e-10
        self.head = head
        assert self.d_model % self.head == 0
        self.d_per_head = self.d_model // self.head

        self.QVK = MLP([self.d_model, self.d_model * 3], bn=False).cuda()
        self.Q_to_heads = MLP([self.d_model, self.d_model], bn=False).cuda()
        self.K_to_heads = MLP([self.d_model, self.d_model], bn=False).cuda()
        self.V_to_heads = MLP([self.d_model, self.d_model], bn=False).cuda()
        self.heads_to_o = MLP([self.d_model, self.d_model], bn=False).cuda()
        self.norm = nn.LayerNorm(self.d_model).cuda()
        self.mlp = MLP([self.d_model, self.expand_ratio * self.d_model, self.d_model], bn=False).cuda()
        self.mlp_norm = nn.LayerNorm(self.d_model).cuda()

    def forward(self, x):
        QVK = self.QVK(x)
        Q = QVK[:, :, 0: self.d_model]
        Q = rearrange(self.Q_to_heads(Q), 'n m ( d h ) -> n m d h', h=self.head)   # (n, m, d/head, head)
        K = QVK[:, :, self.d_model: 2 * self.d_model]
        K = rearrange(self.K_to_heads(K), 'n m ( d h ) -> n m d h', h=self.head)   # (n, m, d/head, head)
        V = QVK[:, :, 2 * self.d_model: 3 * self.d_model]
        V = rearrange(self.V_to_heads(V), 'n m ( d h ) -> n m d h', h=self.head)   # (n, m, d/head, head)
        scores = torch.einsum('nqdh, nkdh -> nhqk', Q, K) / (np.sqrt(self.d_per_head) + self.eps)   # (n, h, q, k)
        scores_map = F.softmax(scores, dim=-1)  # (n, h, q, k)
        v_heads = torch.einsum('nkdh, nhqk -> nhqd', V, scores_map) #   (n, h, m, d_p) -> (n, m, h, d_p)
        v_heads = rearrange(v_heads, 'n h q d -> n q ( h d )')
        attout = self.heads_to_o(v_heads)
        attout = self.norm(attout)  #post LN
        attout = self.mlp(attout)
        attout = self.mlp_norm(attout)  # post LN
        return attout   # No res

```

```python

class Mobile_Former(nn.Module):
    '''Local feature -> Global feature'''
    def __init__(self, d_model, in_channel):
        super(Mobile_Former, self).__init__()
        self.d_model, self.in_channel = d_model, in_channel

        self.project_Q = nn.Linear(self.d_model, self.in_channel).cuda()
        self.unproject = nn.Linear(self.in_channel, self.d_model).cuda()
        self.eps = 1e-10
        self.shortcut = nn.Sequential().cuda()

    def forward(self, local_feature, x):
        _, c, _, _ = local_feature.shape
        local_feature = rearrange(local_feature, 'n c h w -> n ( h w ) c')   # N, L, C
        project_Q = self.project_Q(x)   # N, M, C
        scores = torch.einsum('nmc , nlc -> nml', project_Q, local_feature) * (c ** -0.5)
        scores_map = F.softmax(scores, dim=-1)  # each m to every l
        fushion = torch.einsum('nml, nlc -> nmc', scores_map, local_feature)
        unproject = self.unproject(fushion) # N, m, d
        return unproject + self.shortcut(x)

```

```python
class Mobile(nn.Module):
    '''Without shortcut, if stride=2, donwsample, DW conv expand channel, PW conv squeeze channel'''
    def __init__(self, in_channel, expand_size, out_channel, token_demension, kernel_size=3, stride=1, k=2):
        super(Mobile, self).__init__()
        self.in_channel, self.expand_size, self.out_channel = in_channel, expand_size, out_channel
        self.token_demension, self.kernel_size, self.stride, self.k = token_demension, kernel_size, stride, k

        if stride == 2:
            self.strided_conv = nn.Sequential(
                nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, stride=2, padding=int(self.kernel_size // 2), groups=self.in_channel).cuda(),
                nn.BatchNorm2d(self.expand_size).cuda(),
                nn.ReLU6(inplace=True).cuda()
            )
            self.conv1 = nn.Conv2d(self.expand_size, self.in_channel, kernel_size=1, stride=1).cuda()
            self.bn1 = nn.BatchNorm2d(self.in_channel).cuda()
            self.ac1 = DynamicReLU(self.in_channel, self.token_demension, k=self.k).cuda()      
            self.conv2 = nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, stride=1, padding=1, groups=self.in_channel).cuda()
            self.bn2 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac2 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()          
            self.conv3 = nn.Conv2d(self.expand_size, self.out_channel, kernel_size=1, stride=1).cuda()
            self.bn3 = nn.BatchNorm2d(self.out_channel).cuda()
        else:
            self.conv1 = nn.Conv2d(self.in_channel, self.expand_size, kernel_size=1, stride=1).cuda()
            self.bn1 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac1 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()      
            self.conv2 = nn.Conv2d(self.expand_size, self.expand_size, kernel_size=3, stride=1, padding=1, groups=self.expand_size).cuda()
            self.bn2 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac2 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()          
            self.conv3 = nn.Conv2d(self.expand_size, self.out_channel, kernel_size=1, stride=1).cuda()
            self.bn3 = nn.BatchNorm2d(self.out_channel).cuda()

    def forward(self, x, first_token):
        if self.stride == 2:
            x = self.strided_conv(x)
        x = self.bn1(self.conv1(x))
        x = self.ac1(x, first_token)
        x = self.bn2(self.conv2(x))
        x = self.ac2(x, first_token)
        return self.bn3(self.conv3(x))


```



```python




class Former_Mobile(nn.Module):
    '''Global feature -> Local feature'''
    def __init__(self, d_model, in_channel):
        super(Former_Mobile, self).__init__()
        self.d_model, self.in_channel = d_model, in_channel
        
        self.project_KV = MLP([self.d_model, 2 * self.in_channel], bn=False).cuda()
        self.shortcut = nn.Sequential().cuda()
    
    def forward(self, x, global_feature):
        res = self.shortcut(x)
        n, c, h, w = x.shape
        project_kv = self.project_KV(global_feature)
        K = project_kv[:, :, 0 : c]  # (n, m, c)
        V = project_kv[:, :, c : ]   # (n, m, c)
        x = rearrange(x, 'n c h w -> n ( h w ) c') # (n, l, c) , l = h * w
        scores = torch.einsum('nqc, nkc -> nqk', x, K) # (n, l, m)
        scores_map = F.softmax(scores, dim=-1) # (n, l, m)
        v_agg = torch.einsum('nqk, nkc -> nqc', scores_map, V)  # (n, l, c)
        feature = rearrange(v_agg, 'n ( h w ) c -> n c h w', h=h)
        return feature + res
```



### 网络结构

一个Mobile-Former架构，图像大小为224×224，294M FLOPs，以不同的输入分辨率堆叠11个Mobile-Former块。所有块都有6个维度为192的全局tokens。它以一个3×3的卷积作为stem和第一阶段的轻量瓶颈块，首先膨胀，然后通过3×3 depth-wise卷积和point-wise卷积压缩通道数。第2-5阶段包括 Mobile-Former块。每个阶段的下采样，表示为Mobile-Former分类头在局部特征应用平均池化，首先和全局tokens concat到一起，然后经过两个全连接层，中间是h-swish激活函数。Mobile-Former有七种模型，计算成本从26M到508M FLOPs。它们的结构相似，但宽度和高度不同。

**代码**

```python
class MobileFormerBlock(nn.Module):
    '''main sub-block, input local feature (N, C, H, W) & global feature (N, M, D)'''
    '''output local & global, if stride=2, then it is a downsample Block'''
    def __init__(self, in_channel, expand_size, out_channel, d_model, stride=1, k=2, head=8, expand_ratio=2):
        super(MobileFormerBlock, self).__init__()

        self.in_channel, self.expand_size, self.out_channel = in_channel, expand_size, out_channel
        self.d_model, self.stride, self.k, self.head, self.expand_ratio = d_model, stride, k, head, expand_ratio

        self.mobile = Mobile(self.in_channel, self.expand_size, self.out_channel, self.d_model, kernel_size=3, stride=self.stride, k=self.k).cuda()
        self.former = Former(self.head, self.d_model, expand_ratio=self.expand_ratio).cuda()
        self.mobile_former = Mobile_Former(self.d_model, self.in_channel).cuda()
        self.former_mobile = Former_Mobile(self.d_model, self.out_channel).cuda()
    
    def forward(self, local_feature, global_feature):
        z_hidden = self.mobile_former(local_feature, global_feature)
        z_out = self.former(z_hidden)
        x_hidden = self.mobile(local_feature, z_out[:, 0, :])
        x_out = self.former_mobile(x_hidden, z_out)
        return x_out, z_out
```







## 小结

本文提出了一种基于MobileNet和Transformer的双向式交互并行设计的网络Mobile-Former。它利用了MobileNet在局部信息处理中的效率和Transformer在编码全局交互方面的优势。该设计不仅有效地提高了计算精度，而且还有效地节省了计算成本。在低FLOP条件下，它在图像分类和目标检测方面都优于高效的CNN和ViT变体。

## 本节视频

<html>
<iframe src="https:&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>

## 参考文献

1.[Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to end object detection with transformers. In ECCV, 2020. 2,4, 5, 7, 8](https://arxiv.longhoe.net/abs/2005.12872)

2.[Yinpeng Chen, Xiyang Dai, Mengchen Liu, Dongdong Chen, Lu Yuan, and Zicheng Liu. Dynamic relu. In ECCV,2020. 2, 3, 4, 6](https://arxiv.longhoe.net/abs/2003.10027)

3.[Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Va sudevan, and Quoc V. Le. Autoaugment: Learning augmen tation strategies from data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition(CVPR), June 2019. 5](https://openaccess.thecvf.com/content_CVPR_2019/html/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.html)

4.[St´ephane d’Ascoli, Hugo Touvron, Matthew Leavitt, Ari Morcos, Giulio Biroli, and Levent Sagun. Convit: Improving vision transformers with soft convolutional inductive biases.arXiv preprint arXiv:2103.10697, 2021. 2, 3, 5, 6](https://arxiv.longhoe.net/abs/2103.10697)

5.[Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 5, 6, 12](https://store.computer.org/csdl/proceedings-article/cvpr/2009/05206848/12OmNxWcH55)

6.[Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, and Baining Guo. Cswin transformer: A general vision transformer backbone with cross-shaped windows. arXiv preprint arXiv:2107.00652, 2021. 2](https://arxiv.longhoe.net/abs/2107.00652)

7.[Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl vain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 1, 2, 3](https://arxiv.longhoe.net/abs/2010.11929)

8.[Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron,Pierre Stock, Armand Joulin, Herv´ e J´ egou, and Matthijs Douze. Levit: a vision transformer in convnet’s clothing for faster inference. arXiv preprint arXiv:22104.01136, 2021. 1,2, 3, 6](https://arxiv.longhoe.net/abs/2104.01136)

9.[Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and patter recognition, pages 770–778, 2016. 3, 7, 8](https://arxiv.longhoe.net/abs/1512.03385)

10.[Geoffrey E. Hinton. How to represent part-whole hierarchies in a neural network. CoRR, abs/2102.12627, 2021. 2](https://ui.adsabs.harvard.edu/abs/2021arXiv210212627H/abstract)

11.[Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu,Ruoming Pang, Vijay Vasudevan, Quoc V. Le, and Hartwig Adam. Searching for mobilenetv3. In Proceedings of the IEEE/CVF International Conference on Computer Vision(ICCV), October 2019. 1, 2, 4, 5, 6, 7, 8, 11, 12](https://arxiv.longhoe.net/abs/1905.02244)

12.[Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco An dreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017. 1, 2](https://arxiv.longhoe.net/abs/1704.04861)

13.[Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018. 2](https://arxiv.longhoe.net/abs/1709.01507)

14.[Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu,Xiyang Dai, Lu Yuan, and Lei Zhang. Cvt: Introducing convolutions to vision transformers, 2021. 1, 2, 3](https://arxiv.longhoe.net/abs/2103.15808)

15.[DaquanZhou, Qi-BinHou, Y.Chen, Jiashi Feng, and S. Yan.Rethinking bottleneck structure for efficient mobile network design. In ECCV, August 2020. 2](https://arxiv.longhoe.net/abs/2007.02269)