<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/DeepLearningSystem)版权许可-->

# 六、大模型遇到 AI 系统

大模型是基于 AI 集群的全栈软硬件性能优化，通过最小的每一块AI芯片组成的AI集群，编译器使能到上层的AI框架，训练过程需要分布式并行、集群通信等算法支持，而且在大模型领域最近持续演进如智能体等新技术。

> 希望这个系列能够给朋友们带来一些帮助，也希望ZOMI能够继续坚持完成所有内容哈！欢迎您也参与到这个开源项目的贡献！

## 课程简介

- [《大模型全流程》](./01Introduce/)分为两部分，第一部分在纵深上深入地去根据大模型在 AI 系统全栈中的冲击内容，去体会大模型遇到 AI 系统的整体架构；第二部分根据大模型从集群建设、数据算法、训练微调推理、推理部署应用的全流程每一个环节之间的关系。

- [《AI 集群简介》](./02AICluster/)宏观层面在 AI 集群的基础上整体了解大模型在 AI 集群的训练效率，推理和训练在集群中所占用的显存，硬件层面从 AI 集群的具体硬件模块及其相匹配的基本组成，并基于此形成的 AI 集群服务器的整体架构。

- [《AI 集群存储》](./03Storage/)想要占领大模型应用的高地，数据和算力可以说是不可或缺的基石。和算力相关的讨论已经有很多，以至于英伟达的市值在2023年翻了两番。同样不应小觑的还有数据，除了数据量的爆炸性增长，数据的读取、写入、传输等基础性能，开始遇到越来越多的新挑战。

- [《AI 智能体》](./12Agent/)AI Agent 智能体基于 LLMs 大语言模型的能力，未来不仅会改变每个人与计算机交互的方式。它们还将颠覆软件行业，带来自我们从键入命令到点击图标以来最大的计算革命。

## 课程细节

> `PPT`和`字幕`需要到 [Github](https://github.com/chenzomi12/DeepLearningSystem) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

### [《大模型全流程》](./01Introduce/)

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 大模型 | 01 大模型整体介绍(上) | [slide](./01_introduction.pdf), [video](https://www.bilibili.com/video/BV1a34y137zi/) |
| 大模型 | 02 大模型整体介绍(下) | [slide](./02_introduction.pdf), [video](https://www.bilibili.com/video/BV1F34y1G7Fz/) |

### [《AI 集群简介》](./02AICluster/)

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| AI 集群简介 | 03 大模型训练效率 | [slide](./03efficiency.pdf), [video](https://www.bilibili.com/video/BV1dC4y1d7hd) |
| AI 集群简介 | 04 AI集群硬件组成 | [slide](./04Hardware.pdf), [video](https://www.bilibili.com/video/BV1dC4y1d7hd) |
| AI 集群简介 | 05 集群服务器架构 | [slide](./05ClusterArch.pdf), [video](https://www.bilibili.com/video/BV1384y127iP) |
| AI 集群简介 | 05 大模型训练显存分析 | [slide](./06TrainingMemory.pdf), [video](https://www.bilibili.com/video/BV15Q4y147Uo) |
| AI 集群简介 | 06 大模型推理显存分析 | [slide](./07InferenceMemory.pdf), [video](https://www.bilibili.com/video/BV1Rc411S7jj) |

### [《AI 集群存储》](./03Storage/)

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 大模型 | 01 存储遇到大模型介绍 | [slide](./01Introduce.pdf), [video](https://www.bilibili.com/video/BV1H94y1J7wq) |
| 大模型 | 02 存储硬件介质组成 | [slide](./02Hardware.pdf), [video](https://www.bilibili.com/video/BV1fw411P7FY) |
| 大模型 | 03 存储集群连接方式 | [slide](./03Connect.pdf), [video](https://www.bilibili.com/video/BV1SQ4y147b3) |
| 大模型 | 04 数据存储的类型 | [slide](./04Object.pdf), [video](https://www.bilibili.com/video/BV1fa4y1Z76n) |
| 大模型 | 05 存储遇到大模型的挑战 | [slide](./05Challenge.pdf), [video](https://www.bilibili.com/video/BV1UG411i7SM) |
| 大模型 | 06 训练存储优化方案(上) | [slide](./06Optimizer.pdf), [video](https://www.bilibili.com/video/BV1uw411h7B7) |
| 大模型 | 07 训练存储优化方案(下) | [slide](./07Checkpoint.pdf), [video](https://www.bilibili.com/video/BV11u4y1c7Pu) |
| 大模型 | 08 大模型CKPT优化手段 | [slide](./07Checkpoint.pdf), [video](https://www.bilibili.com/video/BV1wM411d7cc) |
| 大模型 | 09 存算架构思考 | [slide](./08Future.pdf), [video](https://www.bilibili.com/video/BV1kw411h74p/) |

### [AI 智能体](./12Agent/)

| 大纲 | 小节 | 链接|
|:--:|:--:|:--:|
| 大模型 | 01 大模型遇到AI Agent | [slide](./12Agent/01Introduction.pdf), [video](https://www.bilibili.com/video/BV11w411p7dW/) |
| 大模型 | 02 AI Agent具体组成 | [slide](./12Agent/02Component.pdf), [video](https://www.bilibili.com/video/BV11u4y1P73P/) |
| 大模型 | 03 Planning与Prompt关系 | [slide](./12Agent/03Planning.pdf), [video](https://www.bilibili.com/video/BV1kM411f7Gb/) |
| 大模型 | 04 AI Agent应用原理剖析 | [slide](./12Agent/04Application.pdf), [video](https://www.bilibili.com/video/BV1zM411f7n2/) |
| 大模型 | 05 AI Agent问题与未来思考 | [slide](./12Agent/05Summary.pdf), [video](https://www.bilibili.com/video/BV1KC4y1S7ZG/) |

### [大模型专题解读](./13chatGPT/)
| 名称        | 名称               | 备注                                                                              |
| --------- | ---------------- | ------------------------------------------------------------------------------- |
| ChatGPT狂飙 | 01 GPT 系列详解       | [silde](./13chatGPT/chatGPT01.pdf), [video](https://www.bilibili.com/video/BV1kv4y1s7V7/) |
| ChatGPT狂飙 | 02 RLHF 强化学习 PPO   | [silde](./13chatGPT/chatGPT02.pdf), [video](https://www.bilibili.com/video/BV1w8411M7YB/) |
| ChatGPT狂飙 | 03 InstructGPT 解读 | [silde](./13chatGPT/chatGPT03.pdf), [video](https://www.bilibili.com/video/BV1e24y1s7k8/) |
| 视频生成模型 | 01 OpenAI SORA | [silde](./13chatGPT/SORA01.pdf), [video](https://www.bilibili.com/video/BV1jx421C7mG/) |
| 世界模型 | 01 World Model | [silde](./13chatGPT/WorldModel01.pdf), [video]() |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AISys](https://chenzomi12.github.io/) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/videos)，PPT开源在[github](https://github.com/chenzomi12/DeepLearningSystem)，欢迎取用！！！

> 非常希望您也参与到这个开源项目中，B站给ZOMI留言哦！
>
> 欢迎大家使用的过程中发现bug或者勘误直接提交代码PR到开源社区哦！
>
> 希望这个系列能够给大家、朋友们带来一些些帮助，也希望自己能够继续坚持完成所有内容哈！
