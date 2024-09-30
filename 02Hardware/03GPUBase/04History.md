<!--Copyright 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# GPU 架构回顾

1999 年，英伟达发明了 GPU（graphics processing unit），本节将介绍英伟达 GPU 从 Fermi 到 Blackwell 共 9 代架构，时间跨度从 2010 年至 2024 年，具体包括费米（Feimi）、开普勒（Kepler）、麦克斯韦（Maxwell）、帕斯卡（Pashcal）、伏特（Volt）、图灵（Turing）、安培（Ampere）和赫柏（Hopper）和布莱克韦尔（Blackwell）架构。经过 15 年的发展，CUDA 已成为英伟达的技术“护城河”，Tensor Core5.0，NVLink5.0，NVswitch4.0，Transformer Engine2.0 等技术迭代更新，正如英伟达公司官方宣传语所言：“人工智能计算领域的领导者，推动了 AI、HPC、游戏、创意设计、自动驾驶汽车和机器人开发领域的进步。”

| **架构名称** | **中文名字** | **发布时间** | **核心参数** | **特点&优势** | **纳米制程** | **代表型号** |
| --- | --- | --- | --- | --- | --- | --- |
| **Fermi** | 费米 | 2010 | 16 个 SM，每个 SM 包含 32 个 CUDA Cores，一共 512 CUDA Cores | 首个完整 GPU 计算架构，支持与共享存储结合的 Cache 层次 GPU 架构，支持 ECC GPU 架构 | 40/28nm, 30 亿晶体管 | Quadro 7000 |
| **Kepler** | 开普勒 | 2012 | 15 个 SMX，每个 SMX 包括 192 个 FP32+64 个 FP64 CUDA Cores | 游戏性能大幅提升，首次支持 GPU Direct 技术 | 28nm, 71 亿晶体管 | K80, K40M |
| **Maxwell** | 麦克斯韦 | 2014 | 16 个 SM，每个 SM 包括 4 个处理块，每个处理块包括 32 个 CUDA Cores+8 个 LD/ST Unit + 8 SFU | 每组 SM 单元从 192 个减少到每组 128 个，每个 SMM 单元拥有更多逻辑控制电路 | 28nm, 80 亿晶体管 | M5000, M4000GTX 9XX 系列 |
| **Pascal** | 帕斯卡 | 2016 | GP100 有 60 个 SM，每个 SM 包括 64 个 CUDA Cores，32 个 DP Cores | NVLink 第一代，双向互联带宽 160GB/s，P100 拥有 56 个 SM HBM | 16nm, 153 亿晶体管 | P100, P6000, TTX1080 |
| **Volta** | 伏特 | 2017 | 80 个 SM，每个 SM 包括 32 个 FP64+64 Int32+64 FP32+8 个 Tensor Cores | NVLink2.0，Tensor Cores 第一代，支持 AI 运算，NVSwitch1.0 | 12nm, 211 亿晶体管 | V100, TiTan V |
| **Turing** | 图灵 | 2018 | 102 核心 92 个 SM，SM 重新设计，每个 SM 包含 64 个 Int32+64 个 FP32+8 个 Tensor Cores | Tensor Core2.0，RT Core 第一代 | 12nm, 186 亿晶体管 | T4，2080TI, RTX 5000 |
| **Ampere** | 安培 | 2020 | 108 个 SM，每个 SM 包含 64 个 FP32+64 个 INT32+32 个 FP64+4 个 Tensor Cores | Tensor Core3.0，RT Core2.0，NVLink3.0，结构稀疏性矩阵 MIG1.0 | 7nm, 283 亿晶体管 | A100, A30 系列 |
| **Hopper** | 赫柏 | 2022 | 132 个 SM，每个 SM 包含 128 个 FP32+64 个 INT32+64 个 FP64+4 个 Tensor Cores | Tensor Core4.0，NVLink4.0，结构稀疏性矩阵 MIG2.0 | 4nm, 800 亿晶体管 | H100 |
| **Blackwell** | 布莱克韦尔 | 2024 | - | Tensor Core5.0，NVLink5.0, 第二代 Transformer 引擎，支持 RAS | 4NP,  2080 亿晶体管 | B200 |

## Fermi 架构

2006 年英伟达提出 G80 架构，使开发者可以基于 C 语言在 GPU 上进行开发。2008 年基于 G80 架构提出 GT200，增加了流处理器核的数量，更高的精度和并行能力使 GPU 进行科学计算和高性能计算成为可能。

2010 年英伟达提出 Feimi 架构，最大可支持 16 个 SMs，每个 SM 有 32 个 CUDA Core，一共 512 个 CUDA Core，架构设计主要是以当时游戏用户的需求为主，因此整个 GPU 有多个 GPC（图形处理簇），单个 GPC 包含一个光栅引擎（Raster Engine）和 4 个 SM。

GPU 拥有 6 个 64 位内存分区，总共是 384 位内存，最多支持 6 GB GDDR5 DRAM 内存。主机接口通过 PCI-Express 连接 GPU 和 CPU。GigaThread 全局调度器将线程块分配给 SM 线程调度器。因为计算核心较多，因此将 L2 Cache 放在处理器中间位置，使得数据可以在 CUDA Core 之间快速传输。

![英伟达 Feimi 架构](images/04History01.png)

> 恩里科·费米（Enrico Fermi）是意大利裔美国物理学家，20 世纪最重要的物理学家之一，被誉为“原子能时代之父”。他在核物理、量子力学和统计力学等领域做出了重要贡献。主要成就包括：
> 
> 1. 提出费米子统计，即著名的费米-狄拉克统计，描述了半整数自旋的粒子的统计性质。
> 
> 2. 领导了芝加哥大学的“费米堆”课程，成功实现了世界上第一座自持核链反应堆。
> 
> 3. 参与了曼哈顿计划，对原子弹的研发做出了重要贡献。
> 
> 4. 获得了 1938 年的诺贝尔物理学奖，以表彰他在人类利用新的放射性同位素所作出的贡献。

Fermi 架构采用第三代流处理器，每个 SM 有 16 个加载/存储单元（Load/Store, LD/ST），允许为每个时钟 16 个线程计算源地址和目标地址，支持将每个地址的数据加载并存储到缓存或 DRAM 中。特殊功能单元（Special Function Unit, SFU）执行超越函数，如 sin、cos、导数和平方根。每个 SFU 在每个线程、每个时钟执行一条指令，一次 warp（由 32 个线程组成的线程组）要经过 8 个时钟周期。SFU 管线与调度单元解耦，允许调度单元在占用 SFU 时向其他执行单元发出命令。双精度算法是高性能计算应用的核心，每个 SM、每个时钟可执行多达 16 个双精度融合乘加运算。

每个 SM 有两个 warp 调度器和两个指令调度单元，允许同时发出和执行两个 warp。并行计算主要在 CUDA 中进行处理，每个 CUDA 处理器都有一个完整的流水线整数算术逻辑单元（ALU）和浮点单元（FPU），可以选择 FP 32 或者 INT 8 执行计算，但是 FP Unit 和 INT Unit 的执行不是并行的。

![Fermi 架构单个 SM 结构](images/03Concept05.png)

Fermi 架构支持新的并行线程执行 PTX 2.0（Parallel Thread Execution）指令集架构。一个 CUDA 程序被称作并行的 Kernel，线程分为三级，包含线程（Threads）、块（Blocks）和网格（Grid），每个层次结构对应硬件，Thread 可以共享局部内存（Local memory），线程块使用共享内存（Shared Memory），Grid 共享全局内存（Global Memory），具有相应的每个线程专用、每个块共享和每个应用程序全局内存空间。

![Fermi 架构线程分级](images/04History02.png)

## Kepler 架构

2012 年英伟达提出 Kepler 架构，由 7.1 亿个晶体管组成的 Kepler GK110 将提供超过 1 TFlop 的双精度吞吐量，采用台积电 28 nm 制程，每瓦的性能是费米架构的 3 倍。由 15 个 SMX 单元和 6 个 64 bit 内存控制器，内存子系统提供额外的缓存功能，在每个层次结构的存储器有更大的带宽，实现更快的 DRAM I/O，同时为编程模型提供硬件支持。

![Kepler 架构](images/04History03.png)

> 约翰内斯·开普勒（Johannes Kepler）是一位德国天文学家、数学家和占星术士，被誉为现代天文学的奠基人之一。他生活在 16 世纪末至 17 世纪初，是科学革命时期的重要人物，他的工作对天文学和物理学领域产生了深远的影响，为后来伽利略和牛顿等科学家的研究奠定了基础。主要成就包括：
> 
> 1. 提出了行星运动的三大定律，即开普勒定律：
> 
> - 第一定律：行星绕太阳运行的轨道是椭圆形的，太阳位于椭圆的一个焦点上。
> 
> - 第二定律：行星在其轨道上的矢量面积与时间的比率是常数。
> 
> - 第三定律：行星轨道的半长轴与公转周期的平方成正比。
> 
> 2. 通过观测和分析提出了行星运动的椭圆轨道理论，颠覆了当时的圆周运动观念。
> 
> 3. 对光学、天文学和数学领域都做出了重要贡献，为日后牛顿的引力理论奠定了基础。

开普勒架构相比上一代 Fermi 架构，SM（Streaming Multiprocessor）更名为 SMX，但是本身的概念没有改变，每个 SMX 具有四个 warp 调度器和八个指令调度单元，允许同时发出和执行四个 warp。Fermi 架构共有 32 核，Kepler 架构拥有 192 核，大大提升了 GPU 并行处理的能力。Fermi 支持最大线程数是 1536，Kepler 最大线程数达到 2048。64 个双精度（Double-Precision，DP）单元，32 特殊功能单元（SFU）和 32 个 LD/ST（load/store）单元，满足高性能计算场景的实际需求。

![Kepler 架构改进](images/04History04.png)

Kepler 架构支持动态并行（Dynnamic Parallelism），在不需要 CPU 支持的情况下自动同步，在程序执行过程中灵活动态地提供并行数量和形式。Hyper-Q 使多个 CPU 核使用单个 GPU 执行工作，提高 GPU 利用率并显着减少 CPU 空闲时间，允许 32 个同时进行的硬件管理连接，允许从多个 CUDA 流处理，多个消息传递进程中分离出单个进程。使用网格管理单元（Grid Management Unit，GMU）启用动态并行和调度控制，比如挂起或暂停网格和队列直到执行的环境准备好。

英伟达 GPUDirect 可以使单个计算机内的 GPU 或位于网络上不同服务器中的 GPU 直接交换数据，而无需转到 CPU 系统内存，RDMA 特性允许第三方设备直接访问同一系统内多个 GPU 上的内存，减少了对系统内存带宽的需求，释放 GPU DMA 引擎供其它 CUDA 任务使用。

## Maxwell 架构

2014 年英伟达提出 Maxwell 架构，麦克斯韦架构相比上一代架构没有太大改进，其中 SM 又使用了原来的名称，整体的核心个数变为 128 个，因为核心数不需要太多，可以通过超配线程数来提升 GPU 并行计算的能力。

![Maxwell 麦克斯韦架构](images/04History05.png)

SMM 使用基于象限的设计，其中每个 SMM 有四个共 32 核处理块，每个处理块都有一个专用的 warp 调度器，能够在每个时钟调度两条指令。每个 SMM 提供 8 个纹理单元，一个图形的几何处理引擎，以及专用的寄存器文件（Register File）和共享内存（Shared Memory）。单核性能是 Kepler 架构的 1.35 倍，performance/watt（性能与功耗的比率）是 Kepler 架构的两倍，在相同功耗下能够提供更高的性能。

![Maxwell 麦克斯韦 SM 架构](images/04History06.png)

> 詹姆斯·克拉克·麦克斯韦（James Clerk Maxwell）是 19 世纪苏格兰物理学家，被誉为电磁理论之父。他在电磁学和热力学领域做出了重要贡献，开创了现代物理学的新时代。主要成就包括：
> 
> 1. 提出了麦克斯韦方程组，总结了电磁场的基本规律，揭示了电磁波的存在，并将电磁学和光学统一起来。
> 
> 2. 发展了统计力学，提出了分子速度分布的麦克斯韦-玻尔兹曼分布定律，为热力学的发展做出了重要贡献。
> 
> 3. 提出了色散理论，解释了光的色散现象，为光学研究提供了新的理论基础。
> 
> 4. 预言了电磁波的存在，并在后来的实验证实了这一理论，为无线电通信的发展奠定了基础。

对比 Kepler 和 Maxwell 架构，Maxwell 架构拥有更大的专用共享内存，通过将共享内存与 L1 缓存分离，在每个 SMM 中提供专用的 64KB 共享内存，GM204 Maxwell 每个 SMM 的专用共享内存提高到 96KB。和 Kepler 和 Fermi 架构一样，每个线程块的最大共享内存仍然是 48KB。GM204 Maxwell 具有更大的二级缓存（L2 Cache），GK104 Kepler 的四倍，带宽受限的应用程序可以获得更大的性能优势。每个 SM 有更多活动线程块（Thread Blocks），从 16 增加到 32 有助于提高运行在小线程块上的内核使用率。可以对 32 位整数的本机共享内存进行原子操作，使线程块上的列表和栈类型数据更高效，和 Kepler 一样支持动态并行。

| GPU GeForce  | GTX 680 (Kepler GK104) | GTX 980 (Maxwell GM204) |
| --- | --- | --- |
| CUDA Cores | 1536 | 2048 |
| Base Clock | 1006 MHz | 1126 MHz |
| GPU Boost Clock | 1058 MHz | 1216 MHz |
| GFLOPs | 3090 | 4612 |
| Compute Capability | 3.0 | 5.2 |
| SMs | 8 | 16 |
| Shared Memory / SM | 48KB | 96KB |
| Register File Size / SM | 256KB | 256KB |
| Active Blocks / SM | 16 | 32 |
| Texture Units | 128 | 128 |
| Texel fill-rate | 128.8 Gigatexels/s | 144.1 Gigatexels/s |
| Memory | 2048 MB | 4096 MB |
| Memory Clock  | 6008 MHz | 7010 MHz |
| Memory Bandwidth | 192.3 GB/sec | 224.3 GB/sec |
| ROPs | 32 | 64 |
| L2 Cache Size | 512 KB | 2048 KB |
| TDP | 195 Watts | 165 Watts |
| Transistors | 3.54 billion | 5.2 billion |
| Die Size | 294 mm² | 398 mm² |
| Manufacturing Process | 28-nm | 28 nm |

## Pascal 架构

2016 年英伟达提出 Pascal 架构，相比之前的架构，Pascal 帕斯卡架构在应用场景、内存带宽和制程工艺等多个方面做出了创新。将系统内存 GDDR5 换成 HBM2，能够在更高的带宽下处理更大的工作数据集，提高效率和计算吞吐量，并减少从系统内存传输的频率，而且 HBM2 原生支持数据纠错（Error correcting Code, ECC）。采用 16nm FinFET 工艺，拥有 15.3 亿个晶体管，相同功耗下算力提升提升一个数量级。同时提出第一代 NVLink，提升单机卡间通信之外扩展多机之间的带宽。支持统一内存，允许在 GPU 和 CPU 的完整虚拟地址空间之间透明迁移数据，降低了并行编程的门槛。支持计算抢占和针对 Pascal 架构优化的 AI 算法，可应用于高性能计算、深度学习和 GPU 计算密集型领域。

![Pascal 帕斯卡架构主要创新](images/04History07.png)

GP100 Pascal 由图形处理集群（GPCs）、纹理处理集群（TPCs）、流式多处理器（SMs）和内存控制器组成。一个完整的 GP100 由 6 个 GPCs、60 个 Pascal SMs、30 个 TPCs（每个都包括 2 个 SMs）和 8 个 512 位内存控制器（总共 4096 位）组成。每个 GPC 都有 10 个 SMs，每个 SM 有 64 个 CUDA 核和 4 个纹理单元，拥有 60 个 SMs，共有 3840 个单精度 CUDA Cores 和 240 个纹理单元。每个内存控制器都连接到 512 KB 的 L2 高速缓存上，每个 HBM2 DRAM 都由一对内存控制器控制，总共包含 4096 KB L2 高速缓存。

![Pascal 帕斯卡架构](images/04History08.png)

Pascal 架构在 SM 内部作了进一步精简，整体思路是 SM 内部包含的硬件单元类别减少，因为芯片制程工艺的进步，SM 数量每一代都在增加。单个 SM 只有 64 个 FP32 CUDA Cores，相比 Maxwell 的 128 核和 Kepler 的 192 核，数量少了很多，并且 64 个 CUDA Cores 分为了两个区块，每个处理块有 32 个单精度 CUDA Cores、一个指令缓冲区、一个 Warp 调度器和两个调度单元（Dispatch Unit）。分成两个区块之后，Register File 保持相同大小，每个线程可以使用更多的寄存器，单个 SM 可以并发更多的 thread/warp/block，进一步增加并行处理能力。

增加 32 个 FP64 CUDA Cores（DP Unit），FP32 CUDA Core 具备处理 FP16 的能力。此外，每个 SM 具有 32 个双精度（FP64）CUDA Cores，使得 GPU 更有效地处理双精度计算任务。与精度更高的 FP32 或 FP64 相比，存储 FP16 数据可以减少神经网络的内存使用，从而允许训练和部署更大的网络。为加速深度学习支持 FP16，与 FP32 相比可以提高 2 倍性能，同时数据传输需要的时间更少。

![Pascal 帕斯卡架构 SM](images/04History09.png)

> 布莱斯·帕斯卡（Blaise Pascal）是 17 世纪法国数学家、物理学家、哲学家和神学家，视为文艺复兴时期最重要的思想家之一。他在多个领域都有重要的贡献，被认为是现代概率论和流体力学的奠基人之一。主要成就包括：
> 
> 1. 发明了帕斯卡三角形，这是一个数学工具，被广泛用于组合数学和概率论中。
> 
> 2. 提出了帕斯卡定律，描述了液体在容器中的压力传递规律，对流体力学的发展产生了重要影响。
>
> 3. 发展了概率论，提出了帕斯卡概率论，为后来的概率统计学奠定了基础。
> 
> 4. 在哲学和神学领域，他提出了帕斯卡赌注，探讨了信仰与理性的关系，对基督教神学产生了深远的影响。

由于多机之间采用 InfiniBand 和 100 GB Ethernet 通信，单个机器内单 GPU 到单机 8 GPU，PCIe 带宽成为瓶颈，因此 Pascal 架构首次提出 NVLink，针对多 GPU 和 GPU-to-CPU 实现高带宽连接。NVLink 用以单机内多 GPU 内的点对点通信，带宽达到 160 GB/s，大约是 PCIe 3x16 的 5 倍，减少数据传输的延迟，避免大量数据通过 PCIe 回传到 CPU 的内存中，导致数据重复搬运，实现 GPU 整个网络的拓扑互联。在实际训练大模型的过程中，带宽会成为分布式训练系统的主要瓶颈，从而使得 NVLink 成为一项具有重要意义的创新。

![Pascal 帕斯卡架构 NVLink](images/04History10.png)

## Volta 架构

2017 年英伟达提出 Volta 架构，GV100 GPU 有 21.1 亿个晶体管，使用 TSMC 12 nm 工艺。伏特架构做了以下创新：

（1）CUDA Core 拆分，分离 FPU 和 ALU，取消 CUDA Core 整体的硬件概念，一条指令可以同时执行不同计算，同时对 CUDA 应用程序并行线程更进一步，提高了 CUDA 平台的灵活性、生产力和可移植性；

（2）提出独立线程调度，改进单指令多线程 SIMT 模型架构，使得每个线程都有独立的 PC（Program Counter）和 Stack，程序中并行线程之间更细粒度的同步和协作；

（3）专门为深度学习优化了 SM 架构，针对 AI 计算首次提出第一代张量核心 Tersor Core，提高深度学习计算中卷积运算进行加速；

（4）对 NVLink 进行改进，提出第二代 NVLink，一个 GPU 可以连接 6 个 NVLink，而不是 Pascal 时代的 4 个，16 GB HBM2 内存子系统提供了 900GB/秒的峰值内存带宽；

（5）提出 MPS 概念，在多个应用程序单独未充分利用 GPU 执行资源时，允许多个应用程序同时共享 GPU 执行资源，使得多进程服务可以更好的适配到云厂商进行多用户租赁，客户端数量从 Pascal 上的 16 个增加到 Volta 上的 48 个，支持多个单独的推理任务并发地提交给 GPU，提高 GPU 的总体利用率；

（6）结合 Volta 架构新特性优化 GPU 加速库版本，如 cuDNN、cuBLAS 和 TensorRT，为深度学习推理和高性能计算（HPC）应用程序提供更高的性能。英伟达 CUDA 9.0 版本提供了新的 API 支持 Volta 特性，更简单的可编程性。英伟达 TensorRT 是一款用于高性能深度学习推理的 SDK，包含深度学习推理优化器和运行时，可为推理应用程序提供低延迟和高吞吐量。

![Volta 伏特架构主要改进](images/04History11.png)

> 亚历山大·伏特（Alessandro Volta）是 18 世纪意大利物理学家，被誉为电池之父。他是电学领域的先驱之一，发明了第一种真正意义上的化学电池，被称为伏特电池，为电化学和现代电池技术的发展奠定了基础。主要成就包括：
> 
> 1. 发明了伏特电堆，这是由多个铜和锌片交替堆叠而成的装置，能够产生持续的电流，是第一个实用的化学电池。
> 
> 2. 提出了静电感应理论，探讨了静电现象的本质，对电学理论的发展产生了重要影响。
> 
> 3. 研究了气体的电学性质，发现了甲烷和氧气的反应可以产生火花，为后来的火花塞技术和火花点火系统的发展做出了贡献。

与上一代 Pascal GP100 GPU 一样，GV100 GPU 有 6 个 GPU 处理集群（GPCs），每个 GPC 有 7 个纹理处理集群（TPCs）、14 个流式多处理器（SMs），以及内存控制器。

![volta 伏特架构](images/04History12.png)

Volta 伏特架构 SM 结构相比前几代架构，SM 的数目明显增多，SM 被划分为四个处理块，单个 SM 中包含 4 个 Warp Schedule，4 个 Dispatch Unit，64 个 FP32 Core（4*16），64 个 INT32 Core（4*16），32 个 FP64 Core（4*8），8 个 Tensor Core（4*2），32 个 LD/ST Unit（4*8），4 个 SFU，FP32 和 INT32 两组运算单元独立出现在流水线中，每个 Cycle 都可以同时执行 FP32 和 INT32 指令，因此每个时钟周期可以执行的计算量更大。Volt 架构新增了混合精度张量核心（Tensor Core）以及高性能 L1 数据缓存和新的 SIMT 线程模型。单个 SM 通过共享内存和 L1 资源的合并，相比 GP100 64 KB 的共享内存容量，Volta 架构增加到 96KB。

![Volta 伏特架构 SM 结构](images/04History13.png)

新的张量核心使 Volta 架构得以训练大型神经网络，GPU 并行模式可以实现深度学习功能的通用计算，最常见卷积/矩阵乘（Conv/GEMM）操作，依旧被编码成融合乘加运算 FMA（Fused Multiply Add），硬件层面还是需要把数据按照：寄存器-ALU-寄存器-ALU-寄存器方式来回来回搬运数据，因此专门设计 Tensor Core 实现矩阵乘计算。

英伟达计算硬件模型从 SIMT 发展成为了 SIMT+DSA 的混合，每个张量核心单时钟周期内执行 64 个浮点 FMA 操作，而 SM 中的 8 个张量核心单时钟周期总共执行 512 个 FMA 操作（或 1024 个单独的浮点操作）。每个张量核心在一个 4x4 矩阵上操作，并执行计算：$D=A×B+C$，输入 A 和 B 矩阵是 FP16，而计算结果矩阵 C 和 D 可以是 FP16 或 FP32 矩阵，极大地减少了系统内存的开销，一个时钟周期内可以执行更多的矩阵运算，使得 GPU 在能耗上更有优势。CUDA 9 C++ API 有专门的矩阵乘和存储操作，有效地使用 CUDA-C++程序中的张量核心，同时 cuBLAS 和 cuDNN 库利用张量核进行深度学习研究。

![Volta 伏特架构 Tensor Core 计算](images/04History14.png)

英伟达伏特架构的 GPU 以 Tesla V100 Powered DGX Station 的形式对外出售工作站。此时不再使用 PCIe 连接 GPU，而是将多个 GPU 直接封装在同一块主板上，第二代 NVLink 每个连接提供双向各自 25 GB/s 的带宽，并且一个 GPU 可以接 6 个 NVLink，专门用于 GPU-GPU 通信，同时允许从 CPU 直接加载/存储/原子访问到每个 GPU 的 HBM2 内存。

![Volta 伏特架构 V100](images/04History15.png)

此外，NVSwitch1.0 技术是 Volta 架构中的一项重要创新，旨在提高 GPU 之间的通信效率和性能。NVSwitch1.0 可以支持多达 16 个 GPU 之间的通信，可以实现 GPU 之间的高速数据传输，提高系统的整体性能和效率，适用于需要大规模并行计算的场景，比如人工智能训练和科学计算等领域。

![12 个 NVSwitch 1.0 连接 16 个 V100](images/04History16.png)

英伟达 Tesla V100 将深度学习的新架构特性与 GPU 计算性能相结合，提供了更高的神经网络训练和推理性能。NVLink 使多 GPU 系统提供了性能可伸缩性，同时 CUDA 编程的灵活性允许新算法快速开发和部署，满足了人工智能、深度学习系统和算法的训练和推断的持续需求。

## Turing 架构

2018 年 Turing 图灵架构发布，采用 TSMC 12 nm 工艺，总共 18.6 亿个晶体管。在 PC 游戏、专业图形应用程序和深度学习推理方面，效率和性能都取得了重大进步。相比上一代 Volta 架构主要更新了 Tensor Core（专门为执行张量/矩阵操作而设计的专门执行单元，深度学习计算核心）、CUDA 和 CuDNN 库的不断改进，更好地应用于深度学习推理。RT Core（Ray Tracing Core）提供实时的光线跟踪渲染，包括具有物理上精确的投影、反射和折射，更逼真的渲染物体和环境。支持 GDDR6 内存，与 GDDR5 内存相比，拥有 14 Gbps 传输速率，实现了 20%的的效率提升。NVLink2.0 支持 100 GB/s 双向带宽，使特定的工作负载能够有效地跨两个 GPU 进行分割并共享内存。

![NVIDIA Turing TU102 GPU Die](images/04History17.png)

TU102 GPU 包括 6 个图形处理集群（GPCs）、36 个纹理处理集群（TPCs）和 72 个流式多处理器（SMs）。每个 GPC 包括一个专用光栅引擎和 6 个 TPC，每个 TPC 包括两个 SMs。每个 SM 包含 64 个 CUDA 核心、8 个张量核心、一个 256 KB 的寄存器文件、4 个纹理单元和 96 KB 的 L1/共享内存，这些内存可以根据计算或图形工作负载配置为不同的容量。因此总共有 4608 个 CUDA 核心、72 个 RT 核心、576 个张量核心、288 纹理单元和 12 个 32 位 GDDR6 内存控制器（总共 384 位）。

![Turing 图灵架构](images/04History18.png)

> 艾伦·图灵（Alan Turing）是 20 世纪英国数学家、逻辑学家和密码学家，被誉为计算机科学之父。他在计算理论和人工智能领域做出了开创性的工作，对现代计算机科学的发展产生了深远影响。主要成就包括：
> 
> 1. 发展了图灵机概念，这是一种抽象的数学模型，被认为是计算机的理论基础，为计算机科学奠定了基础。
> 
> 2. 在第二次世界大战期间，他领导了英国破解德国恩尼格玛密码的团队，对盟军在战争中的胜利做出了重要贡献。
> 
> 3. 提出了图灵测试，用来衡量机器是否具有智能，为人工智能领域的发展提供了重要思想。
> 
> 4. 在逻辑学领域，他提出了图灵判定问题，对计算机可解性和不可解性做出了重要贡献。

随着神经网络模型的量化部署逐渐成熟，Turing 架构中的 Tensor Core（张量核心）增加了对 INT8/INT4/Binary 的支持，加速神经网络训练和推理函数的矩阵乘法核心。一个 TU102 GPU 包含 576 个张量核心，每个张量核心可以使用 FP16 输入在每个时钟执行多达 64 个浮点融合乘法加法（FMA）操作。SM 中 8 个张量核心在每个时钟中总共执行 512 次 FP16 的乘法和累积运算，或者在每个时钟执行 1024 次 FP 运算，新的 INT8 精度模式以两倍的速率工作，即每个时钟进行 2048 个整数运算。Tensor Core 用于加速基于 AI 的英伟达 NGX 功能，增强图形、渲染和其它类型的客户端应用程序，包括 DLSS（深度学习超级采样）、 AI 绘画、AI Super Rez（图像/视频超分辨率）和 AI Slow-Mo（视频流插帧）。

每个 SMs 分别有 64 个 FP32 核和 64 个 INT32 核，还包括 8 个混合精度的张量核（Tensor Core），每个 SM 被分为四个块，每个块包括一个新的 L0 指令缓存和一个 64 KB 的寄存器文件。四个块共享一个 96 KB L1 数据缓存/共享内存。传统的图形工作负载将 96 KB 的 L1/共享内存划分为 64 KB 的专用图形着色器 RAM 和 32 KB 的用于纹理缓存和寄存器文件溢出区域。计算工作负载可以将 96 KB 划分为 32 KB 共享内存和 64 KB L1 缓存，或者 64 KB 共享内存和 32 KB L1 缓存。

![Turing Tensor Core & RT Core](images/04History19.png)

RT Core 主要用于三角形与光线求交点，并通过 BVH（Bounding Volume Hierarchy）结构加速三角形的遍历，由于布置在 block 之外，相对于普通 ALU 计算来说是异步的，包括两个部分，一部分检测碰撞盒来剔除面片，另一部分做真正的相交测试。RT Core 的使用，使 SM 在很大程度上可以用来做图形计算之外的工作。

> **Bounding Volume Hierarchy（BVH）结构**
> 
> 光线追踪（Ray Tracing）中的 Bounding Volume Hierarchy（BVH）结构是一种用于加速光线追踪算法的数据结构。BVH 通过将场景中的物体分层组织成包围盒（Bounding Volume）的层次结构，从而减少光线与物体的相交测试次数，提高光线追踪的效率。
> 
> 在 BVH 结构中，每个节点都代表一个包围盒，该包围盒可以包含多个物体或其他子包围盒。通过递归地构建 BVH 树，可以将场景中的物体分层组织成一个高效的数据结构，以便快速地确定光线与哪些物体相交，从而减少需要测试的物体数量，提高光线追踪的效率。

当增加 RT Core 之后实现硬件光线追踪，当 RTX 光线追踪技术打开时场景中人物和光线更加逼真，火焰可以在车身上清晰的看到。虽然光线追踪可以产生比栅格化更真实的图像，但是计算密集型使得混合渲染是更优的技术路线，光线追踪用在比栅格化更有效的地方，如渲染反射、折射和阴影。光线追踪可以运行在单个 Quadro RTX 6000 或 GeForce RTX 2080 Ti GPU 上，渲染质量几乎等同于电影实拍效果。

![RT Core 效果对比](images/04History20.png)

除了为高端游戏和专业图形带来革命性的新功能外，Turing 还提供了多精度计算，随着英伟达深度学习平台的持续推进，如 TensorRT 5.0 和 CUDA 10 技术的进步，基于英伟达 GPU 的推理解决方案显著降低了数据中心的成本、规模和功耗。

## Ampere 架构

2020 年 Ampere 安培架构发布，Ampere 架构主要有以下特性：

1）超过 540 亿个晶体管，使其成为 2020 年世界上最大的 7 nm 处理器（英伟达 A100）；

2）提出 Tensor Core3.0，新增 TF32（TensorFloat-32）包括针对 AI 的扩展，可使 FP32 精度的 AI 性能提高 20 倍；

3）多实例 GPU（Multi-Instance GPU，MIG）将单个 A100 GPU 划分为多达 7 个独立的 GPU，为不同任务提供不同算力，为云服务器厂商提供更好的算力切分方案；

4）提出 NVLink3.0 和 NV-Switch，NV-Switch 可以将多台机器进行互联，将 GPU 高速连接的速度加倍，可在服务器中提供有效的性能扩展；

5）利用 AI 数学计算中固有的稀疏特性将性能提升一倍。以上改进使 Ampere 成为新一代数据中心和云计算 GPU 架构，可用于 AI 和高性能计算场景。

![Ampere 安培架构主要特性](images/04History21.png)

> 安德烈-玛丽·安培（André-Marie Ampère）是 19 世纪法国物理学家和数学家，被誉为电磁学之父。他对电流和磁场之间的相互作用进行了深入研究，提出了安培定律，对电磁理论的发展做出了重要贡献。主要成就包括：
> 
> 1. 提出了安培定律，描述了电流元素之间的相互作用，为电磁感应和电磁场的研究奠定了基础。
> 
> 2. 发展了电动力学理论，将电流和磁场的关系系统化，并提出了电流环的磁场理论。
> 
> 3. 研究了电磁感应现象，揭示了磁场和电场之间的关系，为后来法拉第的电磁感应定律的提出奠定了基础。
> 
> 4. 对电磁学和热力学等领域都有重要贡献，被认为是 19 世纪最杰出的物理学家之一。

英伟达 A100 GPU 包括 8 个 GPC，每个 GPC 包含 8 个 TPC，每个 TPC 包含 2S 个 SMs/，每个 GPC 包含 16 个 SM/GPC，整个 GPU 拥有 128 个 SMs。每个 SM 有 64 个 FP32 CUDA 核心，总共 8192 FP32 CUDA 核心。Tensor Core3.0，总共 512 个。6 个 HBM2 存储栈，12 个 512 位内存控制器，内存可达到 40 GB。第三代 NVLink，GPU 和服务器双向带宽为 4.8 TB/s，GPU 之间的互联速度为 600 GB/s。A100 SM 拥有 192 KB 共享内存和 L1 数据缓存，比 V100 SM 大 1.5 倍。

![Ampere 安培架构](images/04History22.png)

A100 Tensor Core3.0 增强操作数共享并提高计算效率，引入了 TF32、BF16 和 FP64 数据类型的支持。平时训练模型的过程中使用更多的是 FP32 和 FP16，TF32 在指数位有 8 位，FP16 在指数为有 5 位，因此 FP32 的位宽比 FP16 更多，小数位决定精度，FP32 在小数位有 23 位，FP16 只有 10 位，在 AI 训练的过程中很多时候 FP16 是够用的，但是动态范围会有限制，因此提出 TF32，指数位保持和 FP32 相同，小数位和 FP16 保持相同，BF16 的指数位和 FP32、TF32 相同，但是小数位少了三位。数百个张量核并行运行，大幅提高吞吐量和计算效率。

![Ampere 架构 TF32、BF16 和 FP64](images/04History23.png)

A100 FP32 FFMA，INT8、INT4 和 Binary 分别提高了 32x、64x 和 256x，与 Volta 架构一样，自动混合精度（AMP）允许用户使用与 FP16 相结合的混合精度来进行 AI 训练，使用 AMP 之后 A100 提供了比 TF32 快 2 倍的张量核心性能。

![Ampere 架构 A100 支持精度](images/04History24.png)

Tensor Core 除了执行乘法和加法操作之外还可以支持稀疏化结构矩阵（Sparse Tensor），实现细粒度的结构化稀疏，支持一个 2:4 的结构化稀疏矩阵与另一个稠密矩阵直接相乘。一种常见的方法是利用稀疏矩阵的结构特点，只对非零元素进行计算，从而减少计算量。一个训练得到的稠密矩阵在推理阶段经过剪枝之后会变成一个稀疏化矩阵，然后英伟达架构对矩阵进行压缩后变成一个稠密的数据矩阵和一个 indices，索引压缩过的数据方便检索记录，最后进行矩阵乘。

![Ampere 架构稀疏化流程](images/04History25.png)

A100 张量核心 GPU 可以被分为 7 个 GPU 实例并被不同任务使用，每个实例的处理器在整个内存系统中都有单独且相互隔离的路径，片上交叉端口、L2 缓存、内存控制器和 DRAM 地址总线都被唯一地分配给一个单独的实例，确保单个用户的工作负载可以在可预测的吞吐量和延迟下运行，同时具有相同的 L2 缓存分配和 DRAM 带宽，即使其他任务正在读写缓存或 DRAM 接口。用户可以将这些虚拟 GPU 实例当成真的 GPU 进行使用，为云计算厂商提供算力切分和多用户租赁服务。

![Ampere 架构多实例分割虚拟 GPU](images/04History26.png)

DGX A100 是英伟达专门构建的第三代 AI 系统，在单个系统中可以提供 5 PFLOPS（petaflop）性能，通过一种新的基础设施结构，彻底改变了企业数据中心，旨在将所有 AI 工作负载统一在一个新的通用平台和架构上。A100 以整机的形式出售，最上面是散热器，中间的 A100 芯片不再通过 PCIe 进行连接，而是直接封装在主板上，这样便于在同一个节点上进行模型并行，但是跨节点跨机器之间训练大模型时带宽就会成为整个大模型训练的瓶颈。内存高达 1TB 或者 2TB，可以直接将数据全部加载到 CPU 里面，然后再不断回传到 GPU 中，加速大模型训练。

![A100 硬件规格](images/04History27.png)

## Hopper 架构

2022 年 Hopper 赫柏架构发布，英伟达 Grace Hopper Superchip 架构将英伟达 Hopper GPU 的突破性性能与英伟达 Grace CPU 的多功能性结合在一起，在单个超级芯片中与高带宽和内存一致的英伟达 NVLink Chip-2-Chip（C2C）互连，并且支持新的英伟达 NVLink 切换系统，CPU 和 GPU、GPU 和 GPU 之间通过 NVLink 进行连接，数据的传输速率高达 900 GB/s，解决了 CPU 和 GPU 之间数据的时延问题，跨机之间通过 PCIe5 进行连接。

![Inside NVIDIA’s First GPU-CPU Superchip](images/04History28.png)

Hopper 架构是第一个真正的异构加速平台，适用于高性能计算（HPC）和 AI 工作负载。英伟达 Grace CPU 和英伟达 Hopper GPU 实现英伟达 NVLink-C2C 互连，高达 900 GB/s 的总带宽的同时支持 CPU 内存寻址为 GPU 内存。NVLink4.0 连接多达 256 个英伟达 Grace Hopper 超级芯片，最高可达 150 TB 的 GPU 可寻址内存。

| H100 | 参数 |
| --- | --- |
| NVIDIA Grace CPU | 72 个 Arm Neoverse V2 内核，每个内核 Armv9.0-A ISA 和 4 个 128 位 SIMD 单元 |
|  | 512 GB LPDDR5X 内存，提供高达 546 GB/s 的内存带宽 |
|  | 117MB 的 L3 缓存，内存带宽高达 3.2 TB/s |
|  | 64 个 PCIe Gen5 通道 |
| NVIDIA Hopper GPU | 144 个第四代 Tensor Core、Transformer Engine、DPX 和 3 倍高 FP32 的 FP64 的 SM |
|  | 96 GB HBM3 内存提供高达 3000 GB/s 的速度 |
|  | 60 MB 二级缓存 |
|  | NVLink 4 和 PCIe 5 |
| NVIDIA NVLink-C2C | Grace CPU 和 Hopper GPU 之间硬件一致性互连 |
|  | 高达 900 GB/s 的总带宽、450 GB/s/dir |
|  | 扩展 GPU 内存功能使 Hopper GPU 能够将所有 CPU 内存寻址为 GPU 内存。每个 Hopper CPU 可以在超级芯片内寻址多达 608 GB 内存 |
| NVIDIA NVLink 切换系统 | 使用 NVLink 4 连接多达 256 个 NVIDIA Grace Hopper 超级芯片 |
|  | 每个连接 NVLink 的 Hopper GPU 都可以寻址网络中所有超级芯片的所有 HBM3 和 LPDDR5X 内存，最高可达 150 TB 的 GPU 可寻址内存 |

H100 一共有 8 组 GPC、66 组 TPC、132 组 SM，总计有 16896 个 CUDA 核心、528 个 Tensor 核心、50MB 二级缓存。显存为新一代 HBM3，容量 80 GB，位宽 5120-bit，带宽高达 3 TB/s。

![Hopper 赫柏架构](images/04History29.png)

**（注意：上面的图是GH100的图，而不是H100的图）**

> 格蕾丝·赫希贝尔·赫柏（Grace Hopper）是 20 世纪美国计算机科学家和海军军官，被誉为计算机编程先驱和软件工程的奠基人之一。在 1934 年获得了耶鲁大学数学博士学位，成为该校历史上第一位女性获得博士学位的人。在计算机领域做出了重要贡献，尤其在编程语言和软件开发方面有突出成就，被尊称为“软件工程之母”和“编程女王”。主要成就包括：
> 
> 1. 开发了第一个编译器，将高级语言翻译成机器码，这项创新大大简化了编程过程，为软件开发奠定了基础。
> 
> 2. 提出了 COBOL（通用商业导向语言）编程语言的概念和设计，这是一种面向商业应用的高级语言，对商业和金融领域的计算机化起到了重要作用。
> 
> 3. 在计算机科学教育和推广方面做出了杰出贡献，她致力于将计算机科学普及到更广泛的人群中，并激励了许多人进入这一领域。
> 
> 4. 作为美国海军的一名军官，她参与了多个计算机课程，包括 UNIVAC 和 Mark 系列计算机的开发，为军事和民用领域的计算机化做出了贡献。

具体到 SM 结构，Hopper 赫柏架构 FP32 Core 和 FP64 Core 两倍于 Ampere 架构，同时采用 Tensor Core4.0 使用新的 8 位浮点精度（FP8），可为万亿参数模型训练提供比 FP16 高 6 倍的性能。FP8 用于 Transformer 引擎，能够应用 FP8 和 FP16 的混合精度模式，大幅加速 Transformer 训练，同时兼顾准确性。FP8 还可大幅提升大型语言模型推理的速度，性能较 Ampere 提升高达 30 倍。新增 Tensor Memory Accelerator，专门针对张量进行数据传输，更好地加速大模型。

| Hopper 赫柏架构 SM 硬件单元 | Hopper 赫柏架构每个 Process Block | 相比 Ampere 架构 |
| --- | --- | --- |
| 4 个 Warp Scheduler，4 个 Dispatch Unit  | 1 个 Warp Scheduler，1 个 Dispatch Unit | 相同 |
| 128 个 FP32 Core（4 * 32）| 32 个 FP32 Core | x2 |
| 64 个 INT32 Core（4 * 16）| 16 个 INT32 Core | 相同 |
| 64 个 FP64 Core（4 * 16）| 16 个 FP32 Core | x2 |
| 4 个 Tensor Core4.0（4 * 1）| 1 个 Tensor Core | Tensor Core3.0 |
| 32 个 LD/ST Unit（4 * 8）| 8 个 LD/ST Unit | 相同 |
| 16 个 SFU（4 * 4）| 4 个 SFU | 相同 |
| Tensor Memory Accelerator |  | 新增 |

![Hopper 赫柏架构 SM](images/04History30.png)

NVIDIA Quantum-2 Infiniband 是英伟达推出的一种高性能互连技术，用于数据中心和高性能计算环境中的互连网络，具有高性能、低延迟、高可靠性和支持异构计算等特点，主要用于连接计算节点、存储系统和其他关键设备，以实现高速数据传输和低延迟通信。

NVIDIA BlueField-3 DPU（Data Processing Unit）是一种数据处理单元，提供数据中心的网络、存储和安全加速功能。BlueField-3 DPU 结合了网络接口控制器（NIC）、存储控制器、加密引擎和智能加速器等功能于一体，为数据中心提供了高性能、低延迟的数据处理解决方案。

![H100 异构系统](images/04History31.png)

NVIDIA CUDA 平台针对 NVIDIA Grace CPU，NVIDIA Grace Hopper Superchip 和 NVIDIA NVLink Switch 系统进行了优化，使得 NVIDIA CUDA 发展成为一个全面、高效、高性能的加速计算平台，为开发人员在异构平台上加速应用程序提供了最佳的体验。

![NVIDIA CUDA Platform and its ecosystem](images/04History32.png)

基于 Hopper 架构，英伟达推出 NVIDIA H100 高性能计算加速器，旨在为各种规模的计算工作负载提供出色的性能和效率。在单服务器规模下，结合主流服务器使用 H100 加速卡可以提供强大的计算能力，加速各种计算密集型工作负载。在多服务器规模下，组成 GPU 集群的多块 H100 加速卡可以构建高性能计算集群，支持分布式计算和并行计算，提高整体计算效率。而在超级计算规模下，大量 H100 加速卡组成的超级计算集群可以处理极端规模的计算任务，支持复杂的科学计算和研究。

从单服务器到多服务器再到超级计算规模（Mainstream Servers to DGX to DGX SuperPOD），NVIDIA H100 在不同层次和规模下展现出色的计算性能和效率，满足各种计算需求和业务目标。企业可以根据自身需求和预算选择适合的 NVIDIA H100 解决方案，加速其计算任务和推动 AI 领域的发展。

![H100-Mainstream Servers to DGX to DGX SuperPOD](images/04History33.png)

## Blackwell 架构

2024 年 3 月，英伟达发布 Blackwell 架构，专门用于处理数据中心规模的生成式 AI 工作流，能效是 Hopper 的 25 倍，新一代架构在以下方面做了创新：

- **新型 AI 超级芯片**：Blackwell 架构 GPU 具有 2080 亿个晶体管，采用专门定制的台积电 4NP 工艺制造。所有 Blackwell 产品均采用双倍光刻极限尺寸的裸片，通过 10 TB/s 的片间互联技术连接成一块统一的 GPU。

- **第二代 Transformer 引擎**：将定制的 Blackwell Tensor Core 技术与英伟达 TensorRT-LLM 和 NeMo 框架创新相结合，加速大语言模型 (LLM) 和专家混合模型 (MoE) 的推理和训练。

- **第五代 NVLink**：为了加速万亿参数和混合专家模型的性能，新一代 NVLink 为每个 GPU 提供 1.8TB/s 双向带宽，支持多达 576 个 GPU 间的无缝高速通信，适用于复杂大语言模型。

- **RAS 引擎**：Blackwell 通过专用的可靠性、可用性和可服务性 (RAS) 引擎增加了智能恢复能力，以识别早期可能发生的潜在故障，从而更大限度地减少停机时间。

- **安全 AI**：内置英伟达机密计算技术，可通过基于硬件的强大安全性保护敏感数据和 AI 模型，使其免遭未经授权的访问。

- **解压缩引擎**：拥有解压缩引擎以及通过 900GB/s 双向带宽的高速链路访问英伟达 Grace CPU 中大量内存的能力，可加速整个数据库查询工作流，从而在数据分析和数据科学方面实现更高性能。

![英伟达 DPU CPU+GPU GPU](images/04History34.jpg)

> 大卫·哈罗德·布莱克韦尔（David Harold Blackwell）是 20 世纪美国著名的数学家和统计学家，他在统计学领域做出了卓越的贡献，被誉为统计学的巨匠，第一个非裔美国人当选为美国国家科学院院士，也是第一个获得美国数学学会最高奖——Leroy P. Steele 奖章的非裔美国人。主要成就包括：
> 
> 1.  在贝叶斯统计学领域做出了开创性的工作，提出了许多重要的方法和理论，推动了贝叶斯分析在统计学中的发展。
> 
> 2.  在信息论方面的研究成果为该领域的发展做出了重要贡献，提供了许多重要的理论基础和方法。

英伟达 GB200 Grace Blackwell 超级芯片通过 900GB/s 超低功耗的片间互联，将两个英伟达 B200 Tensor Core GPU 与英伟达 Grace CPU 相连。在 90 天内训练一个 1.8 万亿参数的 MoE 架构 GPT 模型，需要 8000 个 Hopper 架构 GPU，15 兆瓦功率，Blackwell 架构只需要 2000 个 GPU，以及 1/4 的能源消耗。8 年时间，从 Pascal 架构到 Blackwell 架构，英伟达将 AI 计算性能提升了 1000 倍！

![8 年时间 AI 计算性能提升了 1000 倍](images/04History35.png)

英伟达 GB200 NVL72 集群以机架形式设计连接 36 个 GB200 超级芯片(36 个 Grace cpu 和 72 个 Blackwell GPU)。GB200 NVL72 是一款液冷、机架型 72 GPU NVLink，可以作为单个大规模 GPU，提供比上一代 HGX H100 实现 30 倍的实时万亿参数 LLM 推理，加速下一代 AI 和加速计算。

![英伟达 GB200 NVL72 集群](images/04History36.png)

| | **GB200 NVL72** | **GB200 Grace Blackwell Superchip** |
| --- | --- | --- |
| **Configuration** | 36 Grace CPU : 72 Blackwell GPUs | 1 Grace CPU : 2 Blackwell GPU |
| **FP4 Tensor Core2** | 1,440 PFLOPS | 40 PFLOPS |
| **FP8/FP6 Tensor Core2** | 720 PFLOPS | 20 PFLOPS |
| **INT8 Tensor Core2** | 720 POPS | 20 POPS |
| **FP16/BF16 Tensor Core2** | 360 PFLOPS | 10 PFLOPS |
| **TF32 Tensor Core2** | 180 PFLOPS | 5 PFLOPS |
| **FP64 Tensor Core** | 3,240 TFLOPS | 90 TFLOPS |
| **GPU Memory &#124; Bandwidth** | Up to 13.5 TB HBM3e &#124; 576 TB/s | Up to 384 GB HBM3e &#124; 16 TB/s |
| **NVLink Bandwidth** | 130TB/s | 3.6TB/s |
| **CPU Core Count** | 2,592 Arm Neoverse V2 cores | 72 Arm Neoverse V2 cores |
| **CPU Memory &#124; Bandwidth** | Up to 17 TB LPDDR5X &#124; Up to 18.4 TB/s | Up to 480GB LPDDR5X &#124; Up to 512 GB/s |
| **1. Preliminary specifications. May be subject to change. 1. With sparsity.** |  |  |

随着大模型（LLM）参数量增长对算力的需求，英伟达在存储带宽和内存方面不断创新，P100 上首次使用 HBM2，A100 使用 HBM2e，H100 使用 HBM3，H200 和 B100 使用 HBM3e。

![英伟达为满足模型需要不断创新](images/04History37.jpg)

英伟达 Blackwell HGX B200 和 HGX B100 在生成式 AI 、数据分析和高性能计算方面具有相同的突破性进展。HGX B200 是基于 8 个 B200 x86 平台，提供 144 petaFLOPs 的 AI 性能，每个 GPU 最高可配置 1000 瓦。HGX B100 是基于 8 个 B100 x86 平台，提供 112 petaFLOPs 的 AI 性能，每个 GPU 最高可配置为 700 瓦。

|  |  HGX B200 |  HGX B100 |
| --- | --- | --- |
| Blackwell GPUs | 8 | 8 |
| FP4 Tensor Core | 144 PetaFLOPS | 112 PetaFLOPS |
| FP8/FP6/INT872 | 72 PetaFLOPS | 56 PetaFLOPS |
| Fast Memory | Up to 1.5 TB | Up to 1.5TB |
| Aggregate Memory Bandwidth | Up to 64 TB/s | Up to 64 TB/s |
| Aggregate NVLink Bandwidth | 14.4 TB/s | 14.4 TB/s |
| Per GPU Specifications |  |  |
| FP4 Tensor Core | 18 petaFLOPS | 14 petaFLOPS |
| FP8/FP6 Tensor Core | 9 petaFLOPS | 7 petaFLOPS |
| INT8 Tensor Core | 9 petaOPS | 7 petaOPS |
| FP16/BF16 Tensor Core | 4.5 petaFLOPS | 3.5 petaFLOPS |
| TF32 Tensor Core | 2.2 petaFLOPS | 1.8 petaFLOPS |
| FP64 Tensor Core | 40 teraFLOPS | 30 teraFLOPS |
| GPU memory &#124; Bandwidth | Up to 192 GB HBM3e &#124; Up to 8 TB/s |  |
| Max thermal design power (TDP) | 1,000W | 700W |
| Interconnect | NVLink: 1.8TB/s, PCIe Gen6: 256GB/s | NVLink: 1.8TB/s, PCIe Gen6: 256GB/s |
| Server options | NVIDIA HGX B200 partner and NVIDIA-Certified Systems with 8 GPUs | NVIDIA HGX B100 partner and NVIDIA-Certified Systems with 8 GPUs |
- Preliminary specifications subject to change.
- All petaFLOPS and petaOPS are with Sparsity except FP64 which is dense.

![GPT-3 Inference Performance](images/04History38.jpg)

## 小结与思考

本节主要回顾了从 2010 年到 2024 年英伟达 GPU 架构的发展，其中有几个比较重要的时间节点和事件：

- 2010 年提出 Fermi 架构，开启了架构演进的进程，属于首个完整的 GPU 计算架构，里面提出的新概念一直沿用至今。

- 2016 年提出 Pascal 架构，首次提出 NVLink，双向互联带宽达到 160 GB/s 对 AI 领域产生重大影响，是具有历史性意义的里程碑架构。

- 2017 年提出 Volt 架构，首次提出张量核心（Tensor Core），专门针对神经网络矩阵卷积运算进行加速的第一代核心，同时提出 NVSwitch1.0，提高 GPU 之间的通信效率和性能。

- 2018 年提出 Turing 架构，在消费级显卡上提出 RT Core 以实现硬件光线追踪。

- 2020 年 Ampere 架构因多用户 GPU 实例在 AI 云应用厂商中广泛采用。

- 2022 年 Hopper 架构实现了 CPU 和 GPU 异构，CPU 与 GPU 实现英伟达 NVLink-C2C 互连。

- 2024 年 Blackwell 架构 GPU，英伟达将 AI 计算性能提升了 1000 倍，进一步为生成式 AI 与大模型提供算力支持。

### CUDA Core & Tensor Core

Cuda Core 和 Tensor Core 都是运算单元，与硬件相关。随着科学计算迅速发展，为了使用 GPU 的高算力，需要将科学计算任务适配成图形图像任务，Cuda Core 属于全能通用型浮点运算单元，用于加、乘、乘加运算。随着 AI 迅速发展，对矩阵乘法的算力需求不断增大，Tensor Core 专门为深度学习矩阵运算设计，适用于在高精度矩阵运算。以 Hopper 架构为例，每个 SM 有 128 个 CUDA Core，4 个 Tensor Core，Tensor Core 相比较支持的精度更多，而且 Tensor Core 是可编程的，除了使用 CUDA API 调用 Tensor Core，如 cublas、cudnn 等，还可以使用 WMMA (Warp-level Matrix Multiply Accumulate) API、MMA (Matrix Multiply Accumulate) PTX 进行编程。

|  | Blackwell | Hopper | Ampere | Turing | Volta |
| --- | --- | --- | --- | --- | --- |
| 支持的 Tensor Core 精度 | FP64, TF32, bfloat16, FP16, FP8, INT8, FP6, FP4 | FP64, TF32, bfloat16, FP16, FP8, INT8 | FP64, TF32, bfloat16, FP16, INT8, INT4, INT1 | FP16, INT8, INT4, INT1 | FP16 |
| 支持的 CUDA Core 精度 | FP64, FP32, FP16, bfloat16 | FP64, FP32, bfloat16, FP16, INT8 | FP64, TF32, bfloat16, FP16, INT8 | FP64, fp32, FP16, INT8 | FP64, fp32, FP16, INT8 |
| Tensor Core 版本 | 5.0 | 4.0 | 3.0 | 2.0 | 1.0 |

### NVLink

NVLink 是双向直接 GPU-GPU 互连，第五代 NVLink 连接主机和加速处理器的速度高达每秒 1800GB/s，这是传统 x86 服务器的互连通道——PCIe 5.0 带宽的 14 倍多。英伟达 NVLink-C2C 还将 Grace CPU 和 Hopper GPU 进行连接，加速异构系统可为数万亿和数万亿参数的 AI 模型提供加速性能。

| NVLink  Generation | 1.0 | 2.0 | 3.0 | 4.0  | 5.0 |
| --- | --- | --- | --- | --- | --- |
| NVLink bandwidth per GPU | 300GB/s | 300GB/s | 600GB/s | 900GB/s | 1,800GB/s |
| Maximum Number of Links per GPU | 6 | 6 | 12 | 18 | 18 |
| Architectures | Pascal | Volta | Ampere | Hopper | Blackwell |
| Year | 2014 | 2017 | 2020 | 2022 | 2024 |

### NVSwitch

NVSwitch 是 NVLink 交换机系统的关键使能器，它能够以 NVLink 速度实现 GPU 跨节点的连接。它包含与 400 Gbps 以太网和 InfiniBand 连接兼容的物理（ PHY ）电气接口。随附的管理控制器现在支持附加的八进制小尺寸可插拔（ OSFP ）模块 。

| NVSwitch Generation | 1.0 | 2.0 | 3.0 | NVLink Switch |
| --- | --- | --- | --- | --- |
| Number of GPUs with direct connection within a NVLink domain | Up to 8 | Up to 8 | Up to 8 | Up to 576 |
| GPU-to-GPU bandwidth | 300GB/s | 600GB/s | 900GB/s | 1,800GB/s |
| Total aggregate bandwidth | 2.4TB/s | 4.8TB/s | 7.2TB/s | 1PB/s |
| Architectures | Volta | Ampere  | Hopper | Blackwell |
| Year | 2017 | 2020 | 2022 | 2024 |

## 小结与思考

- 英伟达 GPU 架构发展：英伟达 GPU 架构自 2010 年以来经历了从 Fermi 到 Blackwell 的多代演进，引入了 CUDA Core、Tensor Core、NVLink 和 NVSwitch 等关键技术，显著提升了 GPU 的计算能力和能效。

- Tensor Core 的持续创新：Tensor Core 作为专为深度学习矩阵运算设计的加速器，从第一代发展到第五代，不断增加支持的精度类型和提升性能，以适应 AI 的快速发展。

- NVLink 和 NVSwitch 的技术演进：NVLink 和 NVSwitch 作为 GPU 间和 GPU 与 CPU 间的高速互连技术，其带宽和连接能力随架构代数增加而显著提升，为大规模并行计算和异构计算提供了强大支持。

## 本节视频

<html>
<iframe src="https://player.bilibili.com/player.html?aid=783019461&bvid=BV1x24y1F7kY&cid=1113803137&page=1&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>

<html>
<iframe src="https://player.bilibili.com/player.html?aid=698236135&bvid=BV1mm4y1C7fg&cid=1115170922&page=1&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>
