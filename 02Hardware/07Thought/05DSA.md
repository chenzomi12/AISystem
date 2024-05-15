# 从CUDA和NVIDIA中借鉴

从技术的角度重新看英伟达生态，有很多值得借鉴的方面。本节将主要从流水编排、SIMT前端、分支预测和交互方式等方面进行分析，同时对比DSA架构，思考可以从英伟达CUDA中借鉴的要点。

## 英伟达生态的思考点

从软件和硬件架构的角度出发，CUDA和SIMT之间存在一定的关系，而目前AI芯片采用的DSA架构在编程模型和硬件执行模型上还处于较为早期的状态，英伟达强大的生态同样离不开CUDA在编程方面的易用性。

### SIMT 与 CUDA 的关系

英伟达为了维护CUDA生态对SIMT硬件架构做出了调整和取舍，因此CUDA会在一定程度上对NVIDIA硬件架构产生约束，例如保留SM、Warp、Thread等线程分层概念。CUDA架构在近几年没有做出重大的改变，主要是维护编程体系软件对外的抽象和易用性。

DSA之所以在硬件架构的指令和设计上比较激进，并非软件体系做得好，而是在刚开始并没有太多地考虑编程体系的问题，自然没有为了实现软硬件协同带来的架构约束。CUDA的成功之处在于通过SIMT架构掩盖了流水编排、并行指令隐藏以及CUDA的易用性。

### DSA硬件架构执行方式

DSA硬件架构一般是指单核单线程，线程内指令可以通过多核共享Cache协作。编程模型上缺乏统一的标准，因此需要专门搭建编译器和编程体系，硬件主要以 AI 加速芯片（TPU、NPU 等）为主。

关于DSA的硬件执行方式，DSA硬件目前的裸接口一般是每个核一个线程，每个线程内串行调DSA指令集，指令在硬件上通常会分发到不同的指令执行流水线上，正确性部分靠软件同步实现，部分靠硬件保证。

### CUDA客户能力区分

按照使用CUDA的难易程度，可以将CUDA的使用用户分为三类，分别是初阶、中阶和高阶用户。

- 初阶用户：掌握 CUDA 并行编程能力，了解NV SIMT硬件基础架构，可以拿到并行指令、流水掩盖、并行计算三部分性能。

- 中阶用户：进一步运用 CUDA 提供的切块 Tiling、流水 Pipeline 能力，进一步获取更高的性能收益。

- 高阶用户：深入了解 SIMT 微架构细节，解决线程bank冲突、精细化流水掩盖、精细化指令使用、极致的切块 Tiling 策略，从而实现极致性能。

CUDA在开发方面具有很好的易用性，以下是使用CPU编写的矩阵加法运算：

```c
void add_matrix(float* a, float* b, float* c, int N) {
    int index;

    for (int i = 0; i < N; ++i) {
        index = i + j * N;
        c[index] = a[index] + b[index];
    }
}

int main()
{
    add_matrix(a, b, c, N);
}
```

以下是使用GPU编写的矩阵加法运算，与CPU编程相比，因为使用的是并行计算，所以没有for循环：

```c
__global__ void add_matrix(float* a, float* b, float* c, int N) {
    // Calculate the global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i + j * N;

    // Check if the indices are within bounds
    if (i < N && j < N) {
        // Perform matrix addition
        c[index] = a[index] + b[index];
    }
}

int main() {
    dim3 dimBlock(blocksize, blocksize);
    
    // Calculate the grid size
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    
    // Launch the kernel
    add_matrix<<<dimGrid, dimBlock>>>(a, b, c, N);
}
```

结合优秀的硬件架构和软件生态，英伟达GPU和CUDA是SIMT最成功的实践。

## 借鉴Ⅰ：流水编排

在指令流水线编排方面，最重要的是从硬件设计上解决SIMD data path流水编排问题。程序执行最大的瓶颈是访存和控制流，单线程CPU需要大量资源进行分支预测、超前执行、缓存、预取等机制来缓解访存和控制流遇到的瓶颈。SIMD往往依赖CPU自身乱序、投机、缓存和预取等能力来缓解。英伟达GPU则是依靠多线程交错执行提升整体并行计算的性能，大量的线程通过不同的block和不同的线程读取数据和执行计算指令。

![指令流水线](images/05DSA01.png)

即使在DSA上为SIMD硬件封装了SIMT前端，如果遇到执行指令有依赖，基础性能也会非常差，流水编排仍然需要开发者动手，想写出开箱即用，性能较优的代码同样很难。

## 借鉴Ⅱ：SIMT前端硬件

增加了SIMT前端硬件，通过线程组Warp隐藏线程指令流水。在CUDA编程模型中，每一个线程块（thread block）内部需要有很多并行线程，隐式分成了若干个Warp，每个Warp包含串行交错的访存和计算。GPU通过Warp Scheduler动态交错执行，如果一组Warp0流水阻塞就会切到下一个Warp1，隐式通过Warp的并行掩盖指令流水阻塞，因此开发者可以得到较好的性能。

![通过Warp并行掩盖流水阻塞](images/05DSA02.png)

DSA硬件架构同样可以引入Warp Scheduler进行指令流水掩盖，让每个 DSA 核执行多个线程，相互掩盖流水线阻塞。NVIDIA GPU使用Warp来掩盖指令流水是基于运行时的具体信息，而开发者和编译器只能基于静态信息进行流水编排，很难做到足够均衡，使得SIMD/DSA在进行手工或者编译器自动流水编排时相对困难，资深开发者也很难把流水编排得足够好。

增加SIMT前端硬件同样会带来开销，但是可以实现流水阻塞掩盖，通过SIMT表达将接口暴露给用户，让用户主动写多线程，warp scheduler在硬件层面实现多线程相互掩盖流水阻塞。SIMD指令掩盖可以通过SIMT表达实现用户写通用单线程，同时warp分组组成SIMD指令。

![SIMT前端实现流水阻塞掩盖](images/05DSA03.png)

但是CUDA没有解决DSA指令掩盖，目前只是通过给开发者一个Warp概念，透传指令API来解决表达和使用的问题，因此CUDA的上手门槛并不低，需要在前期充分了解NVIDIA GPU的硬件细节。

## 借鉴Ⅲ：分支预测机制

SPMD编程模型对分支预测和控制流的高容忍度是支撑易用性的重要手段，减少分支和连续访存是软件层面、易用性方面需要关注的优化点。当然，在SIMD的硬件上同样可以通过Predicate/mask和对gather/scatter指令memory coalescing来实现，通过编译器实现分支预测从而让开发者无感，但是在SIMD线程有限的情况下，性能的提升可能会是个难题。

英伟达GPU可以使线程在Warp-base SIMD上执行不同的分支，每个线程都可以执行带条件控制流指令（Conditional Control Flow Instructions），同时不同线程间可以分别执行不同的控制流路径（Different Control Flow Paths），比如分别执行不同的Thread W，Thread X和Thread Y控制流执行路径。

![分支预测机制](images/05DSA04.png)

但是SIMT的控制流仍然存在很多问题，因此不推荐在CUDA编程中出现大量的if/else语句。通常使用SIMD流水线来节省控制逻辑上的面积，例如将 Scalar 线程放在Warps里面。当Warp内部的线程分支到不同的执行路径时，就会发生分支执行冲突，比如当存在Path 1和Path 2两个分支路径时，可以使得不同时间执行不同的路径，但是这样会增加时耗。

![分支执行冲突](images/05DSA05.png)

为了解决分支预测的问题，动态Warp Formating/Merging在分支后动态合并执行相同指令的线程，从正在等待的Warps中形成新的Warps，分支下每条路径线程用于创建新的Warp。可以将Warp X和Warp Y合并为Warp Z，从而更好地执行相同指令。

![动态Warp合并](images/05DSA06.png)

当存在Path 1和Path 2两条路径的时候，由于某些时钟周期为空，因此在动态合并分支之后执行相同指令的线程，以便同时执行不同的代码路径，从而避免线程之间的等待和资源浪费。

![动态分支合并](images/05DSA07.png)

动态warp分组更多是在编译器层面解决分支预测的问题，根据线程执行情况和数据依赖性动态组织warp中的线程，以提高并行计算性能和资源利用率，优化GPU计算，提高程序的执行效率。

![动态Warp分组](images/05DSA08.png)

## 借鉴Ⅳ：交互方式

CUDA可以提供host（CPU）和device（GPU）之间便利的交互方式，CUDA中有很多实现机制与SIMT、SIMD、DSA的硬件架构本身并没有太多关系。CUDA中的所有特性也不是SIMT架构独有的，因此不存在技术上选择SIMT、SIMD、DSA与硬件强行绑定等问题。比如CUDA Runtime提供host和device的C++交互方式，如寒武纪BANG C语言在这个层面就参考了CUDA。在软件层面的交互上，CUDA可以很容易地实现向量加法：

```c
for (int i = 0; i < 10000; ++i) {
    C[i] = A[i] + B[i];
}
```

```c
#include <stdio.h>

// 定义向量大小
#define N 5

// CUDA 核函数，用于执行向量加法
__global__ void vectorAdd(int *a, int *b, int *c) {
    int i = blockDim.x * threadDim.x + threadIdx.x;
    
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int a[N], b[N], c[N];
    int *d_a, *d_b, *d_c;

    // 在设备上分配内存
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // 初始化向量 a 和 b
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = N - i;
    }

    // 将向量 a 和 b 复制到设备
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // 调用 CUDA 核函数
    vectorAdd<<<1, N>>>(d_a, d_b, d_c);

    // 将结果从设备复制回主机
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < N; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    // 释放设备上的内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

CUDA同时具有编程开发的易用性，对初阶用户而言，CUDA的易用性是极致的，入门开发者任意写一个简单的Kernel，就能够获得比CPU高5~10倍的峰值性能。

因为DSA 硬件架构在流水和指令使用上缺乏完备、隐式的支持，指令流水的支持需要开发者通过手工掩盖、切块等其他优化思想补齐这部分性能，而使用底层指令则会让用户在写出正确的Kernel时花费更多的时间。

## 思考

在流水隐藏方面，实现架构层面的隐藏流水编排机制，提出一个形式上与SPMD没有关系的编程模式，而且易用性堪比CUDA的软件是可能的。但是反过来在核心问题上没有得到解决，提出形式上与CUDA类似的编程模型也仍然会有易用性的问题，开发者很难获得一个足够好的初始性能。

在软硬件架构方面，对于DSA架构而言，一方面需要建立一套开放的软硬件架构，联合其他DSA架构一起对抗 CUDA 生态；另一方面需要明确面向不同层级开发者的易用性和软件开发形态。

## 总结

相比较英伟达GPU的强大生态，AI芯片采用的DSA架构在编程模型和硬件执行模型方面还处于较为早期的状态，各项技术还不太成熟，很难与英伟达形成抗衡，同时CUDA在编程方面的易用性让不同层次需求的开发者实现并行加速。

NVIDIA GPU有以下四点值得借鉴：

- 提供良好的流水编排，提供多线程交错并行来提升整体性能。

- 增加了SIMT前端硬件，通过线程组Warp隐藏线程指令流水，SIMD使用户只需要实现单线程，但是需要在前期充分了解GPU的硬件架构。

- 在分支预测方面，通过动态Warp合并，动态分支合并和动态Warp分组，使得GPU并行计算能力大大提高。

- CUDA很好地提供了主机（CPU）和设备（GPU）之间的交互方式，使编程开发更加具有易用性。

## 本节视频

<html>
<iframe src="//player.bilibili.com/player.html?aid=367168309&bvid=BV1j94y1N7qh&cid=1365978059&p=1&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>