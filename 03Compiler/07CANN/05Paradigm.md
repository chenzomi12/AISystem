<!--适用于[License] (https://github.com/chenzomi12/AISystem/blob/main/LICENSE)版权许可-->

# Ascend C 编程范式

AI 的发展日新月异，AI 系统相关软件的更新迭代也是应接不暇，作为一本讲授理论的作品，我们将尽可能地讨论编程范式背后的原理和思考，而少体现代码实现，以期让读者理解 Ascend C 为何这样设计，进而随时轻松理解最新的 Ascend C 算子的编写思路。

本节将针对 Ascend C 的编程范式进行详细讲解，重点讲授向量计算编程范式。

## 向量编程范式

基于 Ascend C 编程范式的方式实现自定义向量算子的流程如下图所示，由三个步骤组成：算子分析是进行编程的前置任务，负责明确自定义算子的各项需求，如输入输出、使用 API 接口等；核函数的定义和封装是编程的第一步，负责声明核函数的名称，并提供进入核函数运算逻辑的接口；基于算子需求实现算子类是整个核函数的核心计算逻辑，其由被分为内存初始化、数据搬入、算子计算逻辑实现、数据搬出四个部分，后三者被又被称为算子的实现流程。

![自定义算子开发流程](images/paradigm1.png)

自定义向量算子核心部分一般由两个函数组成，分别是 `Init()` 函数（初始化函数）与 `Process()` 函数（执行函数）。`Init()` 函数完成板外数据定位以及板上内存初始化工作；`Process()` 函数完成向量算子的实现，分成三个流水任务：CopyIn、Compute、CopyOut。CopyIn 负责板外数据搬入，Compute 负责向量计算，CopyOut 负责板上数据搬出。

流水线任务之间存在数据依赖，需要进行数据传递。Ascend C 中使用 `TQue` 队列完成任务之间的数据通信和同步，提供 EnQue、`DeQue` 等基础 API；`TQue` 队列管理不同层级的物理内存时，用一种抽象的逻辑位置（TPosition）来表达各级别的存储，代替了片上物理存储的概念，开发者无需感知硬件架构。另外，Ascend C 使用 `GlobalTensor` 和 `LocalTensor` 作为数据的基本操作单元，它是各种指令 API 直接调用的对象，也是数据的载体。在向量编程模型中，使用到的 `TQue` 类型如下：搬入数据的存放位置 VECIN、搬出数据的存放位置 VECOUT。

在本节中，我们将从 `add_custom` 这一基本的向量算子着手，根据自定义算子的开发流程，逐步介绍如何根据向量编程范式逐步编写自定义向量算子，最后会介绍 Ascend C 向量编程如何进行数据切分。

### 算子分析

在开发算子代码之前需要分析算子的数学表达式、输入、输出以及计算逻辑的实现，明确需要调用的 Ascend C 接口。

1. 明确算子的数学表达式

Ascend C 提供的向量计算接口的操作元素都为 `LocalTensor`，输入数据需要先搬运进片上存储，以 Add 算子为例，数学表达式为：***z***=***x***+***y***，使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

2. 明确输入和输出

Add 算子有两个输入：x 与 y，输出为 z。

本样例中算子的输入支持的数据类型为 `half(float16)`，算子输出的数据类型与输入数据类型相同。

算子输入支持 shape（8，2048），输出 shape 与输入 shape 相同。算子输入支持的数据格式（shape）为：ND。

3. 确定算子实现所需接口

使用 `DataCopy` 来实现数据搬移；由于向量计算实现较为简单，使用基础 API 完成计算逻辑的实现，在加法算子中使用双目指令接口 Add 实现 x+y；使用 EnQue、`DeQue` 等接口对 `TQue` 队列进行管理。

### 核函数定义与封装

在完成算子分析后，可以正式开始开发算子代码，其第一步应该完成对于核函数的定义和封装。在本小节将介绍如何对函数原型进行定义，并介绍核函数定义中应该遵循的规则；随后将介绍函数原型中所需实现的内容；最后本小节将完成核函数的封装，便于后续对于核函数的调用。

1. 函数原型定义

本样例中，函数原型名为 `add_custom`，根据算子分析中对算子输入输出的分析，确定有 3 个参数 x，y，z，其中 x，y 为输入内存，z 为输出内存。

根据核函数定义的规则，使用`__global__`函数类型限定符来标识它是一个核函数，可以被`<<<...>>>`调用；使用`__aicore__`函数类型限定符来标识该核函数在设备端 AI Core 上执行；为方便起见，统一使用 `GM_ADDR` 宏修饰入参，表示其为入参在内存中的位置。`add_custom` 函数原型的定义见下方程序第 1 行所示。

2. 调用算子类的 Init 和 Process 函数

在函数原型中，首先实例化对应的算子类，并调用该算子类的 `Init()` 和 `Process()` 函数，如下方程序第 2-4 行所示。其中，`Init()` 函数负责内存初始化相关工作，`Process()` 函数则负责算子实现的核心逻辑。

3. 对核函数的调用进行封装

对核函数的调用进行封装，得到 `add_custom_do` 函数，便于主程序调用。下方程序第 6 行所示内容表示该封装函数仅在编译运行 NPU 侧的算子时会用到，编译运行 CPU 侧的算子时，可以直接调用 `add_custom` 函数。

调用核函数时，除了需要传入参数 x，y，z，还需要使用`<<<…>>>`传入 `blockDim`（核函数执行的核数）, `l2ctrl`（保留参数，设置为 nullptr）, `stream`（应用程序中维护异步操作执行顺序的任务流对象）来规定核函数的执行配置，如下方程序第 10 行所示。 

```
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z){
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}

#ifndef __CCE_KT_TEST__

// call of kernel function
void add_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* z){
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z);
}

#endif
```

### 算子数据通路

前文已经提到过在 `Process()` 函数中存在三个流水任务，分别是 CopyIn、Compute 和 CopyOut。本节将详细讲解数据在这三个任务之间的传递过程，并为后续使用 Ascend C 对其进行实现作铺垫。

向量算子三阶段任务流水的数据通路如下图所示。

![算子数据通路](images/paradigm2.png)

上图纵向分为 2 部分，上部分为发生在外部存储（Global Memory）中的数据流通过程，下部分为发生在 AI Core 内（Local Memory）中的数据流通过程；横向分为 3 部分，指代 CopyIn、Compute 和 CopyOut 这三个阶段中的数据流通过程。发生在 AI Core 内的任务间数据传递统一由 TPipe 资源管理模块进行管理。

在 CopyIn 任务中，需要先将执行计算的数据 xGm、yGm 从外部存储通过 `DataCopy` 接口传入板上，存储为 xLocal、yLocal，并通过 EnQue 接口传入数据搬入队列 inQueueX、inQueueY 中，以便进行流水模块间的数据通信与同步。

在 Compute 任务中，需要先将 xLocal、yLocal 使用 `DeQue` 接口从数据搬入队列中取出，并使用相应的向量运算 API 执行计算操作得到结果 zLocal，并将 zLocal 通过 EnQue 接口传入数据搬出队列 outQueueZ 中。

在 CopyOut 任务中，需要先将结果数据 zLocal 使用 `DeQue` 接口从数据搬出队列中取出，并使用 `DataCopy` 接口将板上数据传出到外部存储 zGm 中。

上述为向量算子核心处理部分的数据通路，同时也作为一个程序设计思路，下面将介绍如何用 Ascend C 对其进行实现。

### 算子类实现

在对核函数的声明和定义中，我们会提到需要实例化算子类，并调用其中的两个函数来实现算子。在本节中，将首先展示算子类的成员，随后具体介绍 `Init()` 函数和 `Process()` 函数的作用与实现。

1. 算子类成员定义

算子类的成员如下方程序所示。如第 4-5 行所示，在算子类中，需要声明对外开放的内存初始化函数 `Init()` 和核心处理函数 `Process()`。而为了实现适量算子核内计算流水操作，在向量算子中我们又将 `Process()`函数分为三个部分，即数据搬入阶段 `CopyIn()`、计算阶段 `Compute()`与数据搬出阶段 CopyOut()三个私有类成员，见第 6～9 行。

除了这些函数成员声明外，第 10-14 行还依次声明了内存管理对象 pipe、输入数据 `TQue` 队列管理对象 inQueueX 和 inQueueY、输出数据 `TQue` 队列管理对象 outQueueZ 以及管理输入输出 Global Memory 内存地址的对象 xGm，yGm 与 zGm，这些均作为私有成员在算子实现中被使用。

```
class KernelAdd {

public:
    __aicore__ inline KernelAdd() {} 
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z){} 
    __aicore__ inline void Process(){}

private:
    __aicore__ inline void CopyIn(int32_t progress){}
    __aicore__ inline void Compute(int32_t progress){}
    __aicore__ inline void CopyOut(int32_t progress){}

private:
    TPipe pipe; 
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;  
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;  
    GlobalTensor<half> xGm, yGm, zGm;  
};
```

2. 初始化函数 Init()函数实现

在多核并行计算中，每个核计算的数据是全部数据的一部分。Ascend C 核函数是单个核的处理函数，所以我们需要获取每个核负责的对应位置的数据。此外，我们还需要对于声明的输入输出 `TQue` 队列分配相应的内存空间。

`Init()` 函数实现见下方程序。第 2～5 行通过计算得到该核所负责的数据所在位置，其中 x、y、z 表示 3 个入参在片外的起始地址；BLOCK_LENGTH 表示单个核负责的数据长度，为数据全长与参与计算核数的商；`GetBlockIdx()`是与硬件感知相关的 API 接口，可以得到核所对应的编号，在该样例中为 0-7。通过这种方式可以得到该核函数需要处理的输入输出在 Global Memory 上的内存偏移地址，并将该偏移地址设置在 Global Tensor 中。

第 6～8 行通过 TPipe 内存管理对象为输入输出 `TQue` 分配内存。其调用 API 接口 `InitBuffer()`，接口入参依次为 `TQue` 队列名、是否启动 double buffer 机制以及单个数据块的大小（而非长度）。

```
1   __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
2   {
3       xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
4       yGm.SetGlobalBuffer((__gm__ half*)y + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
5       zGm.SetGlobalBuffer((__gm__ half*)z + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
6       pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
7       pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
8       pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
9   }

```

3. 核心处理函数 Process()函数实现

基于向量编程范式，将核函数的实现分为 3 个基本任务：CopyIn，Compute，CopyOut，`Process()` 函数通过调用顺序调用这三个基本任务完成核心计算任务。然而考虑到每个核内的数据仍然被进一步切分成小块，需要循环执行上述步骤，从而得到最终结果。`Process()` 函数的实现如下方程序所示。 

```
1   public:
2       __aicore__ inline void Process()
3       {
4           constexpr int32_t loopCount = TILE_NUM * BUFFER_NUM;
5           for (int32_t i = 0; i < loopCount; i++) {
6               CopyIn(i);
7               Compute(i);
8               CopyOut(i);
9           }
10      }
11  private:
12      __aicore__ inline void CopyIn(int32_t progress)
13      {
14          LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
15          LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();

16          DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
17          DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);

18          inQueueX.EnQue(xLocal);
19          inQueueY.EnQue(yLocal);
20      }
21      __aicore__ inline void Compute(int32_t progress)
22      {
23          LocalTensor<half> xLocal = inQueueX.DeQue<half>();
24          LocalTensor<half> yLocal = inQueueY.DeQue<half>();
25          LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

26          Add(zLocal, xLocal, yLocal, TILE_LENGTH);
27          outQueueZ.EnQue<half>(zLocal);

28          inQueueX.FreeTensor(xLocal);
29          inQueueY.FreeTensor(yLocal);
30      }
31      __aicore__ inline void CopyOut(int32_t progress)
32      {
33          LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
34          DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
35          outQueueZ.FreeTensor(zLocal);
36      }

```
如上方程序第 4-9 行所示，`Process()` 函数需要首先计算每个核内的分块数量，从而确定循环执行三段流水任务的次数，随后依此循环顺序执行数据搬入任务 `CopyIn()`、向量计算任务 `Compute()` 和数据搬出任务 `CopyOut()`。一个简化的数据通路图如下图所示。根据此图，可以完成各个任务的程序设计。

![三段式流水任务数据通路](images/paradigm3.png)

- CopyIn()私有类函数实现

使用 `AllocTensor` 接口为参与计算的输入分配板上存储空间，如上方程序第 14～15 行代码所示，由于定义的入参数据类型是 half 类型的，所以此处分配的空间大小也为 half。

使用 `DataCopy` 接口将 `GlobalTensor` 数据拷贝到 `LocalTensor`，如第 16～17 行所示，xGm、yGm 存储的是该核所需处理的所有输入，因此根据该分块对应编号找到相关的分块数据拷贝至板上。

使用 EnQue 将 `LocalTensor` 放入 VecIn 的 `TQue` 中，如第 18～19 行所示。

- Compute()私有类函数实现

使用 `DeQue` 从 VecIn 中取出输入 x 和 y，如上方程序第 23-24 行所示。

使用 `AllocTensor` 接口为输出分配板上存储空间，如第 25 行所示。

使用 Ascend C 接口 Add 完成向量计算，如第 26 行所示。该接口是一个双目指令 2 级接口，入参分别为目的操作数、源操作数 1、源操作数 2 和输入元素个数。

使用 EnQue 将计算结果 `LocalTensor` 放入到 VecOut 的 `TQue` 中，如第 27 行所示。

使用 `FreeTensor` 释放不再使用的 `LocalTensor`，即两个用于存储输入的 `LocalTensor`，如第 28～29 行所示。

- CopyOut 私有类函数实现

使用 `DeQue` 接口从 VecOut 的 `TQue` 中取出目标结果 z，如上方程序第 33 行所示。

使用 `DataCopy` 接口将 `LocalTensor` 数据拷贝到 `GlobalTensor` 上，如第 34 行所示。

使用 `FreeTensor` 将不再使用的 `LocalTensor` 进行回收，如第 35 行所示。

### 算子切分策略

正如前文所述，Ascend C 算子编程是 SPMD 编程，其使用多个核进行并行计算，在单个核内还将数据根据需求切分成若干份，降低每次计算负荷，从而起到加快计算效率的作用。这里需要注意，Ascend C 中涉及到的核数其实并不是指实际执行的硬件中所拥有的处理器核数，而是“逻辑核”的数量，即同时运行了多少个算子的实例，是同时执行此算子的进程数量。一般的，建议使用的逻辑核数量是实际处理器核数的整数倍。此外，如果条件允许，还可以进一步将每个待处理数据一分为二，开启 double buffer 机制（一种性能优化方法），实现流水线间并行，进一步减少计算单元的闲置问题。

在本 add_custom 算子样例中，设置数据整体长度 TOTAL_LENGTH 为 8* 2048，平均分配到 8 个核上运行，单核上处理的数据大小 BLOCK_LENGTH 为 2048；对于单核上的处理数据，进行数据切块，将数据切分成 8 块（并不意味着 8 块就是性能最优）；切分后的每个数据块再次切分成 2 块，即可开启 double buffer。此时每个数据块的长度 TILE_LENGTH 为 128 个数据。

具体数据切分示意图下图所示，在确定一个数据的起始内存位置后，将数据整体平均分配到各个核中，随后针对单核上的数据再次进行切分，将数据切分为 8 块，并启动 double buffer 机制再次将每个数据块一分为二，得到单个数据块的长度 TILE_LENGTH。

![算子数据切分策略](images/paradigm4.png)

数据切分中所使用的各参数定义如下程序所示：第 1 行定义了数据全长 TOTAL_LENGTH，约束了输入数据的长度；第 2 行声明了参与计算任务的核数 USE_CORE_NUM；第 3 行计算得到了单个核负责计算的数据长度 BLOCK_LENGTH；第 4 行定义了单个核中数据的切分块数 TILE_NUM；第 5 行决定了是否开启 double buffer 机制，如果不开启则规定 BUFFER_NUM = 1；第六行计算得到单个数据块的数据长度 TILE_LENGTH。

```
1   constexpr int32_t TOTAL_LENGTH = 8 * 2048;  
2   constexpr int32_t USE_CORE_NUM = 8;
3   constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
4   constexpr int32_t TILE_NUM = 8; 
5   constexpr int32_t BUFFER_NUM = 2;
6   constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;
```

## 小结与思考

- 合理的编程范式能够帮助开发者省去许多设计代码结构的思考负担，以及在一定程度上隐藏并行计算等相关性能优化细节，对提升算子开发效率有很大帮助。

- 编程范式除了向量计算范式以外，还有矩阵计算范式，以及向量-矩阵混合（通常是融合算子）编程范式。
