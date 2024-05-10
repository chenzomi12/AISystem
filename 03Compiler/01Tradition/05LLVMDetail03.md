<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# LLVM 后端代码生成

上一节主要讲了 LLVM 的前端和优化层，前端主要对高级语言做一些词法的分析，把高级语言的特性转变为 token，再交给语法分析对代码的物理布局进行判别，之后交给语义分析对代码的的逻辑进行检查。优化层则是对代码进行优化，比如常量折叠、死代码消除、循环展开、内存分配优化等。

本节将介绍 LLVM 后端的生成代码过程，LLVM 后端的作用主要是将优化后的代码生成目标代码，目标代码可以是汇编语言、机器码。

## 代码生成 

LLVM 的后端是与特定硬件平台紧密相关的部分，它负责将经过优化的 LLVM IR 转换成目标代码，这个过程也被称为代码生成（Codegen）。不同硬件平台的后端实现了针对该平台的专门化指令集，例如 ARM 后端实现了针对 ARM 架构的汇编指令集，X86 后端实现了针对 X86 架构的汇编指令集，PowerPC 后端实现了针对 PowerPC 架构的汇编指令集。

在代码生成过程中，LLVM 后端会根据目标硬件平台的特性和要求，将 LLVM IR 转换为适合该平台的机器码或汇编语言。这个过程涉及到指令选择（Instruction Selection）、寄存器分配（Register Allocation）、指令调度（Instruction Scheduling）等关键步骤，以确保生成的目标代码在目标平台上能够高效运行。

LLVM 的代码生成能力使得开发者可以通过统一的编译器前端（如 Clang）生成针对不同硬件平台的优化代码，从而更容易实现跨平台开发和优化。同时，LLVM 后端的可扩展性也使得它能够应对新的硬件架构和指令集的发展，为编译器技术和工具链的进步提供了强大支持。

## LLVM 后端 Pass

整个后端流水线涉及到四种不同层次的指令表示，包括：

- 内存中的 LLVM IR：LLVM 中间表现形式，提供了高级抽象的表示，用于描述程序的指令和数据流。

- SelectionDAG 节点：在编译优化阶段生成的一种抽象的数据结构，用以表示程序的计算过程，帮助优化器进行高效的指令选择和调度。

- Machinelnstr：机器相关的指令格式，用于描述特定目标架构下的指令集和操作码。

- MCInst：机器指令，是具体的目标代码表示，包含了特定架构下的二进制编码指令。

在将 LLVM IR 转化为目标代码需要非常多的步骤，其 Pipeline 如下图所示：

![Pipeline](images/llvm_ir17.png)

LLVM IR 会变成和后端非常很接近的一些指令、函数、全局变量和寄存器的具体表示，流水线越向下就越接近实际硬件的目标指令。其中白色的 pass 是非必要 pass，灰色的 pass 是必要 pass，叫做 Super Path

### 指令选择

========= 介绍下Instruction Selection

- 内存中 LLVM IR 变换为目标特定 SelectionDAG 节点；
- 每个 DAG 能够表示单一基本块的计算；
- DAG 图中的节点表示具体执行的指令，而边编码了指令间的数据流依赖关系；
- 目标是让 LLVM 代码生成程序库能够运用基于树的模式匹配指令选择算法。

![Pipeline](images/llvm_ir22.png)

以上是一个 SelectionDAG 节点的例子

- 红色线：红色连接线主要用于强制相邻的节点在执行时紧挨着，表示这些节点之间必须没有其他指令。
- 蓝色虚线：蓝色虚线连接代表非数据流链，用以强制两条指令的顺序，否则它们就是不相关的。

### 指令调度

========= 深入介绍下 Instruction Scheduling 

- 第 1 次指令调度(Instruction Scheduling)，称为前寄存器分配(RA)调度；
- 对指令(节点)进行排序，同时尝试发现尽可能多的指令层次的并行；
- 指令将被转换为三地址表示的 MachineInstr。

### 寄存器分配

========= 深入介绍下 Register Allocation

- LLVMIR 两个重要的特性之一：LLVM IR 寄存器集是无限；这个性质一直保持着，直到寄存器分配(Register Allocation);

- 寄存器分配的基本任务是将无限数量的虚拟寄存器转换为有限的物理寄存器；

- 编译器会使用挤出（spill）策略将某些寄存器的内容存储到内存中。

- 寄存器分配算法有很多，比如 Greedy Register Allocation,Iterated Register Coalescing,Graph Coloring，基于图的寄存器分配算法。

### 指令调度

========= 深入介绍下 Instruction Scheduling

- 第 2 次指令调度，也称为后寄存器分配(RA)调度；

- 此时可获得真实的寄存器信息，某些类型寄存器存在延迟，它们可被用以改进指令顺序。

若上一步分析中寄存器不足，或者存在计算延迟的风险时可以通过指令的调度改变指令的顺序

### 代码输出

========= 深入介绍下 Code Emission 

- 代码输出阶段将指令从 MachineInstr 表示变换为 MCInst 实例；

- 新的表示更适合汇编器和链接器，可以输出汇编代码或者输出二进制块特定目标代码格式。

## LLVM 编译器全流程

======= LLVM 的全流程，挪动到LLVM架构里面？

编译器工作流程为在高级语言 C/C++ 编译过程中，源代码经历了多个重要阶段，从词法分析到生成目标代码。整个过程涉及前端和后端的多个步骤，并通过中间表示（IR）在不同阶段对代码进行转换、优化和分析。

======= 下面两个图代表不同的意思，深入解释下

![Pipeline](images/llvm_ir18.png)
![Pipeline](images/llvm_ir19.png)

1. 前端阶段

词法分析（Lexical Analysis）：源代码被分解为词法单元，如标识符、关键字和常量。
语法分析（Syntax Analysis）：词法单元被组织成语法结构，构建抽象语法树（AST）。
语义分析（Semantic Analysis）：AST 被分析以确保语义的正确性和一致性。

2. 中间表示（IR）阶段

将 AST 转化为中间表示（IR），采用 SSA 形式的三地址指令表示代码结构。
通过多段 pass 进行代码优化，包括常量传播、死代码消除、循环优化等，以提高代码性能和效率。
IR 进一步转化为 DAG 图，其中每个节点代表一个指令，边表示数据流动。

3. 后端阶段

指令选择（Instruction Selection）：根据目标平台特性选择合适的指令。
寄存器分配（Register Allocation）：分配寄存器以最大程度减少内存访问。
指令调度（Instruction Scheduling）：优化指令执行顺序以减少延迟。
最终生成目标代码，用于目标平台的执行。

Pass 管理:

在编译器的每个模块和 pass 均可通过 pass manager 进行管理，可以动态添加、删除或调整 pass 来优化编译过程中的各个阶段。

## 基于 LLVM 项目

1. Modular

Youtube 上 LLVM 之父 Chris Lattner：编译器的黄金时代[<sup>1</sup>](#ref1)

![Pipeline](images/llvm_ir20.png)

之后 Chris Lattner 创建了 Modular[<sup>2</sup>](#ref2)，标是重建全球 ML 基础设施，包括编译器、运行时，异构计算、边缘到数据中心并重并专注于可用性，提升开发人员的效率。

======= 上面的语句整体不太通顺。

2. XLA：优化机器学习编译器[<sup>3</sup>](#ref3)

XLA(加速线性代数)是 Google 推出的一种针对特定领域的线性代数编译器，能够加快 TensorFlow 模型的运行速度，而且可能完全不需要更改源代码。

![Pipeline](images/llvm_ir23.png)

TensorFlow 中大部分代码和算子都是通过 XLA 编译的，XLA 的底层就是 LLVM，所以 XLA 可以利用到 LLVM 的很多特性，比如优化、代码生成、并行计算等。

3. JAX:高性能的数值计算库[<sup>4</sup>](#ref4)

JAX 是 Autograd 和 XLA 的结合，JAX 本身不是一个深度学习的框架，他是一个高性能的数值计算库，更是结合了可组合的函数转换库，用于高性能机器学习研究。

![Pipeline](images/llvm_ir24.png)

4. TensorFlow:机器学习平台[<sup>5</sup>](#ref5)

TensorFlow 是一个端到端开源机器学习平台。它拥有一个全面而灵活的生态系统，其中包含各种工具、库和社区资源，可助力研究人员推动先进机器学习技术。

TensorFlow 可以更好的应用于工业生产环境，因为它可以利用到硬件加速器，并提供可靠的性能。

![Pipeline](images/llvm_ir25.png)

5. TVM 到端深度学习编译器[<sup>6</sup>](#ref6)

为了使得各种硬件后端的计算图层级和算子层级优化成为可能，TVM 从现有框架中取得 DL 程序的高层级表示，并产生多硬件平台后端上低层级的优化代码，其目标是展示与人工调优的竞争力。

![Pipeline](images/llvm_ir26.png)

6. Julia:面向科学计算的高性能动态编程语言[<sup>7</sup>](#ref7)

在其计算中，Julia 使用 LLVM JIT 编译。LLVM JIT 编译器通常不断地分析正在执行的代码，并且识别代码的一部分，使得从编译中获得的性能加速超过编译该代码的性能开销。

![Pipeline](images/llvm_ir27.png)

## 总结

本节介绍了 LLVM 后端的生成代码过程，LLVM 后端的作用主要是将优化后的代码生成目标代码，目标代码可以是汇编语言、机器码。LLVM 后端的生成过程包括指令选择、寄存器分配、指令调度、代码输出等步骤，这些步骤都可以被不同的后端实现。

LLVM 后端的可扩展性使得它能够应对新的硬件架构和指令集的发展，为编译器技术和工具链的进步提供了强大支持。同时，LLVM 的前端和优化层也为开发者提供了统一的编译器前端，使得开发者可以更容易实现跨平台开发和优化。基于 LLVM 的项目包括 Chris Lattner 的 Modular、XLA、JAX、TensorFlow、TVM、Julia 等。

========= 把重点的内容描述一下，高度提炼，参考本节我写的第一篇和第二篇的总结。

## 本节视频

<html>
<iframe src="https://player.bilibili.com/player.html?aid=390568348&bvid=BV1cd4y1b7ho&cid=903537014&p=1&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>

## 引用

1. https://www.youtube.com/watch?v=4HgShra-KnY
2. https://www.modular.com
3. https://scottamain.github.io/xla
4. https://jax.readthedocs.io
5. https://www.tensorflow.org
6. https://tvm.apache.org/
7. https://julialang.org/
