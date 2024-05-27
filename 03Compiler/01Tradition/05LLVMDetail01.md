<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# LLVM IR 详解

在上一节中，我们已经简要介绍了 LLVM 的基本概念和架构，我们现在将更深入地研究 LLVM 的 IR（中间表示）的概念。

了解 LLVM IR 的重要性是为了能够更好地理解编译器的运作原理，以及在编译过程中 IR 是如何被使用的。LLVM IR 提供了一种抽象程度适中的表示形式，同时能够涵盖绝大多数源代码所包含的信息，这使得编译器能够更为灵活地操作和优化代码。

本节将进一步探究 LLVM IR 的不同表示形式，将有助于我们更好地理解代码在编译器中是如何被处理和转换的。

## LLVM IR 概述

在开发编译器时，通常的做法是将源代码编译到某种中间表示（Intermediate Representation，一般称为 IR），然后再将 IR 翻译为目标体系结构的汇编（比如 MIPS 或 X86），这种做法相对于直接将源代码翻译为目标体系结构的好处主要有两个：

1. 首先，有一些优化技术是目标平台无关的，我们只需要在 IR 上做这些优化，再翻译到不同的汇编，这样就能够在所有支持的体系结构上实现这种优化，这大大的减少了开发的工作量。

2. 其次，假设我们有 m 种源语言和 n 种目标平台，如果我们直接将源代码翻译为目标平台的代码，那么我们就需要编写 m * n 个不同的编译器。然而，如果我们采用一种 IR 作为中转，先将源语言编译到这种 IR ，再将这种 IR 翻译到不同的目标平台上，那么我们就只需要实现 m + n 个编译器。

因此，目前常见的编译器都分为了三个部分，前端（front-end），优化层（middle-end）以及后端（back-end），每一部分都承担了不同的功能：

- 前端：将源高级语言编译到中间表达 IR
- 优化层：对中间表达 IR 进行编译优化
- 后端：将中间表达 IR 翻译为目标机器的语言

同理，LLVM 也是按照这一结构设计进行架构设计:

![编译器](images/llvm_ir07.png)

在 LLVM 中不管是前端、优化层、还是后端都有大量的 IR，使得 LLVM 的模块化程度非常高，可以大量的复用一些相同的代码，非常方便的集成到不同的 IDE 和编译器当中。

## LLVM IR 数据结构

LLVM 并非使用单一的 IR 进行表达，前端传给优化层时传递的是一种抽象语法树（Abstract Syntax Tree，AST）的 IR。因此 IR 是一种抽象表达，没有固定的形态。

![编译器](images/llvm_ir01.png)

抽象语法树的作用在于牢牢抓住程序的脉络，从而方便编译过程的后续环节（如代码生成）对程序进行解读。AST 就是开发者为语言量身定制的一套模型，基本上语言中的每种结构都与一种 AST 对象相对应。

在中端优化完成之后会传一个 DAG 图的 IR 给后端，DAG 图能够非常有效的去表示硬件的指定的顺序。

> DAG（Directed Acyclic Graph，有向无环图）是图论中的一种数据结构，它是由顶点和有向边组成的图，其中顶点之间的边是有方向的，并且图中不存在任何环路（即不存在从某个顶点出发经过若干条边之后又回到该顶点的路径）。
>
> 在计算机科学中，DAG 图常常用于描述任务之间的依赖关系，例如在编译器和数据流分析中。DAG 图具有拓扑排序的特性，可以方便地对图中的节点进行排序，以确保按照依赖关系正确地执行任务。

编译的不同阶段会产生不同的数据结构和中间表达，如前端的抽象语法树（AST）、优化层的 DAG 图、后端的机器码等。后端优化时 DAG 图可能又转为普通的 IR 进行优化，最后再生产机器码。

## LLVM IR 表示形式

LLVM IR 具有三种表示形式，这三种中间格式是完全等价的：

- 在内存中的编译中间语言（无法通过文件的形式得到的指令类等）
- 在硬盘上存储的二进制中间语言（格式为.bc）
- 人类可读的代码语言（格式为.ll）

接下来我们就看一下具体的 .ll 文件格式。

### LLVM IR 示例程序

我们编写一个简单的 C 语言程序，并将其编译为 LLVM IR。

test.c 文件内容如下：

```c
#include <stdio.h>

void test(int a, int b)
{
    int c = a + b;
}

int main(void)
{
    int a = 10;
    int b = 20;
    test(a, b);
    return 0;
}
```

======= 介绍下面 XXXX

```shell
clang -S -emit-llvm .\test.c
```

编译完成后，生成的 test.ll 文件内容如下：

```shell
; ModuleID = '.\test.c'
source_filename = ".\\test.c"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @test(i32 noundef %0, i32 noundef %1) #0 { ;定义全局函数@test(a,b)
  %3 = alloca i32, align 4 ; 局部变量 c
  %4 = alloca i32, align 4 ; 局部变量 d
  %5 = alloca i32, align 4 ; 局部变量 e
  store i32 %0, ptr %3, align 4 ; %0 赋值给%3 c=a
  store i32 %1, ptr %4, align 4 ; %1 赋值给%4 d=b
  %6 = load i32, ptr %3, align 4 ; 读取%3，值给%6 就是参数 a
  %7 = load i32, ptr %4, align 4 ; 读取%4，值给%7 就是参数 b
  %8 = add nsw i32 %6, %7
  store i32 %8, ptr %5, align 4 ; 参数 %9 赋值给%5 e 就是转换前函数写的 int c 变量
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  store i32 10, ptr %2, align 4
  store i32 20, ptr %3, align 4
  %4 = load i32, ptr %2, align 4
  %5 = load i32, ptr %3, align 4
  call void @test(i32 noundef %4, i32 noundef %5)
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 1, !"MaxTLSAlign", i32 65536}
!4 = !{!"(built by Brecht Sanders, r4) clang version 17.0.6"}
```

======== 先介绍基本语法，然后介绍示例程序，然后对示例程序进行解析。上面的示例代码描述的是？

### LLVM IR 基本语法

========== 深入介绍语法规则

1. 注释以 ; 开头  
2. 全局表示以 @ 开头，局部变量以 % 开头  
3. alloca 在函数栈帧中分配内存  
4. i32 32 位 4 个字节的意思  
5. align 字节对齐  
6. store 写入  
7. load 读取

### LLVM IR 指令集

========== 深入指令集介绍

- LLVM IR 是类似于精简指令集(RISC)的底层虚拟指令集;
- 和真实精简指令集一样，支持简单指令的线性序列，例如添加、相减、比较和分支；
- 指令都是三地址形式，它们接受一定数量的输入然后在不同的寄存器中存储计算结果；
- 与大多数精简指令集不同，LVM 使用强类型的简单类型系统，并剥离了机器差异；
- LLVM IR 不使用固定的命名寄存器，它使用以%字符命名的临时寄存器；

## LLVM IR 指导原则

======= 下面的 1,2 没有前后文，看不懂主语。

1. 采用静态单赋值形式（Static single assignment,SSA）表示，代码组织为三地址指令序列和无限寄存器让优化能够快速执行。

2. 整个程序的 IR 存储到磁盘让链接时优化易于实现。

### 静态单赋值（SSA）

静态单赋值是指当程序中的每个变量都有且只有一个赋值语句时，称一个程序是 SSA 形式的。LLVM IR 中，每个变量都在使用前都必须先定义，且每个变量只能被赋值一次。以`1*2+3`为例：

```shell
%0 = mul i32 1, 2
%0 = add i32 %0, 3
ret i32 %0
```

静态单赋值形式是指每个变量只有一个赋值语句，所以上述代码的`%0`不能复用

```shell
%0 = mul i32 1, 2
%1 = add i32 %0, 3
ret i32 %1
```

静态单赋值好处：  

- 每个值只有单一赋值定义了它。每次使用一个值，可以立刻向后追溯到给出其定义的唯一的指令。非常方便编译器正向和反向的编译；
- 极大简化优化，因为 SSA 形式建立了平凡的 use-de 链，也就是一个值到达使用之处的定义的列表。

======= 定义了它？太过于口语，不知道它代指什么。

### 三地址码 

每个三地址码指令，都可以被分解为一个四元组(4-tuple)的形式：（运算符，操作数 1，操作数 2，结果）由于每个陈述都包含了三个变量，即每条指令最多有三个操作数，所以它被称为三地址码

| 指令类型 | 指令格式        | 四元组表示 |
| -------- | --------------- | ---------- |
| 赋值指令 | z=x op y(z=x+y) | (op,x,y,z) |

======= 可以再深入一点

### LLVM IR 内存模型 

当想要在编译的优化层对 LLVM IR 进行优化时，就需要了解 LLVM IR 的内存模型。LLVM IR 的内存模型是基于基本块的，每个基本块都有自己的内存空间，指令只能在其内存空间内执行。

在 LLVM 架构中，几乎所有的东西都是一个 Value。Value 是一个非常基础的基类，一个继承于 Value 的子类表示它的结果可以被其他地方使用。继承于 User 的类表示它会使用一个或多个 Value 对象 根据 Value 与 User 之间的关系，还可以引申出 use-def 链和 def-use 链这两个概念。use-def 链是指被某个 User 使用的 Value 列表，def-use 链是使用某个 Value 的 User 列表。实际上，LLVM 中还定义了一个 Use 类，Use 就是上述的使用关系中的一个边。

- LLVM IR 文件的基本单位称为 module
- 一个 module 中可以拥有多个顶层实体，比如 function 和 global variavle
- 一个 function define 中至少有一个 basicblock
- 每个 basicblock 中有若干 instruction，并且都以 terminator instruction 结尾

1. Module

```shell
; ModuleID = '.\test.c'
source_filename = ".\\test.c"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

......
```

整个.ll 文件就是一个 Module，它包含了 Module 的元数据，比如文件名、目标平台、数据布局等。

Module 类聚合了整个翻译单元用到的所有数据，它是 LLVM 术语中的“module”的同义词。它声明了 Module::iterator typedef，作为遍历这个模块中的函数的简便方法。你可以用 begin()和 end()方法获取这些迭代器。

2. Function

```shell
; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @test(i32 noundef %0, i32 noundef %1) #0 { ;定义全局函数@test(a,b)
  %3 = alloca i32, align 4 ; 局部变量 c
  %4 = alloca i32, align 4 ; 局部变量 d
  %5 = alloca i32, align 4 ; 局部变量 e
  store i32 %0, ptr %3, align 4 ; %0 赋值给%3 c=a
  store i32 %1, ptr %4, align 4 ; %1 赋值给%4 d=b
  %6 = load i32, ptr %3, align 4 ; 读取%3，值给%6 就是参数 a
  %7 = load i32, ptr %4, align 4 ; 读取%4，值给%7 就是参数 b
  %8 = add nsw i32 %6, %7
  store i32 %8, ptr %5, align 4 ; 参数 %9 赋值给%5 e 就是转换前函数写的 int c 变量
  ret void
}
```

在 Module 中，可以定义多个函数(Function)，每个函数都有自己的类型签名、参数列表、局部变量列表、基本块列表、属性列表等。

Function 类包含有关函数定义和声明的所有对象。对于声明来说（用 isDeclaration()检查它是否为声明），它仅包含函数原型。无论定义或者声明，它都包含函数参数的列表，可通过 getArgumentList()方法或者 arg_begin()和 arg_end()这对方法访问它。你可以通过 Function::arg_iteratortypedef 遍历它们。如果 Function 对象代表函数定义，你可以通过这样的语句遍历它的内容：for (Function::iterator i = function.begin(), e = function.end(); i != e; ++i)，你将遍历它的基本块。

3. BasicBlock

每个 Function 可以有多个 BasicBlock，每个 BasicBlock 由若干条指令(Instruction)组成，最后以一个 terminator 指令结束。

BasicBlock 类封装了 LLVM 指令序列，可通过 begin()/end()访问它们。你可以利用 getTerminator()方法直接访问它的最后一条指令，你还可以用一些辅助函数遍历 CFG，例如通过 getSinglePredecessor()访问前驱基本块，当一个基本块有单一前驱时。然而，如果它有多个前驱基本块，就需要自己遍历前驱列表，这也不难，你只要逐个遍历基本块，查看它们的终结指令的目标基本块。

4. Instruction

Instruction 类表示 LLVM IR 的运算原子，一个单一的指令。

利用一些方法可获得高层级的断言，例如 isAssociative()，isCommutative()，isIdempotent()，和 isTerminator()，但是它的精确的功能可通过 getOpcode()获知，它返回 llvm::Instruction 枚举的一个成员，代表了 LLVM IR opcode。可通过 op_begin()和 op_end()这对方法访问它的操作数，它从 User 超类继承得到。

======== 整体再修改下，不是很清晰，很多读起来有点拗口，指代不清楚的地方。

## 总结

本节 LLVM IR 的基本概念，包括三地址码、LLVM IR 表示形式、LLVM IR 基本语法、LLVM IR 指令集、静态单赋值、LLVM IR 内存模型等。在下一节我们将介绍 LLVM 的前端和优化层。

========= 把重点的内容描述一下，高度提炼，参考本节我写的第一篇和第二篇的总结。

## 本节视频

<html>
<iframe src="https://player.bilibili.com/player.html?aid=305431124&bvid=BV1PP411u7NR&cid=900781834&p=1&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>

## 引用

1. https://zh.wikipedia.org/wiki/三位址碼
2. https://buaa-se-compiling.github.io/miniSysY-tutorial/pre/llvm_ir_quick_primer.html
3. https://llvm-tutorial-cn.readthedocs.io/en/latest/chapter-2.html
4. https://buaa-se-compiling.github.io/miniSysY-tutorial/pre/llvm_ir_ssa.html
5. https://buaa-se-compiling.github.io/miniSysY-tutorial/pre/design_hints.html
