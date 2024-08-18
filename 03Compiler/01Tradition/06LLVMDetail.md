<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->

# LLVM IR 详解

在上一节中，我们已经简要介绍了 LLVM 的基本概念和架构，我们现在将更深入地研究 LLVM 的 IR（中间表示）的概念。

了解 LLVM IR 的重要性是为了能够更好地理解编译器的运作原理，以及在编译过程中 IR 是如何被使用的。LLVM IR 提供了一种抽象程度适中的表示形式，同时能够涵盖绝大多数源代码所包含的信息，这使得编译器能够更为灵活地操作和优化代码。

本节将进一步探究 LLVM IR 的不同表示形式，将有助于我们更好地理解代码在编译器中是如何被处理和转换的。

## LLVM IR 指令集

LLVM IR 是 LLVM 编译器框架中的一种中间语言，它提供了一个抽象层次，使得编译器能够在多个阶段进行优化和代码生成。LLVM IR 具有类精简指令集、使用三地址指令格式的特征，使其在编译器设计中非常强大和灵活。

LLVM IR 的设计理念类似于精简指令集（RISC），这意味着它倾向于使用简单且数量有限的指令来完成各种操作。其指令集支持简单指令的线性序列，比如加法、减法、比较和条件分支等。这使得编译器可以很容易地对代码进行线性扫描和优化。

> RISC 架构的一个重要特征是指令执行的效率较高，因为每条指令都相对简单，执行速度快。

### 三地址指令格式

三地址码是一种中间代码表示形式，广泛用于编译器设计中，LLVM IR 也采用三地址码的方式作为指令集的表示方式。它提供了一种简洁而灵活的方式来描述程序的中间步骤，有助于优化和代码生成。下面是对三地址码的详细总结。

1. 什么是三地址码

三地址码（Three-Address Code, TAC）是一种中间表示形式，每条指令最多包含三个操作数：两个源操作数和一个目标操作数。这些操作数可以是变量、常量或临时变量。三地址码可以看作是一系列的四元组（4-tuple），每个四元组表示一个简单的操作。

2. 四元组表示

每个三地址码指令都可以分解为一个四元组的形式：

```
(运算符, 操作数 1, 操作数 2, 结果)
```

- **运算符**（Operator）：表示要执行的操作，例如加法（`+`）、减法（`-`）、乘法（`*`）、赋值（`=`）等。
- **操作数 1**（Operand1）：第一个输入操作数。
- **操作数 2**（Operand2）：第二个输入操作数（有些指令可能没有这个操作数）。
- **结果**（Result）：操作的输出结果存储的位置。

不同类型的指令可以表示为不同的四元组格式：

| 指令类型  | 指令格式 | 四元组表示  |
| -- | - | --- |
| 赋值指令  | z = x| (`=`, `x`, ``, `z`)|
| 算术指令  | z = x op y | (`op`, `x`, `y`, `z`) |
| 一元运算  | z = op y| (`op`, `y`, ``, `z`)  |
| 条件跳转  | if x goto L| (`if`, `x`, ``, `L`)  |
| 无条件跳转 | goto L  | (`goto`, ``, ``, `L`) |
| 函数调用  | z = call f(a, b) | (`call`, `f`, `(a,b)`, `z`) |
| 返回指令  | return x| (`return`, `x`, ``, ``)  |

3. 三地址码的优点

- **简单性**：三地址码具有简单的指令格式，使得编译器可以方便地进行语义分析和中间代码生成。
- **清晰性**：每条指令执行一个简单操作，便于理解和调试。
- **优化潜力**：由于指令简单且结构固定，编译器可以容易地应用各种优化技术，如常量折叠、死代码消除和寄存器分配。
- **独立性**：三地址码独立于具体机器，可以在不同平台之间移植。

### LLVM IR 中三地址码

LLVM IR 是 LLVM 编译器框架使用的一种中间表示，采用了类似三地址码的设计理念。以下是 LLVM IR 指令集的一些特点：

- **虚拟寄存器**：LLVM IR 使用虚拟寄存器，而不是物理寄存器。这些寄存器以 `%` 字符开头命名。
- **类型系统**：LLVM IR 使用强类型系统，每个值都有一个明确的类型。
- **指令格式**：LLVM IR 指令也可以看作三地址指令，例如：

```llvm
%result = add i32 %a, %b
```

在这条指令中，`%a` 和 `%b` 是输入操作数，`add` 是运算符，`%result` 是结果。因此这条指令可以表示为四元组：

```llvm
(add, %a, %b, %result)
```

### LLVM IR 指令示例

以下是一个简单的 LLVM IR 示例，它展示了一个函数实现：

```llvm
; 定义一个函数，接受两个 32 位整数参数并返回它们的和
define i32 @add(i32 %a, i32 %b) {
entry:
    %result = add i32 %a, %b
    ret i32 %result
}
```

这个例子中，加法指令和返回指令分别可以表示为四元组：

```llvm
(add, %a, %b, %result)
(ret, %result, , )
```

三地址码是一种强大且灵活的中间表示形式，通过使用简单的四元组结构，可以有效地描述程序的中间步骤。LLVM IR 采用了类似三地址码的设计，使得编译器能够高效地进行优化和代码生成。理解三地址码的基本原理和其在 LLVM IR 中的应用，有助于深入掌握编译器技术和优化策略。

## LLVM IR 设计原则

LLVM IR 是一种通用的、低级的虚拟指令集，用于编译器和工具链开发。以下是关于 LLVM IR 的指导原则和最佳实践的总结：

1. 模块化设计

LLVM IR 设计为模块化的，代码和数据分为多个模块，每个模块包含多个函数、全局变量和其他定义。这种设计支持灵活的代码生成和优化。

2. 中间表示层次

LLVM IR 是编译过程中的中间表示，位于源代码和机器码之间。这种层次化设计使得不同语言和目标架构可以共享通用的优化和代码生成技术。

3. 静态单赋值形式（SSA）

LLVM IR 采用 SSA 形式，每个变量在代码中只被赋值一次。SSA 形式简化了数据流分析和优化，例如死代码消除和寄存器分配。

4. 类型系统

LLVM IR 使用强类型系统，支持基本类型（如整数、浮点数）和复合类型（如数组、结构体）。类型系统确保了操作的合法性并支持类型检查和转换。

5. 指令集

LLVM IR 提供丰富的指令集，包括算术运算、逻辑运算、内存操作和控制流指令。每条指令都指定了操作数类型，确保了代码的可移植性和一致性。

6. 优化和扩展

LLVM IR 支持多种优化技术，包括常量折叠、循环优化和内联展开。它还支持通过插件和扩展添加自定义优化和分析。

7. 目标无关性

LLVM IR 设计为目标无关的中间表示，可以跨不同的硬件和操作系统使用。这种目标无关性简化了跨平台编译和优化。

8. 调试支持

LLVM IR 包含丰富的调试信息支持，可以生成调试符号和源代码映射，支持调试器如 GDB 和 LLDB。

这些原则和最佳实践使 LLVM IR 成为一个强大且灵活的工具，用于编译器开发和代码优化。它的模块化设计、强类型系统、丰富的指令集和目标无关性使其适用于广泛的应用场景，从语言前端到高级优化和代码生成。

### 静态单赋值（SSA）

静态单赋值是指当程序中的每个变量都有且只有一个赋值语句时，称一个程序是 SSA 形式的。LLVM IR 中，每个变量都在使用前都必须先定义，且每个变量只能被赋值一次。以 `1*2+3` 为例：

```llvm
%0 = mul i32 1, 2
%0 = add i32 %0, 3
ret i32 %0
```

静态单赋值形式是指每个变量只有一个赋值语句，所以上述代码的 `%0` 不能复用：

```llvm
%0 = mul i32 1, 2
%1 = add i32 %0, 3
ret i32 %1
```

静态单赋值好处： 

1. 每个值都由单一的赋值操作定义，这使得我们可以轻松地从值的使用点直接追溯到其定义的指令。这种特性极大地方便了编译器进行正向和反向的编译过程。

2. 此外，由于静态单赋值（SSA）形式构建了一个简单的使用-定义链，即一个值到达其使用点的定义列表，这极大地简化了代码优化过程。在 SSA 形式下，编译器可以更直观地识别和处理变量的依赖关系，从而提高优化的效率和效果。

### LLVM IR 内存模型 

在进行编译器优化时，需要了解 LLVM IR（中间表示）的内存模型。LLVM IR 的内存模型是基于基本块的，每个基本块都有自己的内存空间，指令只能在其内存空间内执行。

在 LLVM 架构中，几乎所有的实体都是一个 `Value`。`Value` 是一个非常基础的基类，其子类表示它们的结果可以被其他地方使用。`User` 类是继承自 `Value` 的一个类，表示能够使用一个或多个 `Value` 的对象。根据 `Value` 与 `User` 之间的关系，可以引申出 use-def 链和 def-use 链这两个概念。

- use-def 链是指被某个 `User` 使用的 `Value` 列表；

- def-use 链是指使用某个 `Value` 的 `User` 列表。

实际上，LLVM 中还定义了一个 `Use` 类，`Use` 是一个对象，它表示对一个 `Value` 的单个引用或使用。主要作用是帮助 LLVM 跟踪每个 `Value` 的所有使用情况，从而支持 def-use 链的构建和数据流分析。

### LLVM IR 基本单位

1. **Module**

一个 LLVM IR 文件的基本单位是 `Module`。它包含了所有模块的元数据，例如文件名、目标平台、数据布局等。

```llvm
; ModuleID = '.\test.c'
source_filename = ".\\test.c"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"
```

`Module` 类聚合了整个翻译单元中用到的所有数据，是 LLVM 术语中的“module”的同义词。可以通过 `Module::iterator` 遍历模块中的函数，使用 `begin()` 和 `end()` 方法获取这些迭代器。

2. **Function**

在 `Module` 中，可以定义多个函数(`Function`)，每个函数都有自己的类型签名、参数列表、局部变量列表、基本块列表和属性列表等。

```llvm
; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @test(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %6 = load i32, ptr %3, align 4
  %7 = load i32, ptr %4, align 4
  %8 = add nsw i32 %6, %7
  store i32 %8, ptr %5, align 4
  ret void
}
```

`Function` 类包含有关函数定义和声明的所有对象。对于声明（可以用 `isDeclaration()` 检查），它仅包含函数原型。无论是定义还是声明，它都包含函数参数列表，可通过 `getArgumentList()` 或者 `arg_begin()` 和 `arg_end()` 方法访问。

3. **BasicBlock**

每个函数可以有多个基本块(`BasicBlock`)，每个基本块由若干条指令(`Instruction`)组成，最后以一个终结指令（terminator instruction）结束。

`BasicBlock` 类封装了 LLVM 指令序列，可通过 `begin()`/`end()` 访问它们。你可以利用 `getTerminator()` 方法直接访问它的最后一条指令，还可以通过 `getSinglePredecessor()` 方法访问前驱基本块。如果一个基本块有多个前驱，就需要遍历前驱列表。

4. **Instruction**

`Instruction` 类表示 LLVM IR 的运算原子，即单个指令。

可以通过一些方法获得高层级的断言，例如 `isAssociative()`，`isCommutative()`，`isIdempotent()` 和 `isTerminator()`。精确功能可以通过 `getOpcode()` 方法获知，它返回 `llvm::Instruction` 枚举的一个成员，代表 LLVM IR opcode。操作数可以通过 `op_begin()` 和 `op_end()` 方法访问，这些方法从 `User` 超类继承而来。

### LLBM IR 整体示例

以下是一个完整的 LLVM IR 示例，包含 `Module`、`Function`、`BasicBlock` 和 `Instruction`：

```llvm
; ModuleID = '.\test.c'
source_filename = ".\\test.c"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

define dso_local void @test(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %6 = load i32, ptr %3, align 4
  %7 = load i32, ptr %4, align 4
  %8 = add nsw i32 %6, %7
  store i32 %8, ptr %5, align 4
  ret void
}
```

在这个示例中，`Module` 定义了文件的元数据，`Function` 定义了一个函数 `@test`，这个函数有两个 `BasicBlock`，其中包含了一系列的 `Instruction`。

## 小结与思考

- **三地址码：** LLVM IR 使用三地址码形式表示指令，每条指令最多有三个操作数，包括目标操作数和两个源操作数。
 
- **LLVM IR 指令集：** LLVM IR 包含一系列指令用于描述计算、内存访问、分支等操作，能够表示复杂的算法和数据流。

- **静态单赋值：** LLVM IR 使用静态单赋值形式表示变量，每个变量只能被赋值一次，在程序执行过程中可以方便进行分析和优化。

- **LLVM IR 内存模型：** LLVM IR 提供了灵活的内存模型，包括指针操作、内存访问、内存管理等功能，支持复杂的数据结构和算法设计。

在下一节中，将介绍 LLVM 的前端和优化层，这些层将进一步处理 LLVM IR 以及源代码，进行语法分析、优化和生成目标机器码的工作。

## 本节视频

<html>
<iframe src="https://player.bilibili.com/player.html?aid=305431124&bvid=BV1PP411u7NR&cid=900781834&p=1&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>
