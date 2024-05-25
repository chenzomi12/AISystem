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

接下来我们使用Clang编译器将C语言源文件test.c编译成LLVM格式的中间代码。具体参数的含义如下： - clang：Clang编译器。 - -S：生成汇编代码而非目标文件。 - -emit-llvm：生成LLVM IR中间代码。 - .\test.c：要编译的C语言源文件。

```shell
clang -S -emit-llvm .\test.c
```

所生成的.ll 文件的基本语法为：
1. 在LLVM IR（Intermediate Representation）中，指令以分号（;）开头表示注释。
2. 全局表示以 @ 开头，局部变量以 % 开头
3. 使用define关键字定义函数，在本例中定义了两个函数：@test和@main。
4. alloca指令用于在堆栈上分配内存，类似于C语言中的变量声明。
5. store指令用于将值存储到指定地址。
6. load指令用于加载指定地址的值。
7. add指令用于对两个操作数进行加法运算。
8.  i32 32 位 4 个字节的意思  
9.  align 字节对齐
10. ret指令用于从函数返回。

编译完成后，生成的 test.ll 文件内容如下：

```llvm
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

以上程序中包含了两个函数：@test和@main。

@test函数接受两个整型参数并计算它们的和，将结果存储在一个局部变量中。

@main函数分配三个整型变量的内存空间，然后分别赋予初始值，并调用@test函数进行计算。最后@main函数返回整数值0。

程序的完整执行流程如下：

1. 在@main函数中，首先分配三个整型变量的内存空间%1，%2，%3，分别存储0，10，20。 
2. 接下来加载%2和%3的值，将10和20作为参数调用@test函数。 
3. 在@test函数中，分别将传入的参数%0和%1存储至本地变量%3和%4中。 
4. 然后加载%3和%4的值，进行加法操作，并将结果存储至%5中。 
5. 最后，程序返回整数值0。

LLVM IR的代码和C语言编译生成的代码在功能实现上具有完全相同的特性。.ll文件作为LLVM IR的一种中间语言，可以通过LLVM编译器将其转换为机器码，从而实现计算机程序的执行。LLVM IR提供了一种抽象层，使程序员可以更灵活地控制程序的编译和优化过程，同时保留了与硬件无关的特性。通过使用LLVM IR，开发人员可以更好地理解程序的行为，提高代码的可移植性和性能优化的可能性。

### LLVM IR 基本语法

除了上述示例代码中涉及到的基本语法外，LLVM IR作为中间语言也同样有着条件语句、循环体和对指针操作的语法规则。

#### 条件语句：
例如以下c语言代码：
```c
#include <stdio.h>
 
int main()
{
   int a = 10;
   if(a%2 == 0)
	   return 0;
   else 
	   return 1;
}
```

在经过编译后的.ll文件的内容如下所示：
```llvm
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 10, i32* %a, align 4
  %0 = load i32, i32* %a, align 4
  %rem = srem i32 %0, 2
  %cmp = icmp eq i32 %rem, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 0, i32* %retval, align 4
  br label %return

if.else:                                          ; preds = %entry
  store i32 1, i32* %retval, align 4
  br label %return

return:                                           ; preds = %if.else, %if.then
  %1 = load i32, i32* %retval, align 4
  ret i32 %1
}
```

icmp指令是根据比较规则，比较两个操作数，将比较的结果以布尔值或者布尔值向量返回，且对于操作数的限定是操作数为整数或整数值向量、指针或指针向量。其中，eq是比较规则，%rem和0是操作数，i32是操作数类型，比较%rem与0的值是否相等，将比较的结果存放到%cmp中。

br指令有两种形式，分别对应于条件分支和无条件分支。该指令的条件分支在形式上接受一个“i1”值和两个“label”值，用于将控制流传输到当前函数中的不同基本块，上面这条指令是条件分支，类似于c中的三目条件运算符< expression ？Statement ：statement>；无条件分支的话就是不用判断，直接跳转到指定的分支，类似于c中goto，比如说这个就是无条件分支br label %return。`br i1 %cmp, label %if.then, label %if.else`指令的意思是，i1类型的变量%cmp的值如果为真，执行if.then，否则执行if.else。

#### 循环体
例如以下c程序代码：
```c
#include <stdio.h>
 
int main()
{
   int a = 0, b = 1;
   while(a < 5)
   {
	   a++;
	   b *= a;
   }
   return b;
}
```
在经过编译后的.ll文件的内容如下所示：
```llvm
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %a, align 4
  store i32 1, i32* %b, align 4
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %0 = load i32, i32* %a, align 4
  %cmp = icmp slt i32 %0, 5
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %1 = load i32, i32* %a, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* %a, align 4
  %2 = load i32, i32* %a, align 4
  %3 = load i32, i32* %b, align 4
  %mul = mul nsw i32 %3, %2
  store i32 %mul, i32* %b, align 4
  br label %while.cond

while.end:                                        ; preds = %while.cond
  %4 = load i32, i32* %b, align 4
  ret i32 %4
}
```

对比if语句可以发现，while中几乎没有新的指令出现，所以说所谓的while循环，也就是“跳转+分支”这一结构。同理，for循环也可以由“跳转+分支”这一结构构成。

#### 指针
例如以下c程序代码：
```llvm
int main(){
	int i = 10;
	int* pi = &i;
	printf("i的值为：%d",i);
	printf("*pi的值为：%d",*pi);
	printf("&i的地址值为：",%d);
	printf("pi的地址值为：",%d);
}
```

在经过编译后的.ll文件的内容如下所示：
```llvm
@.str = private unnamed_addr constant [16 x i8] c"i\E7\9A\84\E5\80\BC\E4\B8\BA\EF\BC\9A%d\00", align 1
@.str.1 = private unnamed_addr constant [18 x i8] c"*pi\E7\9A\84\E5\80\BC\E4\B8\BA\EF\BC\9A%d\00", align 1
@.str.2 = private unnamed_addr constant [23 x i8] c"&i\E7\9A\84\E5\9C\B0\E5\9D\80\E5\80\BC\E4\B8\BA\EF\BC\9A%p\00", align 1
@.str.3 = private unnamed_addr constant [23 x i8] c"pi\E7\9A\84\E5\9C\B0\E5\9D\80\E5\80\BC\E4\B8\BA\EF\BC\9A%p\00", align 1

define i32 @main(){
entry:
  %i = alloca i32, align 4
  %pi = alloca i32*, align 8
  store i32 10, i32* %i, align 4
  store i32* %i, i32** %pi, align 8
  
  %0 = load i32, i32* %i, align 4
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str, i32 0, i32 0), i32 %0)
  %1 = load i32, i32* %i, align 4
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.1, i32 0, i32 0), i32 %1)
  
  %call2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i32 0, i32 0), i32* %i)
  %2 = load i32*, i32** %pi, align 8
  %call3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.3, i32 0, i32 0), i32* %2)
  ret i32 0
}

declare i32 @printf(i8*, ...)
```

对指针的操作就是指针的指针，开辟一块指针类型的内存，里面放个指针`%pi = alloca i32*, align 8`

此外，c语言中常见的操作还有对数组和结构体的操作，内置函数和外部函数的引用等，更深一步的内容可以参考[简单了解LLVM IR基本语法-CSDN博客](https://blog.csdn.net/qq_42570601/article/details/107157224)

### LLVM IR 指令集

LLVM IR（Intermediate Representation， 中间表示）是LLVM编译器框架中的一种中间语言，它提供了一个抽象层次，使得编译器能够在多个阶段进行优化和代码生成。LLVM IR具有以下特征，使其在编译器设计中非常强大和灵活：
#### 1. 类似精简指令集（RISC）

LLVM IR的设计理念类似于精简指令集（RISC），这意味着它倾向于使用简单且数量有限的指令来完成各种操作。RISC架构的一个重要特征是指令执行的效率较高，因为每条指令都相对简单，执行速度快。

- **线性序列**：LLVM IR指令集支持简单指令的线性序列，比如加法、减法、比较和条件分支等。这使得编译器可以很容易地对代码进行线性扫描和优化。

#### 2. 三地址指令格式

三地址码是一种中间代码表示形式，广泛用于编译器设计中。它提供了一种简洁而灵活的方式来描述程序的中间步骤，有助于优化和代码生成。下面是对三地址码的详细总结。

##### 1. 什么是三地址码

三地址码（Three-Address Code, TAC）是一种中间表示形式，每条指令最多包含三个操作数：两个源操作数和一个目标操作数。这些操作数可以是变量、常量或临时变量。三地址码可以看作是一系列的四元组（4-tuple），每个四元组表示一个简单的操作。

##### 2. 四元组表示

每个三地址码指令都可以分解为一个四元组的形式：

```
(运算符, 操作数1, 操作数2, 结果)
```

- **运算符**（Operator）：表示要执行的操作，例如加法（`+`）、减法（`-`）、乘法（`*`）、赋值（`=`）等。
- **操作数1**（Operand1）：第一个输入操作数。
- **操作数2**（Operand2）：第二个输入操作数（有些指令可能没有这个操作数）。
- **结果**（Result）：操作的输出结果存储的位置。

##### 3. 指令类型及其四元组表示

不同类型的指令可以表示为不同的四元组格式：

| 指令类型  | 指令格式             | 四元组表示                       |
| ----- | ---------------- | --------------------------- |
| 赋值指令  | z = x            | (`=`, `x`, ``, `z`)         |
| 算术指令  | z = x op y       | (`op`, `x`, `y`, `z`)       |
| 一元运算  | z = op y         | (`op`, `y`, ``, `z`)        |
| 条件跳转  | if x goto L      | (`if`, `x`, ``, `L`)        |
| 无条件跳转 | goto L           | (`goto`, ``, ``, `L`)       |
| 函数调用  | z = call f(a, b) | (`call`, `f`, `(a,b)`, `z`) |
| 返回指令  | return x         | (`return`, `x`, ``, ``)     |

##### 4. 三地址码的优点

- **简单性**：三地址码具有简单的指令格式，使得编译器可以方便地进行语义分析和中间代码生成。
- **清晰性**：每条指令执行一个简单操作，便于理解和调试。
- **优化潜力**：由于指令简单且结构固定，编译器可以容易地应用各种优化技术，如常量折叠、死代码消除和寄存器分配。
- **独立性**：三地址码独立于具体机器，可以在不同平台之间移植。

##### 5. LLVM IR中的三地址码

LLVM IR（Intermediate Representation）是LLVM编译器框架使用的一种中间表示，采用了类似三地址码的设计理念。以下是LLVM IR的一些特点：

- **虚拟寄存器**：LLVM IR使用虚拟寄存器，而不是物理寄存器。这些寄存器以`%`字符开头命名。
- **类型系统**：LLVM IR使用强类型系统，每个值都有一个明确的类型。
- **指令格式**：LLVM IR指令也可以看作三地址指令，例如：

  ```llvm
  %result = add i32 %a, %b
  ```

  在这条指令中，`%a`和`%b`是输入操作数，`add`是运算符，`%result`是结果。这条指令可以表示为四元组：

  ```llvm
  (add, %a, %b, %result)
  ```

##### 6. 示例代码

以下是一个简单的LLVM IR示例，它展示了一个函数实现：

```llvm
; 定义一个函数，接受两个32位整数参数并返回它们的和
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

三地址码是一种强大且灵活的中间表示形式，通过使用简单的四元组结构，可以有效地描述程序的中间步骤。LLVM IR采用了类似三地址码的设计，使得编译器能够高效地进行优化和代码生成。理解三地址码的基本原理和其在LLVM IR中的应用，有助于深入掌握编译器技术和优化策略。

## LLVM IR 指导原则

LLVM IR (Intermediate Representation) 是一种通用的、低级的虚拟指令集，用于编译器和工具链开发。以下是关于LLVM IR的指导原则和最佳实践的总结：

#### 1. **模块化设计**
   - LLVM IR设计为模块化的，代码和数据分为多个模块，每个模块包含多个函数、全局变量和其他定义。这种设计支持灵活的代码生成和优化【11†source】【13†source】。

#### 2. **中间表示层次**
   - LLVM IR是编译过程中的中间表示，位于源代码和机器码之间。这种层次化设计使得不同语言和目标架构可以共享通用的优化和代码生成技术【12†source】。

#### 3. **静态单赋值形式（SSA）**
   - LLVM IR采用SSA形式，每个变量在代码中只被赋值一次。SSA形式简化了数据流分析和优化，例如死代码消除和寄存器分配【11†source】【12†source】。

#### 4. **类型系统**
   - LLVM IR使用强类型系统，支持基本类型（如整数、浮点数）和复合类型（如数组、结构体）。类型系统确保了操作的合法性并支持类型检查和转换【11†source】【13†source】。

#### 5. **指令集**
   - LLVM IR提供丰富的指令集，包括算术运算、逻辑运算、内存操作和控制流指令。每条指令都指定了操作数类型，确保了代码的可移植性和一致性【11†source】。

#### 6. **优化和扩展**
   - LLVM IR支持多种优化技术，包括常量折叠、循环优化和内联展开。它还支持通过插件和扩展添加自定义优化和分析【13†source】。

#### 7. **目标无关性**
   - LLVM IR设计为目标无关的中间表示，可以跨不同的硬件和操作系统使用。这种目标无关性简化了跨平台编译和优化【12†source】【13†source】。

#### 8. **调试支持**
   - LLVM IR包含丰富的调试信息支持，可以生成调试符号和源代码映射，支持调试器如GDB和LLDB【12†source】。

这些原则和最佳实践使LLVM IR成为一个强大且灵活的工具，用于编译器开发和代码优化。它的模块化设计、强类型系统、丰富的指令集和目标无关性使其适用于广泛的应用场景，从语言前端到高级优化和代码生成。

### 静态单赋值（SSA）

静态单赋值是指当程序中的每个变量都有且只有一个赋值语句时，称一个程序是 SSA 形式的。LLVM IR 中，每个变量都在使用前都必须先定义，且每个变量只能被赋值一次。以`1*2+3`为例：

```llvm
%0 = mul i32 1, 2
%0 = add i32 %0, 3
ret i32 %0
```

静态单赋值形式是指每个变量只有一个赋值语句，所以上述代码的`%0`不能复用

```llvm
%0 = mul i32 1, 2
%1 = add i32 %0, 3
ret i32 %1
```

静态单赋值好处：  

1.每个值都由单一的赋值操作定义，这使得我们可以轻松地从值的使用点直接追溯到其定义的指令。这种特性极大地方便了编译器进行正向和反向的编译过程。

2.此外，由于静态单赋值（SSA）形式构建了一个简单的使用-定义链，即一个值到达其使用点的定义列表，这极大地简化了代码优化过程。在SSA形式下，编译器可以更直观地识别和处理变量的依赖关系，从而提高优化的效率和效果。

### LLVM IR 内存模型 

在进行编译器优化时，需要了解 LLVM IR（中间表示）的内存模型。LLVM IR 的内存模型是基于基本块的，每个基本块都有自己的内存空间，指令只能在其内存空间内执行。

在 LLVM 架构中，几乎所有的实体都是一个 `Value`。`Value` 是一个非常基础的基类，其子类表示它们的结果可以被其他地方使用。继承自 `User` 的类表示它们会使用一个或多个 `Value` 对象。根据 `Value` 与 `User` 之间的关系，可以引申出 use-def 链和 def-use 链这两个概念。use-def 链是指被某个 `User` 使用的 `Value` 列表，而 def-use 链是指使用某个 `Value` 的 `User` 列表。实际上，LLVM 中还定义了一个 `Use` 类，`Use` 表示上述使用关系中的一个边。

#### LLVM IR 的基本单位

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

#### 示例

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

## 总结

- **三地址码：** LLVM IR 使用三地址码形式表示指令，每条指令最多有三个操作数，包括目标操作数和两个源操作数。
    
- **LLVM IR 表示形式：** LLVM IR 是一种中间表示形式，类似汇编语言但更抽象，用于表示高级语言到机器码的转换过程。
    
- **LLVM IR 基本语法：** LLVM IR 使用类似文本的语法表示，包括指令、注释、变量声明等内容。
    
- **LLVM IR 指令集：** LLVM IR 包含一系列指令用于描述计算、内存访问、分支等操作，能够表示复杂的算法和数据流。
    
- **静态单赋值：** LLVM IR 使用静态单赋值形式表示变量，每个变量只能被赋值一次，在程序执行过程中可以方便进行分析和优化。
    
- **LLVM IR 内存模型：** LLVM IR 提供了灵活的内存模型，包括指针操作、内存访问、内存管理等功能，支持复杂的数据结构和算法设计。
    
在下一节中，将介绍 LLVM 的前端和优化层，这些层将进一步处理 LLVM IR 以及源代码，进行语法分析、优化和生成目标机器码的工作。

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
6. [简单了解LLVM IR基本语法-CSDN博客](https://blog.csdn.net/qq_42570601/article/details/107157224)
