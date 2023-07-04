# AI芯片：核心原理

AI芯片这里就很硬核了，从芯片的基础到AI芯片的范围都会涉及，芯片设计需要考虑上面AI框架的前端、后端编译，而不是停留在天天喊着吊打英伟达，被现实打趴。

> 欢迎大家使用的过程中发现bug或者勘误直接提交PR到开源社区哦！

> 请大家尊重开源和作者的努力，引用PPT的内容请规范转载标明出处哦！

## 一、ChatGPT系列视频（乔凯——DONE）

1 01 GPT系列详解

02 02 RLHF强化学习PPO

03 03 InstructGPT解读

一、AI框架系列视频（谢鑫鑫、赵含霖、粟君杰、管一鸣、柯德）

1 AI框架基础——柯德

1. 基本介绍 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Foundation/01.introduction.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1he4y1z7oD/%3Fvd_source%3D26de035c60e6c7f810371fdfd13d14b6)

2. AI框架有什么用 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Foundation/02.fundamentals.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1fd4y1q7qk/%3Fvd_source%3D26de035c60e6c7f810371fdfd13d14b6)

3. AI框架之争（框架发展）[silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Foundation/03.history.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1C8411x7Kn/%3Fvd_source%3D26de035c60e6c7f810371fdfd13d14b6)

4. 编程范式（声明式&命令式）[silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Foundation/04.programing.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1gR4y1o7WT/%3Fvd_source%3D26de035c60e6c7f810371fdfd13d14b6)

5. 自动微分——谢鑫鑫

6. 基本介绍 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/01.introduction.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1FV4y1T7zp/),[article](https://zhuanlan.zhihu.com/p/518198564)

7. 什么是微分 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/02.base_concept.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Ld4y1M7GJ/),[article](https://zhuanlan.zhihu.com/p/518198564)

8. 正反向计算模式 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/03.grad_mode.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1zD4y117bL/),[article](https://zhuanlan.zhihu.com/p/518296942)

9. 三种实现方法 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/04.grad_mode.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1BN4y1P76t/),[article](https://zhuanlan.zhihu.com/p/520065656)

10. 手把手实现正向微分框架 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/05.forward_mode.ipynb),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Ne4y1p7WU/),[article](https://zhuanlan.zhihu.com/p/520451681)

11. 亲自实现一个PyTorch [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/06.reversed_mode.ipynb),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1ae4y1z7E6/),[article](https://zhuanlan.zhihu.com/p/547865589)

12. 自动微分的挑战&未来 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AutoDiff/07.challenge.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV17e4y1z73W/)

13. 计算图——赵含霖

14. 基本介绍 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/01.introduction.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1cG411E7gV/)

15. 什么是计算图 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/02.computation_graph.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1rR4y197HM/)

16. 计算图跟自动微分关系 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/03.atuodiff.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1S24y197FU/)

17. 图优化与图执行调度 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/04.dispatch.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1hD4y1k7Ty/)

18. 计算图的控制流机制实现 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/05.control_flow.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV17P41177Pk/)

19. 计算图未来将会走向何方？[silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/DataFlow/06.future.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1hm4y1A7Nv/)

20. 分布式集群——管一鸣

21. 基本介绍 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/01.introduction.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1ge411L7mi/)

22. AI集群服务器架构 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/02.architecture.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1fg41187rc/)

23. AI集群软硬件通信 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/03.communication.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV14P4y1S7u4/)

24. 集合通信原语 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/04.primitive.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1te4y1e7vz/)

25. AI框架分布式功能 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/05.system.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1n8411s7f3/)

26. 分布式算法——赵含霖

27. 大模型训练的挑战 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/06.challenge.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Y14y1576A/)

28. 算法：大模型算法结构 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/07.algorithm_arch.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Mt4y1M7SE/)

29. 算法：亿级规模SOTA大模型 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/08.algorithm_sota.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1em4y1F7ay/)

30. 分布式并行——粟君杰

31. 基本介绍 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/01.introduction.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1ve411w7DL/)

32. 数据并行 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/02.data_parallel.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1JK411S7gL/)

33. 模型并行之张量并行 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/03.tensor_parallel.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1vt4y1K7wT/)

34. MindSpore张量并行 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/04.mindspore_parallel.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1vt4y1K7wT/)

35. 模型并行之流水并行 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/05.pipeline_parallel.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1WD4y1t7Ba/)

36. 混合并行 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/06.hybrid_parallel.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1gD4y1t7Ut/)

37. 分布式训练总结 [silde](https://link.zhihu.com/?target=https%3A//github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/Parallel/07.summary.pptx),[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1av4y1S7DQ/)

二、AI编译器系列视频（陈志宇、魏铭康、刘旭斌、杨绎、凝渊）

1 编译器基础——陈志宇

1. 课程概述 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1D84y1y73v/)

2. 传统编译器——陈志宇

3. 开源编译器的发展 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1sM411C7Vr/)

4. GCC编译过程和原理 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1LR4y1f7et/)

5. LLVM设计架构 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1CG4y1V7Dn/)

6. (上) LLVM IR详解 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1PP411u7NR/)

7. (中) LLVM前端和优化层 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1vd4y1t7vS/)

8. (下) LLVM后端代码生成 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1cd4y1b7ho/)

9. AI 编译器——凝渊

10. 为什么需要AI编译器 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1pM41167KP/)

11. AI编译器的发展阶段 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1QK411R7iy/)

12. AI编译器的通用架构 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1qD4y1Y73e/)

13. AI编译器的挑战与思考 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Hv4y1R7uc/)

14. 前端优化——魏铭康

15. 内容介绍 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1ne411w7n2/)

16. 计算图层IR [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1kV4y1w72W/)

17. 算子融合策略 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1P24y1D7RV/)

18. (上) 布局转换原理 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1xK411z7Uw/)

19. (下) 布局转换算法 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1gd4y1Y7dc/)

20. 内存分配算法 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1nM411879s/)

21. 常量折叠原理 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1P8411W7dY/)

22. 公共表达式消除 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1rv4y1Q7tp/)

23. 死代码消除 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1hD4y1h7nh/)

24. 代数简化原理 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1g24y1Q7qC/)

25. 后端优化——刘旭斌

26. AI编译器后端优化介绍 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV17D4y177bP/)

27. 算子分为计算与调度 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1K84y1x7Be/)

28. 算子优化手工方式 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1ZA411X7WZ/)

29. 算子循环优化 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1r14y1w7hG/)

30. 指令和内存优化 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV11d4y1a7J6/)

31. Auto-Tuning原理 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1uA411D7JF/)

32. PyTorch2.0——杨绎

33. PyTorch2.0 特性串讲 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1p84y1675B/)

34. TorchScript 静态图尝试 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1JV4y1P7gB/)

35. Torch FX 与 LazyTensor 特性 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1944y1m7fU/)

36. TorchDynamo 来啦 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1jD4y1a7hx/)

37. AOTAutograd 原理 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Me4y1V7Ke/)

38. Dispatch 机制 [video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1L3411d7SM/)

三、推理引擎系列视频（邓实诚、王远航、孙仲琦、曹泽沛）

1. 推理系统——邓实诚

2. 推理内容介绍（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1J8411K7pj/)）

3. 什么是推理系统（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1nY4y1f7G5/)）

4. 推理流程全景（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1M24y1v7rK/)）

5. 推理系统架构（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Gv4y1i7Tw/)）

6. (上) 推理引擎架构（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Mx4y137Er/)）

7. (下) 推理引擎架构（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1FG4y1C7Mn/)）

8. 模型小型化——邓实诚

9. 推理参数了解（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1KW4y1G75J/)）

10. (上) CNN模型小型化（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Y84y1b7xj/)）

11. (下) CNN模型小型化（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1DK411k7qt/)）

12. Transformer小型化（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV19d4y1V7ou/)）

13. 模型压缩——孙仲琦

14. 压缩四件套介绍（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1384y187tL/)）

15. 低比特量化原理（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1VD4y1n7AR/)）

16. 感知量化训练 QAT（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1s8411w7b9/)）

17. 训练后量化PTQ与部署（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1HD4y1n7E1/)）

18. 模型剪枝（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1y34y1Z7KQ/)）

19. (上) 知识蒸馏原理（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1My4y197Tf/)）

20. (下) 知识蒸馏算法（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1vA411d7MF/)）

21. 模型转换——曹泽沛

22. 基本介绍（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1724y1z7ep/)）

23. 架构与文件格式（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV13P4y167sr/)）

24. 自定义计算图IR（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1rx4y177R9/)）

25. 流程细节（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV13341197zU/)）

26. 图优化模块——曹泽沛

27. 计算图优化策略（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1g84y1L7tF/)）

28. 常量折叠&冗余节点消除（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1fA411r7hr/)）

29. 算子融合/替换/前移（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Qj411T7Ef/)）

30. 数据布局转换&内存优化（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Ae4y1N7u7/)）

31. Kernel优化——王远航

32. Kernel优化架构（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Ze4y1c7Bb/)）

33. 卷积操作基础原理（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1No4y1e7KX/)）

34. Im2Col算法（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Ys4y1o7XW/)）

35. Winograd算法（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1vv4y1Y7sc/)）

36. QNNPack算法（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1ms4y1o7ki/)）

37. 推理内存布局（[video](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1eX4y1X7mL/)）

四、AI芯片系列视频（李敏涛、凝渊、张泽斌、郝嘉伟）

1. AI 计算体系——李敏涛

2. 课程内容

3. AI计算模式(上)

4. AI计算模式(下)

5. 关键设计指标

6. 05 核心计算：矩阵乘

7. 06 数据单位：bits

8. 07 AI计算体系总结

9. AI 芯片基础——凝渊

10. 01 CPU 基础

11. 02 CPU 指令集架构

12. 03 CPU 计算本质

13. 04 CPU 计算时延

14. 05 GPU 基础

15. 06 NPU 基础

16. 07 超异构计算

17. 通用图形处理器 GPU——张泽斌

18. 01 GPU工作原理

19. 02 GPU适用于AI

20. 03 GPU架构与CUDA关系

21. 04 GPU架构回顾第一篇

22. 05 GPU架构回顾第二篇

23. 英伟达GPU的AI详解——郝嘉伟

24. 01 TensorCore原理(上)

25. 02 TensorCore架构(中)

26. 03 TensorCore剖析(下)

27. 分布式与NVLink关系

28. AI专用处理器 NPU