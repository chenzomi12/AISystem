# AI编译器 -- 后端优化

《后端优化》后端优化作为AI编译器跟硬件之间的相连接的模块，更多的是算子或者Kernel进行优化，而优化之前需要把计算图转换称为调度树等IR格式，然后针对每一个算子/Kernel进行循环优化、指令优化和内存优化等技术。

## 内容大纲

> *建议优先下载或者使用PDF版本，PPT版本会因为字体缺失等原因导致版本很丑哦~*

| 名称   | 名称               | 备注                                                                                    |
| ---- | ---------------- | ------------------------------------------------------------------------------------- |
|      |                  |                                                                                       |
| 后端优化 | 01 AI编译器后端优化介绍   | [silde](./01.introduction.pdf), [video](https://www.bilibili.com/video/BV17D4y177bP/) |
| 后端优化 | 02 算子分为计算与调度     | [silde](./02.ops_compute.pdf), [video](https://www.bilibili.com/video/BV1K84y1x7Be/)  |
| 后端优化 | 03 算子优化手工方式      | [silde](./03.optimization.pdf), [video](https://www.bilibili.com/video/BV1ZA411X7WZ/) |
| 后端优化 | 04 算子循环优化        | [silde](./04.loop_opt.pdf), [video](https://www.bilibili.com/video/BV17D4y177bP/)     |
| 后端优化 | 05 指令和内存优化       | [silde](./05.other_opt.pdf), [video](https://www.bilibili.com/video/BV11d4y1a7J6/)    |
| 后端优化 | 06 Auto-Tuning原理 | [silde](./06.auto_tuning.pdf), [video](https://www.bilibili.com/video/BV1uA411D7JF/)  |
