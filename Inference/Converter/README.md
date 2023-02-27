# 模型转换&优化

《模型转换&优化》在这一节当中分为模型转换和模型优化，在整体架构图中属于离线模型转换模块。一方面，推理引擎需要把不同 AI 框架训练得到的模型进行转换；另外一方面需要对转换后的模型进行图优化等技术。

## 课程部分

> *建议优先下载或者使用PDF版本，PPT版本会因为字体缺失等原因导致版本很丑哦~*

| 名称   | 内容           | 资源                                                                                    | 备注  |
| ---- | ------------ | ------------------------------------------------------------------------------------- | --- |
|      |              |                                                                                       |     |
| 模型转换&优化  | 01 基本介绍           | [slide](./Converter/01.introduction.pdf), [video](https://www.bilibili.com/video/BV1724y1z7ep/)     |     |
| 模型转换模块   | 02 架构与文件格式        | [slide](./Converter/02.converter_princ.pdf), [video](https://www.bilibili.com/video/BV13P4y167sr/)  |     |
| 模型转换模块   | 03 自定义计算图IR       | [slide](./Converter/03.converter_ir.pdf), [video](https://www.bilibili.com/video/BV1rx4y177R9/)     |     |
| 模型转换模块   | 04 流程细节           | [slide](./Converter/04.converter_detail.pdf), [video](https://www.bilibili.com/video/BV13341197zU/) |     |
| 图优化模块    | 05 计算图优化策略        | [slide](./Converter/05.optimizer.pdf), [video](https://www.bilibili.com/video/BV1g84y1L7tF/)        |     |
| 图优化模块    | 06 常量折叠&冗余节点消除    | [slide](./Converter/06.basic.pdf), [video](https://www.bilibili.com/video/BV1fA411r7hr/)            |     |
| 图优化模块    | 07 算子融合/替换/前移     | [slide](./Converter/06.basic.pdf), [video](https://www.bilibili.com/video/BV1Qj411T7Ef/)            |     |
| 图优化模块    | 08 数据布局转换&内存优化    | [slide](./Converter/07.extend.pdf), [video](https://www.bilibili.com/video/BV1Ae4y1N7u7/)           |     |
