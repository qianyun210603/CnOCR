# Release Notes


### Update 2023.02.11: 发布 cnocr V2.2.2.2

主要变更：

* 修复了识别很窄图片时异常的问题。
* 修复了对 torchvision 0.14 的兼容问题。


### Update 2022.10.30: 发布 cnocr V2.2.2.1

主要变更：

* 修复了与新版 torch 和 torchvision 不兼容的问题。


### Update 2022.09.09: 发布 cnocr V2.2.2

主要变更：

* 修复HTTP服务存在的问题，感谢 [@Sugobet](https://github.com/Sugobet) 。

* 增加图片分类模型，以及配套的训练和预测脚本，具体见 [图片分类工具](clf_command.md)。

* 适配了新版的pytorch_lightning接口，训练中引入`torchmetrics`计算各种指标。

  

### Update 2022.08.21: 发布 cnocr V2.2.1

主要变更：

* 修复了一些bug。
* 加入了基于 FastAPI 的HTTP服务，使用命令 `cnocr serve` 启动HTTP服务，具体见 [安装说明](install.md)。
* 加入了一些工具脚本，如对截屏图片进行OCR，具体见[cnocr/scripts](https://github.com/breezedeus/CnOCR/tree/master/scripts)。



### Update 2022.07.25: 发布 cnocr V2.2

主要变更：

* CnOCR 内部集成 [CnSTD](https://github.com/breezedeus/cnstd) 进行文本检测，降低使用门槛，提升适用场景的范围。
* 对诸多代码做了重构，同时也对文档进行了大幅度的优化。
* 更新了测试用例，清理了过期的用例。


### Update 2022.05.27: 发布 cnocr V2.1.2.1

主要变更：

* 修复 V2.1.2 bug：打包时忘记把 ppocr 模型相关的字符集文件打包进来了 😭。

### Update 2022.05.25: 发布 cnocr V2.1.2

主要变更：

- 引入了对外部模型的支持，此版加入了对 PaddleOCR 模型的 **ONNX** 版本的支持，具体参见 [可用模型](models.md)；
- 新引入的模型支持识别竖排文字、繁体中文（部分模型），具体参见 [可用模型](models.md)。
- 模型输出结果的格式略有调整，具体参见 [使用方法](usage.md)。

### Update 2022.05.15: 发布 cnocr V2.1.1.1

主要变更：

- 增加了对 **ONNX** 模型的支持，支持 **`*-fc`** 模型，提升预测速度；
- 类 `CnOcr` 的初始化中增加了参数 `model_backend` 和 `vocab_fp`，具体参见 [使用方法](usage.md) ；
- 增加了 `cnocr export-onnx` 命令，把训练好的PyTorch模型导出为ONNX模型；
- 去掉了对包 `python-Levenshtein` 的依赖。

### Update 2021.11.06: 发布 cnocr V2.1.0

主要变更：

* 使用了更精简的模型架构：`densenet_lite_*`；
* 使用了更丰富的数据重新训练了所有模型，精度相较于之前版本更高；
* 提供了更多预训练好的模型；
* 加入了 `cnocr evaluate` 命令以评估效果。

### Update 2021.09.21: 发布 cnocr V2.0.1

主要变更：

* 重新训练了模型，模型识别精度略有提升；
* 函数 `CnOcr.ocr_for_single_lines(img_list, batch_size=1)` 中加入了 `batch_size` 参数。

### Update 2021.08.26: 发布 cnocr V2.0.0

主要变更：

* MXNet 越来越小众化，故从基于 MXNet 的实现转为基于 **PyTorch** 的实现；
* 重新实现了识别模型，优化了训练数据，重新训练模型；
* 优化了能识别的字符集合；
* 优化了对英文的识别效果；
* 优化了对场景文字的识别效果；
* 使用接口略有调整，请谨慎更新。

### Update 2021.08.24: 发布 cnocr V1.2.3

主要变更：

* 更改了模型的默认下载urls；
* 依赖中去掉了对numpy的约束。

### Update 2020.05.29: 发布 cnocr V1.2.2

主要变更：

* `CnOcr`加入类函数 `CnOcr.set_cand_alphabet(cand_alphabet) `。可通过此类函数设置`cand_alphabet`。这样同一个实例也可以指定不同的`cand_alphabet`进行识别。
* bugfix:
  * 修复同时初始化多个实例时会报错的问题。

### Update 2020.05.25: 发布 cnocr V1.2.1

主要变更：

* bugfix:
  * 修复了zip文件名的typo。

### Update 2020.05.25: 发布 cnocr V1.2.0

主要变更：

* 优化了对数字识别的准确度。
* 优化了模型结构，进一步降低了模型的大小，提升了预测速度；最小模型从原来的`6.8M`降为`4.7M`。
* 使用了[爱因互动 Ein+](https://einplus.cn)自己的CDN存储模型文件，下载速度超快。
* 提供了预测速度更快的 `shorter (-s)`版预训练模型：`densenet-lite-s-gru`和`densenet-lite-s-fc`。
* 默认模型由之前的`conv-lite-fc`改为`densenet-lite-fc`。
* 预测支持使用GPU。
* bugfixs:
  * Web 调用时的内存泄露。感谢 [@myuanz](https://github.com/myuanz)；
  * 输入图片宽度很小时导致异常；
  * 去掉  `f-print`。

### Update 2020.04.21: 发布 cnocr V1.1.0

V1.1.0对代码做了很大改动，重写了大部分训练的代码，也生成了更多更难的训练和测试数据。训练好的模型相较于之前版本的模型精度有显著提升，尤其是针对英文单词的识别。

以下列出了主要的变更：

* 更新了训练代码，使用mxnet的`recordio`首先把数据转换成二进制格式，提升后续的训练效率。训练时支持对图片做实时数据增强。也加入了更多可传入的参数。

* **允许训练集中的文字数量不同，目前是中文10个字，英文20个字母。**

* 提供了更多的模型选择，允许大家按需训练多种不同大小的识别模型。

* 内置了各种训练好的模型，最小的模型只有之前模型的`1/5`大小。所有模型都可免费使用。

* 相较于之前版本的模型，新的模型精度有显著提升，尤其是针对英文单词的识别。**新模型已经可以识别英文单词间的空格。**

* **支持文字识别只在给定字符集中进行。** 对于一些纯数字或者纯英文字母的应用场景可以带来识别率提升。

* 优化了对黑底白字多行文字图片的支持。

* mxnet依赖升级到更新的版本了。很多人反馈mxnet `1.4.1`经常找不到没法装，现在升级到`>=1.5.0,<1.7.0`。

### Update 2019.07.25: 发布 cnocr V1.0.0

`cnocr`发布了预测效率更高的新版本v1.0.0。**新版本的模型跟以前版本的模型不兼容**。所以如果大家是升级的话，需要重新下载最新的模型文件。具体说明见下面（流程和原来相同）。

主要改动如下：

- **crnn模型支持可变长预测，提升预测效率**
- 支持利用特定数据对现有模型进行精调（继续训练）
- 修复bugs，如训练时`accuracy`一直为`0`
- 依赖的 `mxnet` 版本从`1.3.1`更新至 `1.4.1`
