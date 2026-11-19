# ForensicHub 模型分类与统计口径（2026-11-19）

上次的 124 是本地源码中的模型类声明、懒加载声明和兼容依赖导出的名称并集。将其笼统称为“124 个注册模型”不够准确；其中包含大量输入预处理变体、同方法的不同实现，以及特征骨干。它不能直接作为老师要求的实验数量。

本次没有导入完整模型环境来测量运行时 `MODELS.module_dict`。本地 `ForensicHub/common/backbones/__init__.py` 为空，部分骨干在显式导入具体模块时才注册；124 也不是一个已验证能直接构建的运行时模型数。

用户明确排除 Qwen 等生成式 LLM/VLM，并排除 8 个时序视频模型。[项目范围规则](../configs/experiment_scope.yaml)保留图像取证模型和视觉骨干，当前为 **源码发现 124 项、项目纳入 115 项、范围外 9 项**。18 个项目模型已有完整实验记录，项目内另有 97 个条目尚无完整记录；已知适配阻塞为 19 项。范围外 Qwen 和时序视频模型不下载、不训练，也不计入待补或阻塞数。CLIPDetector、OpenCLIP、UnivFD、Effort 和 CO-SPY 保留。

## 1. 124 个条目的来源

| 分类 | 名称条目数 | 说明 |
|---|---:|---|
| 普通图像骨干规格 | 13 | 属于 10 个家族；ResNet 有 3 个深度、SegFormer 有 2 个规格 |
| 骨干输入预处理变体 | 46 | Sobel 10、Bayer 10、FFT 10、DCT 10、Qt 6 |
| Deepfake 源码及兼容导出 | 38 | DeepfakeBench 35，加本地另实现 3；其中 8 个时序视频模型已排除，Effort 作为图像模型保留 |
| IMDL 源码导出 | 13 | 完整定位模型 12，加 SparseViTBackbone 1 |
| AIGC 本地模型 | 7 | UnivFD、CO-SPY、DIRE、HiFi-Net、DualNet、Synthbuster、FatFormer |
| 文档伪造模型 | 5 | CAFTB-Net、DTD、FFDN、PSNet、TIFDM |
| FIDL 的 CLIP LoRA 实验模型 | 1 | CLIP_LORA_PURE |
| 文本生成 VLM（项目范围外） | 1 | Qwen3-VL-8B-Instruct，保留源码来源说明，不下载、不训练 |
| 合计 | **124** | **源码声明/兼容导出数量，包含变体和另实现** |

骨干源码目录的条目合计 60：普通骨干 13 + 变体 46 + Qwen 1。抽象类、包装器和后处理函数没有计入 124。

本项目的视觉骨干条目为 59 项：普通骨干 13 + 变体 46。上表仍保留原始 124 项的来源，实际实验清单采用 115 项范围：排除 Qwen 1 项和时序视频 8 项。

## 2. 通用骨干：按家族理解

| 家族 | 普通 RGB 规格 | 变体数量 |
|---|---|---:|
| ResNet | `Resnet50`、`Resnet101`、`Resnet152` | 5 |
| SegFormer | `Segformerb2`、`Segformerb3` | 5 |
| EfficientNet | `EfficientB4` | 5 |
| Swin | `SwinSmall` | 5 |
| ConvNeXt | `ConvNextSmall` | 5 |
| Xception | `Xception` | 5 |
| UNet | `UNetBase` | 4 |
| ViT | `ViTBase` | 4 |
| MobileNet | `MobileNetV3` | 4 |
| DenseNet | `DenseNet121` | 4 |

例如 `Resnet101`、`SobelResnet101`、`BayerResnet101`、`FFTResnet101`、`DCTResnet101`、`QtResnet101` 是同一个骨干家族的不同输入方案；这些名称不代表六种独立的伪造检测论文方法。ResNet50/101/152 是网络深度规格。

46 个变体应在实验表中单列为预处理/消融组。是否需要全部运行，应依据老师指定的实验范围，不能仅凭它们有源码声明就推断必须全部做。

## 3. 按模型用途分组

本节按用途归类。Effort 虽由 DeepfakeBench 导出，本项目已有配置将它用于 AIGC，因此放入 AIGC 组；这不会额外增加条目。

### 3.1 图像 Deepfake 方法与检测基线：26 项

这里是 DeepfakeBench 的图像/特殊监督方法，已排除时序视频模型和 Effort。LSDA、UCF、IID 等需要域配对或身份/辅助监督，归入这一组不代表它们都能直接用现有 RGB + label 批次运行。

| 模型/源码名称 | 项目配置键 | 完成记录 |
|---|---|---|
| CLIP 检测基线 (`CLIPDetector`) | `clip` | 暂无此实现的完整实验记录 |
| CapsuleNet (`CapsuleNetDetector`) | `capsule-deepfakebench` | 同方法已有其他实现的实验；此实现未验收 |
| CORE (`CoreDetector`) | `core` | 暂无此实现的完整实验记录 |
| EfficientNet-B4 检测基线 (`EfficientDetector`) | `efficient` | 暂无此实现的完整实验记录 |
| F3Net (`F3netDetector`) | `f3net` | 此实现已有完整实验 |
| FFD (`FFDDetector`) | `ffd` | 暂无此实现的完整实验记录 |
| FWA (`FWADetector`) | `fwa` | 暂无此实现的完整实验记录 |
| Face X-ray (`FaceXrayDetector`) | `facexray` | 暂无此实现的完整实验记录 |
| IID (`IIDDetector`) | `iid` | 暂无此实现的完整实验记录 |
| LRL (`LRLDetector`) | `lrl` | 暂无此实现的完整实验记录 |
| LSDA (`LSDADetector`) | `lsda` | 暂无此实现的完整实验记录 |
| Meso4 (`Meso4Detector`) | `meso4` | 暂无此实现的完整实验记录 |
| Meso4Inception (`Meso4InceptionDetector`) | `meso4inception` | 暂无此实现的完整实验记录 |
| Multi-Attention (`MultiAttentionDetector`) | `multiattention` | 暂无此实现的完整实验记录 |
| PCL (`PCLXceptionDetector`) | `pclxception` | 暂无此实现的完整实验记录 |
| RFM (`RFMDetector`) | `rfm` | 暂无此实现的完整实验记录 |
| RECCE (`RecceDetector`) | `recce-deepfakebench` | 同方法已有其他实现的实验；此实现未验收 |
| ResNet34 检测基线 (`ResnetDetector`) | `resnet-registry` | 暂无此实现的完整实验记录 |
| SBI (`SBIDetector`) | `sbi` | 暂无此实现的完整实验记录 |
| SIA (`SIADetector`) | `sia` | 此实现已有完整实验 |
| SLADD (`SLADDXceptionDetector`) | `sladdxception` | 暂无此实现的完整实验记录 |
| SRM (`SRMDetector`) | `srm` | 此实现已有完整实验 |
| SPSL (`SpslDetector`) | `spsl` | 此实现已有完整实验 |
| UCF (`UCFDetector`) | `ucf` | 暂无此实现的完整实验记录 |
| UIA-ViT (`UIAViTDetector`) | `uiavit` | 暂无此实现的完整实验记录 |
| Xception 检测基线 (`XceptionDetector`) | `xception` | 此实现已有完整实验 |

ForensicHub 本地还声明了 3 个同方法的另实现：

| 本地实现 | 同方法的 DeepfakeBench 实现 | 统计方式 |
|---|---|---|
| `Capsule_net` | `CapsuleNetDetector` | 方法家族可去重；不同结构的 checkpoint/实验不能自动互认 |
| `Recce` | `RecceDetector` | 同上 |
| `Spsl` | `SpslDetector` | 同上 |

### 3.2 时序视频 Deepfake：8 项，均在项目范围外

用户已明确排除视频模型。下表只保留源码来源与旧配置键，8 个模型均不下载、不训练，也不作为当前项目待补或阻塞项。LSDA、UCF 等图像批次配对监督不属于时序视频，CLIPDetector 的 2D 图像编码器也继续保留。

| 模型/源码名称 | 项目配置键 | 完成记录 |
|---|---|---|
| AltFreezingDetector (`AltFreezingDetector`) | `altfreezing`（旧键） | 范围外，不下载、不训练 |
| FTCNDetector (`FTCNDetector`) | `ftcn`（旧键） | 范围外，不下载、不训练 |
| I3DDetector (`I3DDetector`) | `i3d`（旧键） | 范围外，不下载、不训练 |
| STILDetector (`STILDetector`) | `stil`（旧键） | 范围外，不下载、不训练 |
| TALLDetector (`TALLDetector`) | `tall`（旧键） | 范围外，不下载、不训练 |
| TimeSformerDetector (`TimeSformerDetector`) | `timesformer`（旧键） | 范围外，不下载、不训练 |
| VideoMAEDetector (`VideoMAEDetector`) | `videomae`（旧键） | 范围外，不下载、不训练 |
| XCLIPDetector (`XCLIPDetector`) | `xclip`（旧键） | 范围外，不下载、不训练 |

### 3.3 IMDL 图像篡改定位：12 个完整模型

| 模型/源码名称 | 项目配置键 | 完成记录 |
|---|---|---|
| APSCNet (`APSCNet`) | `apsc` | 暂无此实现的完整实验记录 |
| Cat_Net (`Cat_Net`) | `cat-net` | 暂无此实现的完整实验记录 |
| IML_ViT (`IML_ViT`) | `iml-vit` | 此实现已有完整实验 |
| MSCDI_Net (`MSCDI_Net`) | `mscdi` | 暂无此实现的完整实验记录 |
| MVSSNet (`MVSSNet`) | `mvss` | 暂无此实现的完整实验记录 |
| MantraNet (`MantraNet`) | `mantranet` | 暂无此实现的完整实验记录 |
| Mesorch (`Mesorch`) | `mesorch` | 此实现已有完整实验 |
| ObjectFormer (`ObjectFormer`) | `objectformer` | 暂无此实现的完整实验记录 |
| PSCC_Net (`PSCC_Net`) | `pscc` | 暂无此实现的完整实验记录 |
| SPAN (`SPAN`) | `span` | 暂无此实现的完整实验记录 |
| SparseViT (`SparseViT`) | `sparse-vit` | 暂无此实现的完整实验记录 |
| Trufor (`Trufor`) | `trufor` | 此实现已有完整实验 |

另有 `SparseViTBackbone`，它是 SparseViT 的特征骨干类，应归入组件/骨干清单；不要把它当作一篇新的定位方法。

### 3.4 AIGC 检测：8 项

包含 7 个 ForensicHub 本地 AIGC 模型和兼容库导出的 Effort。

| 模型/源码名称 | 项目配置键 | 完成记录 |
|---|---|---|
| CO_SPY (`CO_SPY`) | `co-spy` | 此实现已有完整实验 |
| Dire (`Dire`) | `dire` | 暂无此实现的完整实验记录 |
| DualNet (`DualNet`) | `dualnet` | 暂无此实现的完整实验记录 |
| Effort (`EffortDetector`) | `effort` | 此实现已有完整实验 |
| FatFormer (`FatFormer`) | `fatformer` | 暂无此实现的完整实验记录 |
| HiFi_Net (`HiFi_Net`) | `hifi-net` | 暂无此实现的完整实验记录 |
| Synthbuster (`Synthbuster`) | `synthbuster` | 暂无此实现的完整实验记录 |
| UnivFD (`UnivFD`) | `univfd` | 此实现已有完整实验 |

### 3.5 文档伪造检测/定位：5 项

| 模型/源码名称 | 项目配置键 | 完成记录 |
|---|---|---|
| CAFTB_Net (`CAFTB_Net`) | `caftb` | 暂无此实现的完整实验记录 |
| DTD (`DTD`) | `dtd` | 暂无此实现的完整实验记录 |
| FFDN (`FFDN`) | `ffdn` | 暂无此实现的完整实验记录 |
| PSNet (`PSNet`) | `psnet` | 暂无此实现的完整实验记录 |
| Tifdm (`Tifdm`) | `tifdm` | 暂无此实现的完整实验记录 |

### 3.6 其他扩展

| 名称 | 建议归类 | 说明 |
|---|---|---|
| `CLIP_LORA_PURE` | 跨域 CLIP LoRA 实验基线 | 是可训练的适配模型，与完整 forensic 论文方法、原始 CLIP 基线分别记录 |
| `Qwen3-VL-8B-Instruct` | 源码来源记录，已排除 | 用户明确划为项目范围外；不下载、不训练，不作为待补实验 |

## 4. 更贴近四域图像实验的官方配置范围

逐文件读取本地 `ForensicHub/ForensicHub/statics/crossdataset_image/*train.yaml`，共有 **45 份训练配置、45 个不同模型目标**。展开 Mask2LabelWrapper 后，分为：

| 基础目标分类 | 数量 | 名称 |
|---|---:|---|
| RGB 骨干 | 6 | ResNet101、EfficientNet-B4、SegFormer-B3、Swin-Small、ConvNeXt-Small、Xception |
| Deepfake | 3 | CapsuleNet、RECCE、SPSL（此模板使用本地实现） |
| IMDL | 5 | IML-ViT、TruFor、Mesorch、MVSS、PSCC |
| AIGC | 5 | CO-SPY、DualNet、FatFormer、HiFi-Net、UnivFD |
| Document | 2 | DTD、FFDN |
| 基础模型目标合计 | **21** | |
| 预处理变体 | **24** | 上述 6 个骨干分别增加 Sobel、Bayer、DCT、Qt |
| 配置合计 | **45** | 这一目录没有对应的 FFT 训练模板 |

这 45 份模板的范围比兼容库的全量类名更接近原有跨域图像实验；但它也不是教师要求的权威数量。SIA、Effort、SRM、F3Net、后续新增定位模型等并未全部出现在这一目录。本项目已经运行的 18 个模型仍保留，不能因为某个模型缺少这个目录的模板就删除其记录。

建议项目内实验清单按“领域检测模型、通用骨干基线、预处理变体、需要特殊图像监督的模型”分别列出。Qwen 等生成式 LLM/VLM 和 8 个时序视频模型只保留范围外源码记录，不作为必做扩展。

## 5. 当前已经完成的 18 个项目模型

| 实验分组 | 已完成数量 | 已完成模型 |
|---|---:|---|
| 通用骨干 | 5 | ResNet101、EfficientNet-B4、SegFormer-B3、Swin-Small、ConvNeXt-Small |
| Deepfake | 7 | CapsuleNet、RECCE、SIA、SPSL、SRM、F3Net、Xception 检测器 |
| IMDL | 3 | IML-ViT、TruFor、Mesorch |
| AIGC | 3 | UnivFD、CO-SPY、Effort |
| Document | 0 | FFDN 未完成 |
| 合计 | **18** | 不同初始化合计 23 组完整实验 |

完成状态沿用已有验收记录和 2026-11-18 总结文档，未在本次重新访问 A40。算法家族去重仅用于分类；权重加载和结果验收仍按具体实现记录。

HF 国内镜像能用于准备部分模型权重，但下载方式不改变上述分类：SparseViT 的 Uniformer、timm 视觉骨干、CLIP/OpenCLIP/VAE 初始化与完整检测任务参数分别归档。HF 可用来源及格式限制见 [下载清单的 HF 部分](未完成模型与ckpt下载地址_20261119.md#8-hf-国内镜像可获取的资源与格式)。本轮实际下载状态由 A40 归档记录确认。

## 6. 骨干预处理变体完整名录

### Sobel（10 项）

`SobelConvNextSmall`、`SobelDenseNet121`、`SobelEfficientB4`、`SobelMobileNetV3`、`SobelResnet101`、`SobelSegformerb3`、`SobelSwinSmall`、`SobelUNet`、`SobelViTBase`、`SobelXception`。

### Bayer（10 项）

`BayerConvNextSmall`、`BayerDenseNet121`、`BayerEfficientB4`、`BayerMobileNetV3`、`BayerResnet101`、`BayerSegformerb3`、`BayerSwinSmall`、`BayerUNet`、`BayerViTBase`、`BayerXception`。

### FFT（10 项）

`FFTConvNextSmall`、`FFTDenseNet121`、`FFTEfficientB4`、`FFTMobilenetV3`、`FFTResnet101`、`FFTSegformerb3`、`FFTSwinSmall`、`FFTUNet`、`FFTViTBase`、`FFTXception`。

### DCT（10 项）

`DCTConvNextSmall`、`DCTDenseNet121`、`DCTEfficientB4`、`DCTMobilenetV3`、`DCTResnet101`、`DCTSegformerb3`、`DCTSwinSmall`、`DCTUNet`、`DCTViTBase`、`DCTXception`。

### Qt（6 项）

`QtConvNextSmall`、`QtEfficientB4`、`QtResnet101`、`QtSegformerb3`、`QtSwinSmall`、`QtXception`。

## 7. 核对来源

- [静态声明扫描代码](../src/forensichub_inventory.py)
- [DeepfakeBench 导出清单：DeepfakeBench/training/detectors/__init__.py](https://github.com/scu-zjz/DeepfakeBench)
- [IMDLBenCo 模型导出清单：IMDLBenCo/model_zoo/__init__.py](https://github.com/scu-zjz/IMDLBenCo)
- [官方跨域图像配置目录：ForensicHub/statics/crossdataset_image](https://github.com/scu-zjz/ForensicHub)
- [项目完成记录](experiment_completion_20261119.json)
- [未完成模型与 ckpt 下载地址](未完成模型与ckpt下载地址_20261119.md)
