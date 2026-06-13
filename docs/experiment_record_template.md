# 实验记录模板

文件命名：

```text
experiments/YYYYMMDD_E编号_实验名称.md
```

## 1. 基本信息

- 实验编号：
- 实验名称：
- 日期：
- 负责人：
- 代码版本：
- 运行设备：

## 2. 实验目的

说明本次实验要验证什么问题。

示例：

验证 ResNet18 在 GTSRB 数据集上的基础识别能力，为后续对抗攻击实验提供目标模型。

## 3. 实验配置

- 数据集：
- 模型：
- 输入尺寸：
- batch size：
- epoch：
- optimizer：
- learning rate：
- random seed：
- 攻击方法，若有：
- 防御方法，若有：

配置文件路径：

```text
configs/xxx.yaml
```

## 4. 运行命令

```bash
python -m src.xxx --config configs/xxx.yaml
```

## 5. 输出路径

```text
results/xx_xxx/xxx/
```

## 6. 关键结果

| 指标 | 数值 |
|---|---:|
| Accuracy | |
| Precision | |
| Recall | |
| F1-score | |
| Attack Success Rate | |
| Robust Accuracy | |

## 7. 保存的图片

| 图片 | 路径 | 用途 |
|---|---|---|
| 训练曲线 |  | 报告基础实验 |
| 混淆矩阵 |  | 错误类别分析 |
| 对抗样本图 |  | 攻击效果展示 |
| 防御对比图 |  | 防御效果展示 |

## 8. 现象与分析

记录观察到的现象，例如：

- 哪些类别容易被误分类。
- epsilon 增大后准确率下降速度。
- PGD 是否比 FGSM 更强。
- 对抗训练是否降低 clean accuracy。
- 防御失败的典型原因。

## 9. 遇到的问题

记录 bug、训练不稳定、数据问题、依赖问题等。

## 10. 下一步计划

说明下一次实验要做什么。

