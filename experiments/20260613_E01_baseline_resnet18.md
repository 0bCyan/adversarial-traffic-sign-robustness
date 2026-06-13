# E01 ResNet18 基础识别实验

## 1. 基本信息

- 实验编号：E01
- 实验名称：ResNet18 交通标志基础识别
- 日期：2026-06-13
- 数据集：GTSRB
- 配置文件：`configs/baseline_resnet18.yaml`
- 输出目录：`results/01_baseline/resnet18/`

## 2. 实验目的

训练一个正常交通标志识别模型，作为后续 FGSM 和 PGD 对抗攻击实验的目标模型。

## 3. 实验配置

- 模型：ResNet18
- 输入尺寸：64 x 64
- 类别数：43
- Batch size：256
- Epoch：30
- 优化器：AdamW
- 学习率：0.001
- 调度器：Cosine
- 验证集比例：15%

## 4. 运行命令

```bash
python -m src.train_classifier --config configs/baseline_resnet18.yaml
```

## 5. 关键结果

| 指标 | 数值 |
|---|---:|
| Test Accuracy | 0.9781 |
| Precision Macro | 0.9678 |
| Recall Macro | 0.9591 |
| F1 Macro | 0.9610 |
| Best Epoch | 13 |

## 6. 保存素材

- `reports/figures/fig_05_baseline_loss_curve.png`
- `reports/figures/fig_06_baseline_accuracy_curve.png`
- `reports/figures/fig_07_baseline_confusion_matrix.png`
- `reports/figures/fig_08_baseline_per_class_accuracy.png`
- `reports/figures/fig_09_baseline_correct_samples.png`
- `reports/figures/fig_10_baseline_wrong_samples.png`
- `reports/tables/table_03_baseline_test_metrics.json`

## 7. 结论

ResNet18 在正常测试集上取得较高准确率，能够作为可靠的基础识别模型和后续对抗攻击目标模型。

