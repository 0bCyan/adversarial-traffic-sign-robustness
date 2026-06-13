# 预期结果表

后续实验完成后，将真实结果填入 CSV 或 Markdown 表格。本文件用于提前规划报告需要哪些表。

## 表 1：基础识别模型结果

| 模型 | Accuracy | Precision | Recall | F1-score | 参数量 | 推理时间 |
|---|---:|---:|---:|---:|---:|---:|
| Simple CNN | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| ResNet18 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |

## 表 2：FGSM 攻击结果

| Epsilon | Clean Accuracy | Adversarial Accuracy | Attack Success Rate | Mean Confidence Drop |
|---:|---:|---:|---:|---:|
| 0.005 | 待实验 | 待实验 | 待实验 | 待实验 |
| 0.010 | 待实验 | 待实验 | 待实验 | 待实验 |
| 0.020 | 待实验 | 待实验 | 待实验 | 待实验 |
| 0.030 | 待实验 | 待实验 | 待实验 | 待实验 |
| 0.050 | 待实验 | 待实验 | 待实验 | 待实验 |

## 表 3：PGD 攻击结果

| Epsilon | Clean Accuracy | Adversarial Accuracy | Attack Success Rate | Mean Confidence Drop |
|---:|---:|---:|---:|---:|
| 0.005 | 待实验 | 待实验 | 待实验 | 待实验 |
| 0.010 | 待实验 | 待实验 | 待实验 | 待实验 |
| 0.020 | 待实验 | 待实验 | 待实验 | 待实验 |
| 0.030 | 待实验 | 待实验 | 待实验 | 待实验 |
| 0.050 | 待实验 | 待实验 | 待实验 | 待实验 |

## 表 4：防御前后鲁棒性对比

| 模型/防御方法 | Clean Accuracy | FGSM Accuracy | PGD Accuracy | 备注 |
|---|---:|---:|---:|---|
| 原始 ResNet18 | 待实验 | 待实验 | 待实验 | 无防御 |
| 输入预处理 | 待实验 | 待实验 | 待实验 | 滤波/压缩 |
| 对抗训练 ResNet18 | 待实验 | 待实验 | 待实验 | FGSM 训练 |

