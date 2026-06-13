# 实验结果汇总

本文档汇总当前已经完成、可以写入报告的实验结果。未完整跑完或未验证的实验不纳入正式结论。

## 1. 已完成实验

| 阶段 | 实验内容 | 状态 | 输出目录 |
|---|---|---|---|
| E00 | GTSRB 数据集检查与可视化 | 已完成 | `results/00_dataset_check/` |
| E01 | ResNet18 基础交通标志识别 | 已完成 | `results/01_baseline/resnet18/` |
| E02 | FGSM / PGD 对抗攻击 | 已完成 | `results/02_attack/fgsm_pgd/` |
| E03 | 输入预处理防御 | 已完成 | `results/03_defense/input_preprocessing/` |
| E04 | 对抗训练防御 | 未完成 | 不纳入正式结果 |

## 2. 数据集统计

使用 GTSRB 数据集：

| 项目 | 数值 |
|---|---:|
| 类别数 | 43 |
| 训练集图片数 | 26640 |
| 测试集图片数 | 12630 |
| 总图片数 | 39270 |
| 最少训练样本类别数 | 150 |
| 最多训练样本类别数 | 1500 |

报告可用图：

- `reports/figures/fig_01_class_distribution.png`
- `reports/figures/fig_03_train_samples.png`
- `reports/figures/fig_04_test_samples.png`

## 3. 基础识别结果

模型：ResNet18  
输入尺寸：64 x 64  
类别数：43  
最优 epoch：13

| 指标 | 数值 |
|---|---:|
| Test Accuracy | 0.9781 |
| Precision Macro | 0.9678 |
| Recall Macro | 0.9591 |
| F1 Macro | 0.9610 |
| 参数量 | 11190891 |

结论：

ResNet18 在 GTSRB 测试集上取得了较高识别准确率，可以作为后续对抗攻击实验的目标模型。

报告可用图：

- `reports/figures/fig_05_baseline_loss_curve.png`
- `reports/figures/fig_06_baseline_accuracy_curve.png`
- `reports/figures/fig_07_baseline_confusion_matrix.png`
- `reports/figures/fig_08_baseline_per_class_accuracy.png`
- `reports/figures/fig_09_baseline_correct_samples.png`
- `reports/figures/fig_10_baseline_wrong_samples.png`

## 4. 对抗攻击结果

评估设置：

- 模型：基础 ResNet18
- 测试样本：从测试集分层抽样 3000 张
- 攻击方法：FGSM、PGD
- PGD 参数：alpha=0.005，steps=7

### 4.1 FGSM 攻击

| Epsilon | Clean Accuracy | Adversarial Accuracy | Attack Success Rate |
|---:|---:|---:|---:|
| 0.005 | 0.9793 | 0.9527 | 0.0272 |
| 0.010 | 0.9793 | 0.9013 | 0.0796 |
| 0.020 | 0.9793 | 0.7827 | 0.2008 |
| 0.030 | 0.9793 | 0.6710 | 0.3148 |
| 0.050 | 0.9793 | 0.5023 | 0.4871 |

### 4.2 PGD 攻击

| Epsilon | Clean Accuracy | Adversarial Accuracy | Attack Success Rate |
|---:|---:|---:|---:|
| 0.005 | 0.9793 | 0.9467 | 0.0334 |
| 0.010 | 0.9793 | 0.8683 | 0.1133 |
| 0.020 | 0.9793 | 0.6897 | 0.2958 |
| 0.030 | 0.9793 | 0.6000 | 0.3873 |
| 0.050 | 0.9793 | 0.5427 | 0.4459 |

结论：

随着 epsilon 增大，模型在对抗样本上的准确率明显下降。epsilon=0.03 时，FGSM 将准确率降至 67.10%，PGD 将准确率降至 60.00%，说明模型对小幅梯度扰动具有明显脆弱性。

报告可用图：

- `reports/figures/fig_11_attack_accuracy_curve.png`
- `reports/figures/fig_12_attack_success_curve.png`
- `reports/figures/fig_13_attack_confidence_drop.png`
- `reports/figures/fig_14_fgsm_eps003_triplets.png`
- `reports/figures/fig_15_pgd_eps003_triplets.png`

## 5. 输入预处理防御结果

评估设置：

- 攻击方法：FGSM、PGD
- Epsilon：0.03、0.05
- 防御方法：Gaussian Blur、Median Filter、JPEG Compression
- 测试样本：与攻击实验相同的 3000 张分层抽样样本

### 5.1 Epsilon=0.03

| Attack | Before Defense | Gaussian Blur | Median Filter | JPEG Compression |
|---|---:|---:|---:|---:|
| FGSM | 0.6710 | 0.7180 | 0.7107 | 0.7937 |
| PGD | 0.6030 | 0.6733 | 0.6780 | 0.7943 |

### 5.2 Epsilon=0.05

| Attack | Before Defense | Gaussian Blur | Median Filter | JPEG Compression |
|---|---:|---:|---:|---:|
| FGSM | 0.5023 | 0.5577 | 0.5477 | 0.6270 |
| PGD | 0.5447 | 0.6133 | 0.6153 | 0.7173 |

结论：

输入预处理可以缓解部分对抗扰动影响，其中 JPEG Compression 效果最明显。在 PGD epsilon=0.03 场景下，准确率从 60.30% 提升到 79.43%。但该方法仍无法恢复到 clean accuracy，说明简单输入预处理只能作为轻量防御策略，无法完全解决模型鲁棒性问题。

报告可用图：

- `reports/figures/fig_16_fgsm_input_defense_curve.png`
- `reports/figures/fig_17_pgd_input_defense_curve.png`
- `reports/figures/fig_18_input_defense_accuracy_bar.png`
- `reports/figures/fig_19_input_defense_examples.png`

## 6. 当前结论

1. ResNet18 能够在正常 GTSRB 测试集上取得较高准确率。
2. FGSM 和 PGD 都能明显降低模型识别准确率，证明交通标志识别模型存在对抗脆弱性。
3. PGD 在 epsilon=0.03 时比 FGSM 更强，体现迭代攻击的优势。
4. 输入预处理防御能部分恢复准确率，其中 JPEG 压缩效果最佳。
5. 输入预处理防御仍存在上限，后续可继续补充对抗训练等模型层防御方法。

