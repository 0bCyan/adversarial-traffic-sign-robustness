# 答辩 PPT 内容提纲

建议 18-20 页，控制在 20 分钟以内。以下内容已经按当前真实实验结果填写。

## 1. 标题页

题目：

**基于对抗攻击与输入防御的交通标志识别系统设计与实现**

副标题：

**以 GTSRB 数据集为例的模型鲁棒性分析**

## 2. 研究背景

要点：

- 交通标志识别是自动驾驶和智能交通的重要感知任务。
- 高准确率不等于高安全性。
- 对抗扰动可能导致模型在视觉变化较小的情况下预测错误。

## 3. 项目目标

展示流程：

```text
基础识别 -> 对抗攻击 -> 输入防御 -> 对比分析
```

要强调：

- 不只是做分类器。
- 重点分析模型脆弱性和防御效果。

## 4. 数据集介绍

使用图：

- `fig_03_train_samples.png`
- `fig_01_class_distribution.png`

关键数据：

- 43 类交通标志。
- 训练集 26640 张。
- 测试集 12630 张。

## 5. 数据预处理

内容：

- 图像统一缩放到 64 x 64。
- 训练阶段使用随机旋转、颜色扰动、随机仿射。
- 使用归一化处理输入。
- 训练集划分 15% 作为验证集。

## 6. 基础识别模型

模型：ResNet18。

改动：

- 第一层卷积改为 3 x 3。
- 去掉初始 maxpool。
- 输出层改为 43 类。

## 7. 基础识别结果

使用图：

- `fig_06_baseline_accuracy_curve.png`
- `fig_07_baseline_confusion_matrix.png`

关键指标：

| 指标 | 数值 |
|---|---:|
| Accuracy | 97.81% |
| Precision Macro | 96.78% |
| Recall Macro | 95.91% |
| F1 Macro | 96.10% |

## 8. 错误案例分析

使用图：

- `fig_10_baseline_wrong_samples.png`

讲解角度：

- 光照较暗。
- 图像模糊。
- 类别相似。
- 标志尺寸较小。

## 9. 对抗样本概念

展示公式：

```text
x_adv = x + delta
||delta|| <= epsilon
f(x_adv) != y
```

讲解：

- 扰动受 epsilon 限制。
- 人眼变化可能不明显。
- 模型预测可能发生明显改变。

## 10. FGSM 原理

公式：

```text
x_adv = x + epsilon * sign(grad_x J(theta, x, y))
```

特点：

- 单步攻击。
- 速度快。
- 适合快速验证模型脆弱性。

## 11. PGD 原理

内容：

- PGD 是迭代攻击。
- 每一步沿梯度方向更新。
- 每次更新后投影回 epsilon 范围。

参数：

- alpha = 0.005。
- steps = 7。

## 12. 攻击可视化

使用图：

- `fig_14_fgsm_eps003_triplets.png`
- `fig_15_pgd_eps003_triplets.png`

讲解：

- 原图、扰动图、对抗样本。
- 模型预测类别和置信度发生变化。

## 13. 攻击结果曲线

使用图：

- `fig_11_attack_accuracy_curve.png`
- `fig_12_attack_success_curve.png`

关键结论：

- epsilon 增大，准确率下降。
- epsilon=0.03 时：
  - FGSM 准确率降到 67.10%。
  - PGD 准确率降到 60.00%。

## 14. 为什么需要防御

使用图：

- `fig_13_attack_confidence_drop.png`

讲解：

- 对抗扰动不仅改变类别，也会改变模型置信度。
- 交通标志识别在安全场景中需要鲁棒性。

## 15. 输入预处理防御方法

三种方法：

- Gaussian Blur：平滑局部噪声。
- Median Filter：抑制异常像素。
- JPEG Compression：削弱高频扰动。

特点：

- 不需要重新训练模型。
- 部署简单。
- 适合作为轻量防御基线。

## 16. 防御结果

使用图：

- `fig_18_input_defense_accuracy_bar.png`

关键结果：

| Attack | Before | Best Defense | After |
|---|---:|---|---:|
| FGSM eps=0.03 | 67.10% | JPEG | 79.37% |
| PGD eps=0.03 | 60.30% | JPEG | 79.43% |
| PGD eps=0.05 | 54.47% | JPEG | 71.73% |

## 17. 防御案例展示

使用图：

- `fig_19_input_defense_examples.png`

讲解：

- 对抗样本预测错误。
- 经过预处理后部分样本恢复正确。
- JPEG 压缩对高频扰动有削弱效果。

## 18. 实验总结

结论：

1. ResNet18 正常识别准确率高，达到 97.81%。
2. FGSM 和 PGD 能明显降低模型准确率。
3. PGD 在 epsilon=0.03 下攻击更强。
4. 输入预处理能部分防御，其中 JPEG Compression 效果最好。
5. 简单输入防御仍有上限，失败案例和 BPDA 自适应攻击说明 JPEG 不是完备安全方案。
6. FGSM 对抗训练作为模型级防御，将 PGD epsilon=0.03 鲁棒准确率提升到 88.63%。

## 19. 不足与改进

不足：

- 尚未完成交互式 Demo。
- 尚未完成 PGD/TRADES 等更强鲁棒训练。
- 尚未完成跨模型结构消融。

改进：

- 完整运行 PGD adversarial training 或 TRADES。
- 补充 SimpleCNN、不同 ResNet stem 的结构消融。
- 使用 Streamlit 做交互式攻击防御演示。

## 20. 小组分工

使用表格展示：

| 成员 | 工作内容 | 贡献占比 |
|---|---|---:|
| 成员 A | 数据处理、可视化 | 20% |
| 成员 B | 基础模型训练 | 20% |
| 成员 C | 对抗攻击实现 | 20% |
| 成员 D | 防御实验实现 | 20% |
| 成员 E | 报告与 PPT | 20% |
