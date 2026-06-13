# 实验与实现计划

## 1. 项目目标

本项目围绕交通标志识别任务，研究深度神经网络在对抗扰动下的脆弱性，并设计防御方法提升模型鲁棒性。

项目不是只完成一个分类器，而是形成完整实验闭环：

```text
基础识别模型 -> 对抗攻击 -> 脆弱性分析 -> 鲁棒防御 -> 对比评估 -> 可视化演示
```

最终交付内容包括：

- 可复现代码。
- 训练好的模型或模型下载说明。
- 实验报告。
- 答辩 PPT。
- 攻击与防御演示 Demo。
- 完整结果图、指标表和中间过程图片。

## 2. 推荐题目

**基于对抗攻击与鲁棒防御的交通标志识别系统设计与实现**

可选副标题：

**以 GTSRB 数据集为例的模型安全性分析与防御实验**

## 3. 数据集计划

### 3.1 主数据集

使用 GTSRB 交通标志识别数据集。

特点：

- 类别数：43 类。
- 图片内容直观，适合展示。
- 属于经典图像分类数据集，便于复现和对比。
- 可用于分析自动驾驶、智能交通等应用场景下的模型安全问题。

### 3.2 数据处理

计划处理流程：

1. 下载并解压 GTSRB 数据集。
2. 统一图像大小，例如 `64 x 64`。
3. 划分训练集、验证集和测试集。
4. 统计类别分布。
5. 保存样例图和类别统计图。

### 3.3 必须保存的展示素材

保存到 `results/00_dataset_check/`：

- `class_distribution.png`：类别数量分布柱状图。
- `sample_grid_train.png`：训练集样例宫格图。
- `sample_grid_test.png`：测试集样例宫格图。
- `image_size_distribution.png`：原始图片尺寸分布图。
- `dataset_summary.csv`：类别数量、训练/测试样本数统计。

复制精选图片到 `reports/figures/`：

- `fig_dataset_samples.png`
- `fig_class_distribution.png`

## 4. 阶段一：基础交通标志识别

### 4.1 实验目的

训练一个正常交通标志识别模型，作为后续攻击和防御实验的基础。

### 4.2 模型选择

建议实现两个层次：

- Baseline：简单 CNN。
- 主模型：ResNet18。

如果时间有限，优先完成 ResNet18。

### 4.3 训练设置

建议配置：

- 输入尺寸：`64 x 64`
- 优化器：AdamW
- 学习率：`1e-3`
- Epoch：`30`
- 损失函数：CrossEntropyLoss
- 数据增强：随机旋转、颜色扰动、随机仿射

### 4.4 评价指标

需要保存：

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- 每类准确率
- 推理速度，可选

### 4.5 必须保存的中间结果

保存到 `results/01_baseline/resnet18/`：

```text
checkpoints/
  best_model.pth
  last_model.pth
metrics/
  train_log.csv
  validation_log.csv
  test_metrics.json
  per_class_metrics.csv
figures/
  loss_curve.png
  accuracy_curve.png
  confusion_matrix.png
  per_class_accuracy.png
  correct_samples_grid.png
  wrong_samples_grid.png
```

报告中可展示：

- 模型结构图。
- 训练 loss / accuracy 曲线。
- 混淆矩阵。
- 错误分类案例。

## 5. 阶段二：对抗扰动攻击

### 5.1 实验目的

验证交通标志识别模型对细微扰动的敏感性，展示神经网络的安全风险。

### 5.2 攻击方法

至少完成：

1. FGSM：Fast Gradient Sign Method。
2. PGD：Projected Gradient Descent。

FGSM 用于讲清楚原理，PGD 用于体现攻击强度。

### 5.3 实验变量

扰动强度 epsilon 建议：

```text
0.005, 0.01, 0.02, 0.03, 0.05
```

对每个 epsilon 保存：

- 对抗样本准确率。
- 攻击成功率。
- 平均置信度变化。
- 若干攻击成功样例。
- 若干攻击失败样例。

### 5.4 必须保存的中间图片

保存到 `results/02_attack/fgsm_pgd/`：

```text
metrics/
  fgsm_metrics.csv
  pgd_metrics.csv
  attack_summary.json
figures/
  epsilon_accuracy_curve.png
  attack_success_rate_curve.png
  confidence_drop_curve.png
samples/
  fgsm_eps_0.01_grid.png
  fgsm_eps_0.03_grid.png
  pgd_eps_0.01_grid.png
  pgd_eps_0.03_grid.png
  noise_map_examples.png
  original_noise_adversarial_triplets.png
```

每组攻击样例图建议包含：

```text
原始图片 | 扰动热力图 | 对抗样本 | 原预测 | 攻击后预测 | 置信度变化
```

报告中可展示：

- FGSM 原理公式。
- PGD 迭代攻击流程。
- 原图、扰动、对抗样本三联图。
- epsilon 增大时准确率下降曲线。
- 攻击成功案例和失败案例。

## 6. 阶段三：鲁棒防御

### 6.1 实验目的

尝试提升模型在对抗样本下的识别能力，比较不同防御策略的效果。

### 6.2 防御方法

建议实现两类防御：

1. 输入预处理防御。
2. 对抗训练。

输入预处理方法可选：

- Gaussian Blur
- Median Filter
- JPEG Compression
- Denoising

对抗训练：

- 在训练阶段混合干净样本和 FGSM 对抗样本。
- 比例建议：`clean_ratio=0.5`, `adversarial_ratio=0.5`。

### 6.3 对比实验

需要比较：

- 原始 ResNet18。
- 输入预处理防御。
- 对抗训练 ResNet18。

测试场景：

- Clean Test Set
- FGSM Attack
- PGD Attack

### 6.4 必须保存的结果

保存到 `results/03_defense/`：

```text
adversarial_training/
  checkpoints/
  metrics/
    train_log.csv
    robust_accuracy.csv
    defense_summary.json
  figures/
    clean_vs_robust_accuracy.png
    robustness_under_epsilons.png
    before_after_defense_samples.png

input_preprocessing/
  metrics/
    preprocessing_metrics.csv
  figures/
    preprocessing_comparison_grid.png
    preprocessing_accuracy_bar.png
```

报告中可展示：

- 防御前后在 FGSM / PGD 下的准确率表。
- clean accuracy 与 robust accuracy 的权衡。
- 防御成功案例。
- 防御失败案例。

## 7. 阶段四：可解释性分析

### 7.1 实验目的

从模型关注区域角度解释攻击为什么有效，以及防御是否让模型关注区域更合理。

### 7.2 推荐方法

使用 Grad-CAM。

### 7.3 展示设计

选择 5-10 张代表性图片，生成：

```text
原图 Grad-CAM
对抗样本 Grad-CAM
防御模型 Grad-CAM
```

保存到 `results/04_explainability/`：

```text
gradcam_clean_vs_attack.png
gradcam_before_after_defense.png
gradcam_case_study_01.png
gradcam_case_study_02.png
```

报告中可写：

- 攻击前模型关注交通标志主体区域。
- 攻击后注意力可能扩散或偏移。
- 对抗训练后模型关注区域更稳定。

## 8. 阶段五：演示系统

### 8.1 Demo 目标

构建一个可交互系统，方便答辩时展示攻击和防御过程。

推荐使用 Streamlit。

### 8.2 功能模块

必须功能：

- 上传交通标志图片。
- 原始模型预测。
- 选择攻击方法：FGSM / PGD。
- 调节 epsilon。
- 展示对抗样本。
- 展示攻击前后预测结果和置信度。
- 选择防御模型或预处理防御。
- 展示防御后预测结果。

增强功能：

- 显示扰动热力图。
- 显示 Top-5 类别置信度柱状图。
- 一键导出当前案例图。

### 8.3 保存结果

保存到 `results/05_demo/`：

- `demo_screenshot_home.png`
- `demo_screenshot_attack.png`
- `demo_screenshot_defense.png`
- `demo_case_export_01.png`
- `demo_recording.mp4`，可选

## 9. 实验编号建议

每次实验用固定编号，方便报告引用：

| 编号 | 实验名称 | 目的 |
|---|---|---|
| E00 | 数据集检查 | 统计类别、查看样本质量 |
| E01 | 基础 CNN | 建立简单基线 |
| E02 | ResNet18 识别 | 获得主分类模型 |
| E03 | FGSM 攻击 | 验证单步扰动攻击 |
| E04 | PGD 攻击 | 验证迭代攻击 |
| E05 | 输入预处理防御 | 测试简单防御策略 |
| E06 | 对抗训练 | 提升鲁棒性 |
| E07 | Grad-CAM 分析 | 可解释性展示 |
| E08 | Demo 系统 | 答辩展示 |

## 10. 报告结构建议

1. 项目背景与任务定义。
2. 交通标志识别模型原理。
3. 对抗样本攻击原理。
4. 鲁棒防御方法设计。
5. 数据集与实验设置。
6. 基础识别实验结果。
7. 对抗攻击实验结果。
8. 防御实验结果。
9. 可视化与 Demo 展示。
10. 失败案例与原因分析。
11. 小组分工与总结。

## 11. PPT 展示顺序

建议 20 分钟汇报结构：

| 时间 | 内容 |
|---:|---|
| 2 min | 背景和问题：交通标志识别为什么需要安全性 |
| 3 min | 基础识别模型和数据集 |
| 4 min | FGSM / PGD 对抗攻击原理 |
| 4 min | 攻击实验结果和可视化 |
| 4 min | 防御方法和鲁棒性对比 |
| 2 min | Demo 演示 |
| 1 min | 总结和分工 |

## 12. 最小可行版本

如果时间紧，优先完成：

1. ResNet18 训练。
2. FGSM 攻击。
3. PGD 攻击。
4. 对抗训练防御。
5. 攻击前后图片可视化。
6. 准确率对比表。
7. 简单 Streamlit Demo。

## 13. 加分扩展

时间充足时增加：

- Grad-CAM 可解释性分析。
- MobileNetV3 轻量模型对比。
- 不同模型鲁棒性对比。
- 黑盒迁移攻击。
- 防御方法组合实验。
- 物理攻击模拟，例如贴纸遮挡、局部噪声块。

