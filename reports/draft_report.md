# 基于对抗攻击与输入防御的交通标志识别系统设计与实现

## 摘要

交通标志识别是自动驾驶、辅助驾驶和智能交通系统中的重要视觉感知任务。深度学习模型虽然能够在标准测试集上取得较高准确率，但在面对对抗扰动时可能出现预测错误，从而带来潜在安全风险。本项目以 GTSRB 交通标志识别数据集为研究对象，首先构建基于 ResNet18 的基础识别模型，然后使用 FGSM 和 PGD 方法生成对抗样本，分析模型在不同扰动强度下的鲁棒性变化，最后使用 Gaussian Blur、Median Filter 和 JPEG Compression 三种输入预处理方法进行防御实验。实验结果表明，ResNet18 在正常测试集上达到 97.81% 的准确率，但在 PGD epsilon=0.03 攻击下准确率下降至 60.00%。输入预处理防御可以部分恢复模型性能，其中 JPEG Compression 在 PGD epsilon=0.03 下将准确率提升至 79.43%。本项目验证了交通标志识别模型在对抗扰动下的脆弱性，并初步探索了轻量级输入防御方法的有效性。

关键词：交通标志识别；GTSRB；ResNet18；对抗样本；FGSM；PGD；输入预处理防御

## 1. 实验背景

### 1.1 研究背景

交通标志识别是智能交通系统中的基础任务之一。在自动驾驶场景中，车辆需要准确识别限速、禁止通行、转向、让行等交通标志，从而辅助决策系统完成速度控制、路径规划和风险规避。近年来，卷积神经网络在交通标志识别任务中取得了较高准确率，使得基于深度学习的识别方法成为主流方案。

然而，深度神经网络存在对抗脆弱性。所谓对抗样本，是指在原始输入上加入人眼难以察觉的细微扰动，却能导致模型产生错误预测的样本。对于交通标志识别任务而言，如果模型在轻微扰动、压缩噪声、光照变化或恶意干扰下识别错误，可能会影响智能交通系统的安全性。因此，本项目不只关注交通标志识别准确率，还进一步分析模型在对抗攻击下的鲁棒性。

### 1.2 实验目标

本项目的目标包括：

1. 构建交通标志识别基础模型，并评估其正常识别性能。
2. 实现 FGSM 和 PGD 对抗攻击方法，生成对抗样本。
3. 分析不同扰动强度 epsilon 对模型准确率和攻击成功率的影响。
4. 使用输入预处理方法进行防御实验，比较不同防御策略的效果。
5. 保存完整实验指标、中间过程图片和可视化结果，为报告和答辩展示提供材料。

### 1.3 技术路线

本项目整体流程如下：

```text
GTSRB 数据集
  -> 数据检查与可视化
  -> ResNet18 基础识别模型训练
  -> FGSM / PGD 对抗样本生成
  -> 对抗攻击效果分析
  -> Gaussian Blur / Median Filter / JPEG Compression 输入防御
  -> 防御效果对比与总结
```

## 2. 数据集与预处理

### 2.1 GTSRB 数据集

本项目使用 GTSRB 交通标志识别数据集。该数据集包含 43 类交通标志，图像来自真实道路场景，存在不同光照、角度、模糊和尺寸变化，适合用于交通标志分类实验。

数据统计如下：

| 项目 | 数值 |
|---|---:|
| 类别数 | 43 |
| 训练集图片数 | 26640 |
| 测试集图片数 | 12630 |
| 总图片数 | 39270 |
| 最少训练样本类别数 | 150 |
| 最多训练样本类别数 | 1500 |

报告中可插入：

- 图 1：`reports/figures/fig_01_class_distribution.png`
- 图 3：`reports/figures/fig_03_train_samples.png`
- 图 4：`reports/figures/fig_04_test_samples.png`

### 2.2 数据预处理

实验中将输入图像统一缩放为 `64 x 64`。训练阶段使用随机旋转、颜色扰动和随机仿射等数据增强方法，以提高模型对视角和光照变化的适应能力。图像随后转换为 Tensor，并使用固定均值和标准差进行归一化。

训练集进一步划分为训练子集和验证子集，其中验证比例为 15%。测试集用于最终性能评估和攻击防御实验。

## 3. 基础交通标志识别模型

### 3.1 模型选择

本项目选用 ResNet18 作为基础分类模型。ResNet 的核心思想是引入残差连接，使网络能够学习输入与输出之间的残差映射，从而缓解深层网络训练中的梯度消失和退化问题。

由于 GTSRB 图像尺寸较小，本项目对 ResNet18 的前几层结构进行了适配：

- 将第一层卷积调整为 `3 x 3` 卷积。
- 去除初始最大池化层，以保留更多空间细节。
- 将最后的全连接层输出类别数改为 43。

### 3.2 训练设置

| 参数 | 设置 |
|---|---|
| 模型 | ResNet18 |
| 输入尺寸 | 64 x 64 |
| 类别数 | 43 |
| 优化器 | AdamW |
| 学习率 | 0.001 |
| 权重衰减 | 0.0001 |
| 学习率调度 | Cosine |
| Epoch | 30 |
| Batch Size | 256 |
| 损失函数 | CrossEntropyLoss |

### 3.3 基础识别结果

模型在测试集上的结果如下：

| 指标 | 数值 |
|---|---:|
| Test Accuracy | 0.9781 |
| Precision Macro | 0.9678 |
| Recall Macro | 0.9591 |
| F1 Macro | 0.9610 |
| 参数量 | 11190891 |
| 最优 Epoch | 13 |

报告中可插入：

- 图 5：`reports/figures/fig_05_baseline_loss_curve.png`
- 图 6：`reports/figures/fig_06_baseline_accuracy_curve.png`
- 图 7：`reports/figures/fig_07_baseline_confusion_matrix.png`
- 图 8：`reports/figures/fig_08_baseline_per_class_accuracy.png`
- 图 9：`reports/figures/fig_09_baseline_correct_samples.png`
- 图 10：`reports/figures/fig_10_baseline_wrong_samples.png`

从结果可以看出，ResNet18 在正常测试集上能够取得较高准确率，说明基础识别模型已经能够较好完成交通标志分类任务。同时，错误样例显示部分交通标志在低光照、模糊或类别相似情况下仍容易被误分类。

## 4. 对抗攻击方法

### 4.1 对抗样本定义

对抗样本是在原始样本上加入微小扰动后得到的输入。设原始图像为 `x`，真实标签为 `y`，分类模型为 `f`，攻击目标是在扰动大小受限的情况下构造 `x_adv`，使模型预测错误：

```text
x_adv = x + delta
||delta|| <= epsilon
f(x_adv) != y
```

其中 epsilon 用于控制扰动强度。epsilon 越大，攻击能力通常越强，但图像变化也可能越明显。

### 4.2 FGSM 攻击

FGSM，即 Fast Gradient Sign Method，是一种单步梯度攻击方法。其核心思想是沿着损失函数对输入图像梯度的符号方向添加扰动，使模型损失快速增大。

FGSM 公式如下：

```text
x_adv = x + epsilon * sign(grad_x J(theta, x, y))
```

FGSM 的优点是计算速度快，适合快速验证模型脆弱性。

### 4.3 PGD 攻击

PGD，即 Projected Gradient Descent，是一种迭代式对抗攻击方法。PGD 每一步沿梯度方向更新对抗样本，并将扰动限制在 epsilon 范围内。相比 FGSM，PGD 通常攻击能力更强。

PGD 迭代过程可表示为：

```text
x_adv^(t+1) = Projection(x_adv^t + alpha * sign(grad_x J(theta, x_adv^t, y)))
```

本项目中 PGD 参数设置为：

- alpha = 0.005
- steps = 7

### 4.4 攻击实验设置

对抗攻击实验使用从测试集中分层抽样得到的 3000 张图片，以保证不同类别均有代表性。实验比较 FGSM 和 PGD 在不同 epsilon 下的攻击效果。

epsilon 设置为：

```text
0.005, 0.01, 0.02, 0.03, 0.05
```

评价指标包括：

- Clean Accuracy
- Adversarial Accuracy
- Attack Success Rate
- Mean Confidence Drop

## 5. 对抗攻击实验结果

### 5.1 FGSM 攻击结果

| Epsilon | Clean Accuracy | Adversarial Accuracy | Attack Success Rate |
|---:|---:|---:|---:|
| 0.005 | 0.9793 | 0.9527 | 0.0272 |
| 0.010 | 0.9793 | 0.9013 | 0.0796 |
| 0.020 | 0.9793 | 0.7827 | 0.2008 |
| 0.030 | 0.9793 | 0.6710 | 0.3148 |
| 0.050 | 0.9793 | 0.5023 | 0.4871 |

### 5.2 PGD 攻击结果

| Epsilon | Clean Accuracy | Adversarial Accuracy | Attack Success Rate |
|---:|---:|---:|---:|
| 0.005 | 0.9793 | 0.9467 | 0.0334 |
| 0.010 | 0.9793 | 0.8683 | 0.1133 |
| 0.020 | 0.9793 | 0.6897 | 0.2958 |
| 0.030 | 0.9793 | 0.6000 | 0.3873 |
| 0.050 | 0.9793 | 0.5427 | 0.4459 |

报告中可插入：

- 图 11：`reports/figures/fig_11_attack_accuracy_curve.png`
- 图 12：`reports/figures/fig_12_attack_success_curve.png`
- 图 13：`reports/figures/fig_13_attack_confidence_drop.png`
- 图 14：`reports/figures/fig_14_fgsm_eps003_triplets.png`
- 图 15：`reports/figures/fig_15_pgd_eps003_triplets.png`

### 5.3 结果分析

实验结果表明，随着 epsilon 增大，模型在对抗样本上的准确率明显下降。FGSM 在 epsilon=0.03 时将模型准确率从 97.93% 降至 67.10%，PGD 在相同 epsilon 下进一步降至 60.00%。这说明即使模型在正常测试集上具有较高准确率，也可能受到较小输入扰动的显著影响。

PGD 在多数扰动强度下比 FGSM 更强，原因在于 PGD 通过多次迭代寻找更有效的扰动方向，而 FGSM 只进行单步更新。对抗样本三联图显示，扰动在视觉上并不总是非常明显，但模型预测类别和置信度已经发生显著变化。

## 6. 输入预处理防御

### 6.1 防御思路

输入预处理防御的基本思想是在图像送入模型前进行一定变换，以削弱对抗扰动的影响。本项目选取三种轻量方法：

1. Gaussian Blur：使用高斯模糊平滑局部噪声。
2. Median Filter：使用中值滤波抑制局部异常像素。
3. JPEG Compression：通过图像压缩去除部分高频扰动。

这些方法不需要重新训练模型，部署简单，适合作为轻量防御基线。

### 6.2 防御实验设置

防御实验选择较强攻击场景：

- FGSM epsilon=0.03 和 0.05
- PGD epsilon=0.03 和 0.05

测试样本与攻击实验保持一致，均为同一批 3000 张分层抽样测试图片。

### 6.3 防御实验结果

epsilon=0.03 下的结果：

| Attack | Before Defense | Gaussian Blur | Median Filter | JPEG Compression |
|---|---:|---:|---:|---:|
| FGSM | 0.6710 | 0.7180 | 0.7107 | 0.7937 |
| PGD | 0.6030 | 0.6733 | 0.6780 | 0.7943 |

epsilon=0.05 下的结果：

| Attack | Before Defense | Gaussian Blur | Median Filter | JPEG Compression |
|---|---:|---:|---:|---:|
| FGSM | 0.5023 | 0.5577 | 0.5477 | 0.6270 |
| PGD | 0.5447 | 0.6133 | 0.6153 | 0.7173 |

报告中可插入：

- 图 16：`reports/figures/fig_16_fgsm_input_defense_curve.png`
- 图 17：`reports/figures/fig_17_pgd_input_defense_curve.png`
- 图 18：`reports/figures/fig_18_input_defense_accuracy_bar.png`
- 图 19：`reports/figures/fig_19_input_defense_examples.png`

### 6.4 防御结果分析

三种输入预处理方法均能在一定程度上恢复模型准确率，其中 JPEG Compression 效果最明显。在 PGD epsilon=0.03 场景下，模型准确率从 60.30% 提升至 79.43%；在 PGD epsilon=0.05 场景下，从 54.47% 提升至 71.73%。

JPEG Compression 效果较好的原因可能是对抗扰动常包含较多高频细节，而 JPEG 压缩会丢弃部分高频信息，从而削弱扰动影响。Gaussian Blur 和 Median Filter 也能抑制局部噪声，但可能同时损失交通标志边缘和纹理细节，因此提升幅度较小。

不过，输入预处理防御并不能完全恢复到正常测试集准确率。这说明简单输入变换只能作为基础防御方法，面对更强攻击或自适应攻击时仍存在局限。

## 7. 程序实现说明

### 7.1 项目结构

核心目录如下：

```text
configs/        实验配置文件
src/            源代码
results/        完整实验结果
reports/        报告素材、结果汇总和报告草稿
docs/           实验计划和规范文档
```

### 7.2 核心代码文件

| 文件 | 功能 |
|---|---|
| `src/data/check_gtsrb.py` | 数据集检查、统计表和样例图生成 |
| `src/models/classifiers.py` | SimpleCNN 和 ResNet18 模型定义 |
| `src/train_classifier.py` | 基础分类模型训练与评估 |
| `src/attacks/methods.py` | FGSM 和 PGD 攻击方法 |
| `src/evaluate_attacks.py` | 对抗攻击评估与可视化 |
| `src/evaluate_input_defense.py` | 输入预处理防御评估 |

### 7.3 运行命令

数据集检查：

```bash
python -m src.data.check_gtsrb --config configs/dataset_check.yaml
```

基础识别训练：

```bash
python -m src.train_classifier --config configs/baseline_resnet18.yaml
```

对抗攻击评估：

```bash
python -m src.evaluate_attacks --config configs/attack_fgsm_pgd.yaml
```

输入预处理防御评估：

```bash
python -m src.evaluate_input_defense --config configs/defense_input_preprocessing.yaml
```

## 8. 实验过程中遇到的问题

### 8.1 PGD 攻击计算耗时较长

PGD 是迭代攻击方法，每个 batch 需要多次前向和反向传播，因此在全测试集、多 epsilon 设置下耗时较长。为保证实验可控，本项目采用分层抽样的 3000 张测试图进行攻击与防御评估，并保存抽样索引，确保不同实验在同一批样本上比较。

### 8.2 可视化图片排版问题

初始样例宫格图中类别名称较长，出现文字拥挤。后续调整了单元格宽度和标签显示方式，使样例图更适合报告展示。

### 8.3 输入防御存在性能上限

输入预处理可以提升对抗样本下的准确率，但也可能损失原图中的边缘和纹理信息。实验显示其效果有限，无法完全恢复到 clean accuracy。

## 9. 创新点

本项目的创新性主要体现在：

1. 没有停留在普通交通标志分类，而是进一步分析模型安全性和鲁棒性。
2. 构建了“基础识别 - 对抗攻击 - 防御评估”的完整实验闭环。
3. 同时比较 FGSM 和 PGD 两种攻击方法，分析不同 epsilon 下的性能变化。
4. 保存原图、扰动图、对抗样本和防御恢复案例，增强实验展示性。
5. 使用同一批分层抽样测试样本比较攻击和防御结果，提高实验公平性。

## 10. 不足与改进方向

当前项目仍存在以下不足：

1. 输入预处理防御属于轻量防御，面对自适应攻击时可能效果下降。
2. 对抗训练防御代码已创建，但尚未完整跑完并验证，因此未纳入正式结果。
3. 尚未加入 Grad-CAM 可解释性分析，无法从注意力区域角度解释攻击效果。
4. 尚未完成交互式 Demo，后续可使用 Streamlit 实现上传图片、生成攻击和防御结果的可视化系统。

后续可继续扩展：

- 完整运行 FGSM 对抗训练。
- 加入 PGD 对抗训练或 TRADES 等鲁棒优化方法。
- 使用 Grad-CAM 分析攻击前后模型关注区域变化。
- 设计黑盒迁移攻击实验。
- 构建 Streamlit 演示系统。

## 11. 小组分工

以下表格为模板，需根据实际成员填写。

| 成员 | 工作内容 | 贡献占比 |
|---|---|---:|
| 成员 A | 数据集处理、数据可视化、结果整理 | 20% |
| 成员 B | ResNet18 模型训练与基础识别实验 | 20% |
| 成员 C | FGSM / PGD 攻击实现与实验 | 20% |
| 成员 D | 输入预处理防御实现与对比分析 | 20% |
| 成员 E | 实验报告、PPT、答辩 Demo 整理 | 20% |

## 12. 总结

本项目完成了基于 ResNet18 的交通标志识别模型，并在 GTSRB 测试集上取得 97.81% 的准确率。在此基础上，项目实现了 FGSM 和 PGD 两种对抗攻击方法，实验发现模型在对抗扰动下准确率显著下降，说明深度学习交通标志识别模型存在明显鲁棒性问题。进一步地，项目使用 Gaussian Blur、Median Filter 和 JPEG Compression 三种输入预处理方法进行防御实验，结果表明 JPEG Compression 可以较明显提升对抗样本下的识别准确率。

整体来看，本项目从普通图像分类任务扩展到模型安全性分析，形成了识别、攻击、防御和可视化的完整实验链路，具有较好的技术深度和展示效果。

