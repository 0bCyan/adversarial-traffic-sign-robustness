# 结果保存规范

本项目是课程报告型项目，实验过程本身就是展示材料。所有实验都要尽量保存可视化图片、指标表、日志和典型案例。

## 1. 总原则

每次实验都保存四类内容：

1. 配置：本次实验用了什么参数。
2. 指标：结果好不好。
3. 图片：报告和 PPT 能不能展示。
4. 日志：出了问题能不能复现和解释。

不要只保存最终准确率。报告中最有说服力的是过程图、对比图和失败案例。

## 2. 实验目录命名

推荐格式：

```text
results/<阶段编号>_<阶段名称>/<方法名>/<日期时间>_<简短说明>/
```

示例：

```text
results/01_baseline/resnet18/20260613_1530_seed42/
results/02_attack/fgsm_pgd/20260614_1015_eps_sweep/
results/03_defense/adversarial_training/20260615_2200_fgsm_eps002/
```

如果只保留一份稳定结果，也可以使用：

```text
results/01_baseline/resnet18/
```

## 3. 每次实验必备文件

每个实验目录建议包含：

```text
config.yaml
run_info.json
metrics/
figures/
samples/
logs/
checkpoints/
```

### 3.1 config.yaml

保存本次实验配置，例如：

- 数据集路径。
- 模型名称。
- 输入尺寸。
- batch size。
- epoch。
- learning rate。
- attack epsilon。
- 随机种子。

### 3.2 run_info.json

保存运行环境，例如：

- 运行时间。
- git commit hash。
- Python 版本。
- PyTorch 版本。
- CUDA 是否可用。
- GPU 名称。
- 运行命令。

### 3.3 metrics/

保存 CSV 或 JSON：

- `train_log.csv`
- `validation_log.csv`
- `test_metrics.json`
- `per_class_metrics.csv`
- `attack_metrics.csv`
- `defense_metrics.csv`

### 3.4 figures/

保存可直接放进报告的图：

- 曲线图。
- 混淆矩阵。
- 柱状图。
- 热力图。
- 对比图。

### 3.5 samples/

保存样例图片：

- 原始图片。
- 扰动噪声图。
- 对抗样本。
- 防御后图片。
- 攻击成功案例。
- 攻击失败案例。
- 防御成功案例。
- 防御失败案例。

### 3.6 logs/

保存：

- `train.log`
- `test.log`
- `error.log`

### 3.7 checkpoints/

保存：

- `best_model.pth`
- `last_model.pth`

模型文件可能较大，默认不提交到 git。报告中说明模型文件位置或下载方式即可。

## 4. 图片保存要求

### 4.1 分辨率

报告图片建议：

- 单图宽度不少于 1200 px。
- 宫格图宽度不少于 1600 px。
- DPI 建议 150 或 300。

### 4.2 文件名

文件名必须表达清楚内容：

```text
loss_curve_resnet18.png
confusion_matrix_resnet18.png
fgsm_eps_003_success_cases.png
pgd_eps_003_noise_map.png
defense_before_after_eps_003.png
gradcam_clean_attack_defense.png
```

不要使用：

```text
1.png
test.png
new.png
final_final.png
```

### 4.3 对抗样本图格式

推荐每个案例保存为三联图或四联图：

```text
原图 | 扰动图 | 对抗样本 | 防御后样本
```

图下注明：

- true label
- clean prediction
- adversarial prediction
- defended prediction
- epsilon
- confidence

## 5. 报告素材导出

`results/` 保存完整实验结果。  
`reports/figures/` 只保存精选后要放入报告和 PPT 的图片。  
`reports/tables/` 只保存精选表格。

建议每完成一个阶段，从 `results/` 复制精选素材到 `reports/`：

```text
reports/figures/
  fig_01_dataset_samples.png
  fig_02_training_curve.png
  fig_03_confusion_matrix.png
  fig_04_fgsm_triplet.png
  fig_05_attack_accuracy_curve.png
  fig_06_defense_comparison.png
  fig_07_gradcam_case.png

reports/tables/
  table_01_baseline_metrics.csv
  table_02_attack_metrics.csv
  table_03_defense_metrics.csv
```

## 6. 建议保存的关键图表清单

### 数据集部分

- 类别分布图。
- 样例图片宫格。
- 图片尺寸分布。

### 基础识别部分

- 模型结构示意图。
- loss 曲线。
- accuracy 曲线。
- 混淆矩阵。
- 每类准确率柱状图。
- 错误分类样例图。

### 攻击部分

- FGSM 原理示意图。
- PGD 迭代流程图。
- 原图 / 扰动 / 对抗样本三联图。
- epsilon 与准确率曲线。
- epsilon 与攻击成功率曲线。
- Top-5 置信度变化柱状图。

### 防御部分

- 防御流程图。
- 防御前后准确率对比表。
- 不同 epsilon 下鲁棒准确率曲线。
- 防御成功案例。
- 防御失败案例。

### 可解释性部分

- clean / attack / defense 的 Grad-CAM 对比图。

### Demo 部分

- 首页截图。
- 攻击页面截图。
- 防御页面截图。
- 录屏或导出的案例图。

## 7. 实验记录要求

每次实验都在 `experiments/` 新建一份记录，使用模板：

```text
experiments/YYYYMMDD_E编号_实验名称.md
```

示例：

```text
experiments/20260613_E00_dataset_check.md
experiments/20260614_E03_fgsm_attack.md
```

记录中必须包含：

- 实验目的。
- 实验配置。
- 运行命令。
- 输出路径。
- 关键结果。
- 保存的图片和表格。
- 遇到的问题。
- 下一步计划。

