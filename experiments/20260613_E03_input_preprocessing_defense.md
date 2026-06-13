# E03 输入预处理防御实验

## 1. 基本信息

- 实验编号：E03
- 实验名称：输入预处理防御
- 日期：2026-06-13
- 配置文件：`configs/defense_input_preprocessing.yaml`
- 输出目录：`results/03_defense/input_preprocessing/`

## 2. 实验目的

评估 Gaussian Blur、Median Filter 和 JPEG Compression 三种输入预处理方法对 FGSM / PGD 对抗样本的防御效果。

## 3. 实验配置

- 目标模型：基础 ResNet18
- 测试样本：与攻击实验相同的 3000 张分层抽样样本
- 攻击方法：FGSM、PGD
- Epsilon：0.03、0.05
- 防御方法：
  - Gaussian Blur
  - Median Filter
  - JPEG Compression

## 4. 运行命令

```bash
python -m src.evaluate_input_defense --config configs/defense_input_preprocessing.yaml
```

## 5. 关键结果

| Attack | Epsilon | Before Defense | Best Defense | Best Accuracy |
|---|---:|---:|---|---:|
| FGSM | 0.03 | 0.6710 | JPEG Compression | 0.7937 |
| FGSM | 0.05 | 0.5023 | JPEG Compression | 0.6270 |
| PGD | 0.03 | 0.6030 | JPEG Compression | 0.7943 |
| PGD | 0.05 | 0.5447 | JPEG Compression | 0.7173 |

## 6. 保存素材

- `reports/figures/fig_16_fgsm_input_defense_curve.png`
- `reports/figures/fig_17_pgd_input_defense_curve.png`
- `reports/figures/fig_18_input_defense_accuracy_bar.png`
- `reports/figures/fig_19_input_defense_examples.png`
- `reports/tables/table_07_input_defense_metrics.csv`

## 7. 结论

输入预处理可以部分缓解对抗扰动影响，其中 JPEG Compression 效果最好。但该方法无法完全恢复到 clean accuracy，说明轻量输入防御存在上限。

