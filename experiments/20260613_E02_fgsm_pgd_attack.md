# E02 FGSM / PGD 对抗攻击实验

## 1. 基本信息

- 实验编号：E02
- 实验名称：FGSM / PGD 对抗攻击
- 日期：2026-06-13
- 配置文件：`configs/attack_fgsm_pgd.yaml`
- 输出目录：`results/02_attack/fgsm_pgd/`

## 2. 实验目的

验证基础 ResNet18 模型在对抗扰动下的脆弱性，并比较 FGSM 和 PGD 攻击在不同 epsilon 下的攻击效果。

## 3. 实验配置

- 目标模型：`results/01_baseline/resnet18/checkpoints/best_model.pth`
- 测试样本：从测试集中分层抽样 3000 张
- FGSM epsilon：0.005、0.01、0.02、0.03、0.05
- PGD epsilon：0.005、0.01、0.02、0.03、0.05
- PGD alpha：0.005
- PGD steps：7

## 4. 运行命令

```bash
python -m src.evaluate_attacks --config configs/attack_fgsm_pgd.yaml
```

## 5. 关键结果

| Attack | Epsilon | Clean Accuracy | Adversarial Accuracy | Attack Success Rate |
|---|---:|---:|---:|---:|
| FGSM | 0.03 | 0.9793 | 0.6710 | 0.3148 |
| FGSM | 0.05 | 0.9793 | 0.5023 | 0.4871 |
| PGD | 0.03 | 0.9793 | 0.6000 | 0.3873 |
| PGD | 0.05 | 0.9793 | 0.5427 | 0.4459 |

## 6. 保存素材

- `reports/figures/fig_11_attack_accuracy_curve.png`
- `reports/figures/fig_12_attack_success_curve.png`
- `reports/figures/fig_13_attack_confidence_drop.png`
- `reports/figures/fig_14_fgsm_eps003_triplets.png`
- `reports/figures/fig_15_pgd_eps003_triplets.png`
- `reports/tables/table_06_attack_metrics.csv`

## 7. 结论

FGSM 和 PGD 都能显著降低模型准确率。PGD 在 epsilon=0.03 时将准确率降低至 60.00%，体现出迭代攻击比单步攻击更强的攻击能力。

