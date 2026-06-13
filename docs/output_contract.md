# 代码输出约定

后续所有训练、攻击、防御和可视化脚本都应遵守本文档，保证结果可复现、可比较、可直接用于报告。

## 1. 脚本参数

每个主脚本至少支持：

```bash
--config configs/xxx.yaml
--output-dir results/xxx
--seed 42
```

如果 `--output-dir` 未提供，则从配置文件中的 `outputs.result_dir` 读取。

## 2. 输出目录结构

每次运行创建：

```text
output_dir/
  config.yaml
  run_info.json
  logs/
  metrics/
  figures/
  samples/
  checkpoints/
```

## 3. 指标文件格式

训练日志 `train_log.csv`：

```text
epoch,train_loss,train_acc,val_loss,val_acc,lr
```

测试指标 `test_metrics.json`：

```json
{
  "accuracy": 0.0,
  "precision_macro": 0.0,
  "recall_macro": 0.0,
  "f1_macro": 0.0
}
```

攻击指标 `attack_metrics.csv`：

```text
attack,epsilon,clean_accuracy,adversarial_accuracy,attack_success_rate,mean_confidence_drop
```

防御指标 `defense_metrics.csv`：

```text
model,defense,attack,epsilon,accuracy,robust_accuracy
```

## 4. 图片命名

训练：

```text
figures/loss_curve.png
figures/accuracy_curve.png
figures/confusion_matrix.png
samples/wrong_samples_grid.png
```

攻击：

```text
figures/epsilon_accuracy_curve.png
figures/attack_success_rate_curve.png
samples/fgsm_eps_0.03_triplets.png
samples/pgd_eps_0.03_triplets.png
samples/noise_map_examples.png
```

防御：

```text
figures/robust_accuracy_comparison.png
figures/clean_vs_robust_accuracy.png
samples/before_after_defense.png
```

可解释性：

```text
figures/gradcam_clean_attack_defense.png
```

## 5. 报告导出

关键图片和表格要复制到：

```text
reports/figures/
reports/tables/
```

报告中只引用 `reports/` 中的精选素材，完整过程保留在 `results/`。

