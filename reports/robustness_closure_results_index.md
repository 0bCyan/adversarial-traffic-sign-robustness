# 鲁棒性闭环补强结果索引

本索引记录为补齐“识别 -> 攻击 -> 防御 -> 防御边界 -> 模型级防御”证据链新增的实验结果。

## 1. 类别鲁棒性与失败案例

- 配置：`configs/per_class_failure_analysis.yaml`
- 脚本：`src/evaluate_per_class_failure.py`
- 结果目录：`results/04_extended_analysis/per_class_failure_analysis/`
- 报告表格：
  - `reports/tables/table_16_per_class_robustness.csv`
  - `reports/tables/table_17_defense_failure_cases.csv`
- 报告图片：
  - `reports/figures/fig_42_per_class_attack_success.png`
  - `reports/figures/fig_43_per_class_defense_recovery.png`
  - `reports/figures/fig_44_pgd_jpeg_failure_examples.png`

用途：说明 PGD 攻击和 JPEG 防御在不同交通标志类别上的效果并不均匀，并展示 JPEG Q75 仍无法恢复的典型失败案例。

## 2. JPEG 防御下 BPDA 自适应攻击

- 配置：`configs/adaptive_jpeg_attack.yaml`
- 脚本：`src/evaluate_adaptive_jpeg_attack.py`
- 结果目录：`results/04_extended_analysis/adaptive_jpeg_bpda/`
- 报告表格：`reports/tables/table_18_adaptive_jpeg_bpda_metrics.csv`
- 报告图片：
  - `reports/figures/fig_45_adaptive_jpeg_bpda_accuracy.png`
  - `reports/figures/fig_46_adaptive_jpeg_bpda_examples.png`

关键结论：普通 PGD + JPEG Q75 后准确率为 0.7893；BPDA 自适应攻击 + JPEG Q75 后准确率为 0.7803。下降幅度不大，但证明输入预处理防御依赖威胁模型，不能被解释为完备安全方案。

## 3. FGSM 对抗训练模型级防御

- 配置：`configs/defense_adversarial_training.yaml`
- 脚本：`src/train_adversarial.py`
- 结果目录：`results/03_defense/adversarial_training/`
- 报告表格：
  - `reports/tables/table_19_adv_training_train_log.csv`
  - `reports/tables/table_20_adv_training_robust_metrics.csv`
- 报告图片：
  - `reports/figures/fig_47_adv_training_loss_curve.png`
  - `reports/figures/fig_48_adv_training_accuracy_curve.png`
  - `reports/figures/fig_49_adv_training_robust_curve.png`
  - `reports/figures/fig_50_adv_training_eps003_bar.png`
  - `reports/figures/fig_51_adv_training_robust_examples.png`

关键结论：对抗训练模型在 3000 张固定抽样测试图上 clean accuracy 为 0.9847；FGSM epsilon=0.03 accuracy 为 0.8800；PGD epsilon=0.03 accuracy 为 0.8863。相比 baseline PGD epsilon=0.03 accuracy 0.6000，模型级防御提升明显。
