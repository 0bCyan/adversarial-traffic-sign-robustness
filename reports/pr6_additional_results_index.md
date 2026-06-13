# PR #6 补充实验结果索引

本索引记录从 PR #6 移植并实际运行得到的补充实验产物。所有实验均基于已有
`results/01_baseline/resnet18/checkpoints/best_model.pth` 推理评估，没有重新训练模型。

## 1. Grad-CAM 可解释性分析

- 配置：`configs/explainability_gradcam.yaml`
- 结果目录：`results/04_explainability/gradcam/`
- 报告图片：`reports/figures/fig_33_gradcam_clean_attack_defense.png`
- 报告表格：`reports/tables/table_13_gradcam_cases.csv`

用途：展示 clean、PGD attack、JPEG defense 三种输入下 ResNet18 最后一层卷积特征的关注区域变化。当前 6 个案例均为攻击后误分类、JPEG 防御后恢复正确分类的样本。

## 2. JPEG Quality 消融实验

- 配置：`configs/defense_jpeg_ablation.yaml`
- 结果目录：`results/03_defense/jpeg_quality_ablation/`
- 报告图片：
  - `reports/figures/fig_34_fgsm_jpeg_quality_curve.png`
  - `reports/figures/fig_35_pgd_jpeg_quality_curve.png`
  - `reports/figures/fig_36_jpeg_quality_eps003_bar.png`
  - `reports/figures/fig_37_jpeg_clean_accuracy_cost.png`
  - `reports/figures/fig_38_fgsm_jpeg_quality_examples.png`
  - `reports/figures/fig_39_pgd_jpeg_quality_examples.png`
- 报告表格：`reports/tables/table_14_jpeg_quality_metrics.csv`

关键结论：

- clean baseline accuracy 为 0.9793。
- clean 输入经 JPEG Q50/Q75/Q90 后准确率分别为 0.9687、0.9723、0.9767。
- FGSM epsilon=0.03 下，攻击后准确率为 0.6710；JPEG Q50/Q75/Q90 防御后分别为 0.8150、0.7937、0.7620。
- PGD epsilon=0.03 下，攻击后准确率为 0.6030；JPEG Q50/Q75/Q90 防御后分别为 0.8243、0.7943、0.7470。

用途：说明 JPEG 压缩质量参数存在 clean accuracy 与 robust accuracy 的权衡。Q50 鲁棒恢复最强，但 clean accuracy 损失也更大；Q75 是更均衡的展示配置。

## 3. 运行效率统计

- 配置：`configs/runtime_benchmark.yaml`
- 结果目录：`results/06_runtime/`
- 报告图片：
  - `reports/figures/fig_40_runtime_ms_per_image.png`
  - `reports/figures/fig_41_runtime_images_per_second.png`
- 报告表格：`reports/tables/table_15_runtime_summary.csv`

关键结论：

- clean inference 平均 0.287 ms/image。
- FGSM generation 平均 0.938 ms/image。
- PGD generation 平均 6.601 ms/image。
- JPEG defense plus inference 平均约 0.499 ms/image。

用途：说明 PGD 的计算代价明显高于 FGSM，也说明 JPEG 输入预处理作为防御展示模块具有较低额外开销。

## 4. 全测试集关键配置验证

- 攻击配置：`configs/attack_fgsm_pgd_full_test.yaml`
- 攻击结果目录：`results/02_attack/fgsm_pgd_full_test/`
- 防御配置：`configs/defense_input_preprocessing_full_test.yaml`
- 防御结果目录：`results/03_defense/input_preprocessing_full_test/`

关键结论：

- 全测试集共 12630 张图，clean accuracy 为 0.9781。
- FGSM epsilon=0.03 后准确率为 0.6667，攻击成功率为 0.3184。
- PGD epsilon=0.03 后准确率为 0.5925，攻击成功率为 0.3943。
- JPEG Q75 防御后，FGSM epsilon=0.03 准确率恢复到 0.7835，PGD epsilon=0.03 恢复到 0.7804。

用途：验证 3000 张抽样实验的趋势在完整测试集上仍成立，可在报告中作为“结果可靠性验证”或“补充验证”使用。
