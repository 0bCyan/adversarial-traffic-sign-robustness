# 报告图表索引

## 图像素材

| 编号 | 文件 | 建议章节 | 用途 |
|---|---|---|---|
| 图 1 | `fig_01_class_distribution.png` | 数据集与预处理 | 展示 GTSRB 类别分布 |
| 图 2 | `fig_02_image_size_distribution.png` | 数据集与预处理 | 展示原始图片尺寸分布 |
| 图 3 | `fig_03_train_samples.png` | 数据集与预处理 | 展示训练集样例 |
| 图 4 | `fig_04_test_samples.png` | 数据集与预处理 | 展示测试集样例 |
| 图 5 | `fig_05_baseline_loss_curve.png` | 基础识别模型 | 展示训练损失变化 |
| 图 6 | `fig_06_baseline_accuracy_curve.png` | 基础识别模型 | 展示训练/验证准确率变化 |
| 图 7 | `fig_07_baseline_confusion_matrix.png` | 基础识别结果 | 展示分类混淆情况 |
| 图 8 | `fig_08_baseline_per_class_accuracy.png` | 基础识别结果 | 展示每类准确率 |
| 图 9 | `fig_09_baseline_correct_samples.png` | 基础识别结果 | 展示正确分类样例 |
| 图 10 | `fig_10_baseline_wrong_samples.png` | 错误案例分析 | 展示错误分类样例 |
| 图 11 | `fig_11_attack_accuracy_curve.png` | 对抗攻击结果 | 展示 epsilon 与准确率关系 |
| 图 12 | `fig_12_attack_success_curve.png` | 对抗攻击结果 | 展示攻击成功率变化 |
| 图 13 | `fig_13_attack_confidence_drop.png` | 对抗攻击结果 | 展示置信度下降 |
| 图 14 | `fig_14_fgsm_eps003_triplets.png` | FGSM 攻击展示 | 展示原图/扰动/对抗样本 |
| 图 15 | `fig_15_pgd_eps003_triplets.png` | PGD 攻击展示 | 展示原图/扰动/对抗样本 |
| 图 16 | `fig_16_fgsm_input_defense_curve.png` | 输入防御结果 | 展示 FGSM 下防御效果 |
| 图 17 | `fig_17_pgd_input_defense_curve.png` | 输入防御结果 | 展示 PGD 下防御效果 |
| 图 18 | `fig_18_input_defense_accuracy_bar.png` | 输入防御对比 | 展示 epsilon=0.03 防御对比 |
| 图 19 | `fig_19_input_defense_examples.png` | 防御案例分析 | 展示防御恢复案例 |
| 图 20 | `fig_20_adv_training_loss_curve.png` | 对抗训练 | 展示对抗训练损失变化 |
| 图 21 | `fig_21_adv_training_accuracy_curve.png` | 对抗训练 | 展示对抗训练准确率变化 |
| 图 22 | `fig_22_adv_training_robust_curve.png` | 对抗训练 | 展示对抗训练鲁棒准确率曲线 |
| 图 23 | `fig_23_adv_training_eps003_bar.png` | 防御对比 | 展示对抗训练与基础模型对比 |
| 图 24 | `fig_24_defense_strategy_comparison.png` | 防御对比 | 展示无防御、JPEG 和对抗训练对比 |
| 图 25 | `fig_25_clean_vs_robust_tradeoff.png` | 防御权衡 | 展示 clean accuracy 与 robust accuracy 权衡 |
| 图 26 | `fig_26_adv_training_robust_examples.png` | 对抗训练案例 | 展示对抗训练模型鲁棒案例 |
| 图 27 | `fig_27_gradcam_clean_attack_defense.png` | 可解释性分析 | 展示 clean / attack / defense Grad-CAM |
| 图 28 | `fig_28_fgsm_jpeg_quality_curve.png` | JPEG 消融 | 展示 FGSM 下不同 quality 的防御效果 |
| 图 29 | `fig_29_pgd_jpeg_quality_curve.png` | JPEG 消融 | 展示 PGD 下不同 quality 的防御效果 |
| 图 30 | `fig_30_jpeg_quality_eps003_bar.png` | JPEG 消融 | 展示 epsilon=0.03 的 quality 对比 |
| 图 31 | `fig_31_jpeg_clean_accuracy_cost.png` | JPEG 消融 | 展示 JPEG 对 clean accuracy 的影响 |
| 图 32 | `fig_32_fgsm_jpeg_quality_examples.png` | JPEG 消融案例 | 展示 FGSM 下 JPEG quality 案例 |
| 图 33 | `fig_33_pgd_jpeg_quality_examples.png` | JPEG 消融案例 | 展示 PGD 下 JPEG quality 案例 |
| 图 34 | `fig_34_runtime_ms_per_image.png` | 运行效率 | 展示每张图片平均耗时 |
| 图 35 | `fig_35_runtime_images_per_second.png` | 运行效率 | 展示处理吞吐率 |

## 表格素材

| 编号 | 文件 | 建议章节 | 用途 |
|---|---|---|---|
| 表 1 | `table_01_dataset_summary.csv` | 数据集与预处理 | 类别样本数统计 |
| 表 2 | `table_02_dataset_overview.json` | 数据集与预处理 | 数据集总览 |
| 表 3 | `table_03_baseline_test_metrics.json` | 基础识别结果 | 测试集整体指标 |
| 表 4 | `table_04_baseline_per_class_metrics.csv` | 基础识别结果 | 每类准确率 |
| 表 5 | `table_05_baseline_train_log.csv` | 基础识别训练 | 每 epoch 训练日志 |
| 表 6 | `table_06_attack_metrics.csv` | 对抗攻击结果 | FGSM/PGD 指标 |
| 表 7 | `table_07_input_defense_metrics.csv` | 输入防御结果 | 三种输入预处理防御指标 |
| 表 8 | `table_08_adv_training_train_log.csv` | 对抗训练 | 对抗训练日志 |
| 表 9 | `table_09_adv_training_robust_metrics.csv` | 对抗训练 | clean / FGSM / PGD 鲁棒指标 |
| 表 10 | `table_10_defense_strategy_comparison.csv` | 防御对比 | 无防御、JPEG 和对抗训练对比 |
| 表 11 | `table_11_gradcam_cases.csv` | 可解释性分析 | Grad-CAM 案例预测记录 |
| 表 12 | `table_12_jpeg_quality_metrics.csv` | JPEG 消融 | 不同 JPEG quality 防御指标 |
| 表 13 | `table_13_runtime_summary.csv` | 运行效率 | 推理、攻击和防御耗时统计 |

## PPT 建议精选素材

答辩 PPT 不需要放所有图，建议精选：

1. `fig_03_train_samples.png`
2. `fig_06_baseline_accuracy_curve.png`
3. `fig_07_baseline_confusion_matrix.png`
4. `fig_14_fgsm_eps003_triplets.png`
5. `fig_15_pgd_eps003_triplets.png`
6. `fig_11_attack_accuracy_curve.png`
7. `fig_18_input_defense_accuracy_bar.png`
8. `fig_19_input_defense_examples.png`
9. `fig_27_gradcam_clean_attack_defense.png`
10. `fig_30_jpeg_quality_eps003_bar.png`
11. `fig_34_runtime_ms_per_image.png`

