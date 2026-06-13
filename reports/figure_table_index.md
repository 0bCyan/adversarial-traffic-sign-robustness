# 报告图表索引

本索引记录当前正式报告可用的精选图表。完整 Word 报告由
`scripts/build_teacher_format_report.py` 和 `scripts/build_deep_word_report.py`
生成。

## 图像素材

| 范围 | 文件 | 章节用途 |
|---|---|---|
| fig_01-04 | 数据集分布、尺寸分布、训练/测试样例 | 数据集与预处理 |
| fig_05-10 | 训练曲线、混淆矩阵、per-class、正确/错误样例 | ResNet18 基础识别 |
| fig_11-15 | FGSM/PGD 攻击曲线与样例 | 对抗攻击 |
| fig_16-19 | 输入预处理防御曲线与恢复样例 | 输入防御 |
| fig_20-22 | PSNR 与真实小扰动视觉分析 | 扰动可感知性 |
| fig_23-26 | 项目流程、ResNet 结构、攻击/防御链路图 | 原理解释 |
| fig_27-32 | 随机噪声、PGD 步数、防御参数、margin 分析 | 消融实验 |
| fig_33 | Grad-CAM clean/attack/defense | 可解释性分析 |
| fig_34-39 | JPEG quality 曲线、clean 代价和样例 | 防御参数消融 |
| fig_40-41 | 单图耗时与吞吐率 | 运行效率 |
| fig_42-44 | 类别鲁棒性和 JPEG 防御失败案例 | 严格防御验证 |
| fig_45-46 | JPEG 防御下 BPDA 自适应攻击 | 自适应攻击验证 |
| fig_47-51 | FGSM 对抗训练曲线、鲁棒对比和样例 | 模型级防御 |

## 表格素材

| 范围 | 文件 | 章节用途 |
|---|---|---|
| table_01-02 | 数据集 summary / overview | 数据集统计 |
| table_03-05 | clean 测试指标、per-class、训练日志 | 基础识别 |
| table_06-08 | 攻击指标、输入防御指标、扰动可感知性 | 攻击与防御主实验 |
| table_09-12 | 随机噪声、PGD 步数、防御参数、margin 指标 | 补充消融 |
| table_13-15 | Grad-CAM、JPEG quality、runtime | PR #6 补充实验 |
| table_16-17 | per-class robustness、defense failure cases | 类别差异与失败案例 |
| table_18 | adaptive_jpeg_bpda_metrics.csv | 自适应攻击验证 |
| table_19-20 | adversarial training train log / robust metrics | 模型级防御 |

## PPT 建议精选素材

答辩 PPT 建议精选以下图片，覆盖“识别 -> 攻击 -> 小扰动 -> 防御 -> 局限 -> 模型级防御”：

1. `fig_03_train_samples.png`
2. `fig_06_baseline_accuracy_curve.png`
3. `fig_11_attack_accuracy_curve.png`
4. `fig_21_fgsm_perceptual_grid.png`
5. `fig_27_random_vs_adversarial_accuracy.png`
6. `fig_18_input_defense_accuracy_bar.png`
7. `fig_33_gradcam_clean_attack_defense.png`
8. `fig_44_pgd_jpeg_failure_examples.png`
9. `fig_45_adaptive_jpeg_bpda_accuracy.png`
10. `fig_50_adv_training_eps003_bar.png`
