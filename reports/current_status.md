# 当前项目状态

更新时间：2026-06-14

## 已完成

- 项目仓库与目录结构。
- GTSRB 数据集检查脚本。
- ResNet18 基础识别训练脚本。
- FGSM / PGD 对抗攻击评估脚本。
- 输入预处理防御评估脚本。
- Grad-CAM、JPEG quality 消融、运行效率统计。
- 类别鲁棒性、失败案例、自适应 BPDA 攻击分析。
- FGSM 对抗训练模型级防御实验。
- 报告图片与表格导出。
- 中文实验报告草稿所需的核心数据。

## 可写入正式报告的实验

1. 数据集检查与可视化。
2. ResNet18 基础交通标志识别。
3. FGSM / PGD 对抗扰动攻击。
4. Gaussian Blur / Median Filter / JPEG Compression 输入预处理防御。
5. Grad-CAM 可解释性、JPEG 参数消融、全测试集关键验证和运行效率。
6. 类别鲁棒性、JPEG 防御失败案例、自适应攻击和 FGSM 对抗训练。

## 暂不写入正式结论的实验

- Streamlit/Gradio 交互式 Demo 录屏。
- PGD adversarial training、TRADES 等更强模型级防御。
- SimpleCNN / ResNet stem 等跨模型结构消融。

## 下一步建议

优先完成：

1. 使用 `reports/word/教师要求版-基于ResNet交通标志识别的对抗扰动攻击与输入防御实验报告.docx` 作为主报告。
2. 根据 `reports/figure_table_index.md` 和 `reports/pr6_additional_results_index.md` 制作 PPT。
3. 若老师要求演示，录制离线 Demo：clean 识别 -> 攻击 -> 防御 -> 失败案例 -> 对抗训练对比。

有余力再补：

1. PGD/TRADES 对抗训练。
2. 跨模型结构消融。
3. Streamlit Demo。
