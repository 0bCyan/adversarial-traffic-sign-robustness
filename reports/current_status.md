# 当前项目状态

更新时间：2026-06-13

## 已完成

- 项目仓库与目录结构。
- GTSRB 数据集检查脚本。
- ResNet18 基础识别训练脚本。
- FGSM / PGD 对抗攻击评估脚本。
- 输入预处理防御评估脚本。
- 报告图片与表格导出。
- 中文实验报告草稿所需的核心数据。

## 可写入正式报告的实验

1. 数据集检查与可视化。
2. ResNet18 基础交通标志识别。
3. FGSM / PGD 对抗扰动攻击。
4. Gaussian Blur / Median Filter / JPEG Compression 输入预处理防御。

## 暂不写入正式结论的实验

对抗训练防御代码已创建，但当前没有完整跑完并验证的结果。因此：

- 可以在报告“改进方向”中提到。
- 不建议在结果章节写具体指标。
- 不建议在 PPT 中宣称已经完成。

## 下一步建议

优先完成：

1. 根据 `reports/draft_report.md` 整理正式 Word/PDF 报告。
2. 根据 `reports/figure_table_index.md` 制作 PPT。
3. 根据 `reports/result_summary.md` 填写结果表。

有余力再补：

1. 对抗训练完整实验。
2. Grad-CAM 可解释性分析。
3. Streamlit Demo。

