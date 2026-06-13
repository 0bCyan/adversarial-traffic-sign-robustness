# 提交清单

## 已具备

- [x] 实验报告草稿：`reports/draft_report.md`
- [x] 结果汇总：`reports/result_summary.md`
- [x] PPT 内容提纲：`reports/ppt_outline_filled.md`
- [x] 图表索引：`reports/figure_table_index.md`
- [x] 数据集检查结果：`results/00_dataset_check/`
- [x] 基础识别结果：`results/01_baseline/resnet18/`
- [x] 对抗攻击结果：`results/02_attack/fgsm_pgd/`
- [x] 输入预处理防御结果：`results/03_defense/input_preprocessing/`
- [x] 报告精选图片：`reports/figures/`
- [x] 报告精选表格：`reports/tables/`

## 提交时建议包含

- `README.md`
- `configs/`
- `src/`
- `docs/`
- `experiments/`
- `reports/`
- `results/00_dataset_check/`
- `results/01_baseline/resnet18/figures/`
- `results/01_baseline/resnet18/metrics/`
- `results/01_baseline/resnet18/samples/`
- `results/02_attack/fgsm_pgd/`
- `results/03_defense/input_preprocessing/`

## 注意事项

- `data/raw/` 数据集较大，通常不提交；报告中说明数据集来源和下载方式。
- 模型权重 `.pth` 文件较大，当前 `.gitignore` 默认不提交；如老师要求可单独打包。
- `results/03_defense/adversarial_training/` 未完整跑完，已清理，不作为正式结果。
- 对抗训练只作为后续扩展，不在正式结果中写具体指标。

## 正式报告待补

- [ ] 小组成员姓名。
- [ ] 小组贡献比例。
- [ ] 学号、课程、教师、日期等封面信息。
- [ ] 将 Markdown 报告排版为 Word 或 PDF。
- [ ] 将 PPT 提纲制作成正式 PPT。

