# E00 项目初始化

## 1. 基本信息

- 实验编号：E00
- 实验名称：项目初始化与实验计划制定
- 日期：2026-06-13
- 负责人：待填写
- 代码版本：初始化版本
- 运行设备：待填写

## 2. 实验目的

建立项目仓库结构，明确后续实验路线，并提前规划报告需要展示的结果图、指标表和中间过程图片。

## 3. 本次完成内容

- 创建项目仓库。
- 初始化 git。
- 建立 `data/`、`src/`、`results/`、`reports/`、`docs/` 等目录。
- 编写实验与实现计划。
- 编写结果保存规范。
- 编写报告大纲和 PPT 设计。
- 编写实验记录模板。

## 4. 输出路径

```text
docs/experiment_implementation_plan.md
docs/result_saving_guide.md
docs/report_outline.md
docs/ppt_storyboard.md
docs/task_checklist.md
```

## 5. 关键结论

本项目采用如下技术路线：

```text
交通标志基础识别 -> FGSM/PGD 对抗攻击 -> 对抗训练/输入预处理防御 -> 可解释性分析 -> Demo 展示
```

## 6. 下一步计划

1. 下载 GTSRB 数据集。
2. 实现数据检查与样例图导出。
3. 训练 ResNet18 基础识别模型。

