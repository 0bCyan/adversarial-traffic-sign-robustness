# 基于对抗攻击与鲁棒防御的交通标志识别系统

本仓库用于课程期末大作业：从交通标志基础识别出发，构造对抗扰动样本，分析模型脆弱性，并通过鲁棒防御方法提升模型在扰动场景下的稳定性。

## 选题定位

题目建议：

**基于对抗攻击与鲁棒防御的交通标志识别系统设计与实现**

核心路线：

```text
GTSRB 数据集
  -> 基础识别模型训练
  -> FGSM / PGD 对抗攻击
  -> 对抗训练 / 输入预处理防御
  -> 指标对比与可视化展示
  -> Streamlit 或 Gradio 演示系统
```

## 项目亮点

- 从普通分类任务扩展到模型安全性分析，技术深度更强。
- 对抗样本具备强展示性：图片肉眼变化很小，但模型预测显著改变。
- 攻击与防御形成完整闭环，便于写实验过程和问题分析。
- 每个阶段都保存中间图片、指标表、日志和报告素材，方便答辩展示。

## 仓库结构

```text
.
├── configs/                  # 实验配置文件
├── data/                     # 数据集目录，不提交大文件
│   ├── raw/                  # 原始数据
│   ├── processed/            # 预处理后的数据索引或小样本
│   └── samples/              # 报告展示用样例图
├── docs/                     # 计划、实验记录、报告素材说明
├── experiments/              # 每次实验的运行记录
├── notebooks/                # 探索性分析
├── reports/                  # 实验报告、PPT素材、图表导出
│   ├── figures/              # 报告用精选图片
│   └── tables/               # 报告用表格
├── results/                  # 模型、指标、预测图和中间结果
│   ├── 00_dataset_check/
│   ├── 01_baseline/
│   ├── 02_attack/
│   ├── 03_defense/
│   ├── 04_explainability/
│   ├── 04_extended_analysis/
│   ├── 06_runtime/
│   └── 05_demo/
├── src/                      # 源代码
│   ├── data/
│   ├── models/
│   ├── attacks/
│   ├── defenses/
│   ├── visualization/
│   └── demo/
└── tests/                    # 基础测试
```

## 预期实验阶段

1. 数据集检查与可视化。
2. 基础交通标志识别模型训练。
3. FGSM 与 PGD 对抗攻击。
4. 对抗训练和输入预处理防御。
5. Grad-CAM、类别鲁棒性、失败案例和运行效率可视化。
6. 自适应攻击验证与交互式演示材料整理。

详见 [实验与实现计划](docs/experiment_implementation_plan.md)。

## 当前进度

已完成并可写入报告：

- GTSRB 数据集检查与可视化。
- ResNet18 基础识别训练。
- FGSM / PGD 对抗攻击评估。
- Gaussian Blur / Median Filter / JPEG Compression 输入预处理防御。
- Grad-CAM 可解释性分析。
- JPEG quality 参数消融、完整测试集关键配置验证、运行效率统计。
- 按类别鲁棒性统计和 JPEG 防御失败案例分析。
- JPEG 防御下 BPDA 自适应攻击验证。
- FGSM 对抗训练模型级防御实验。

仍作为后续扩展：

- Streamlit/Gradio 交互式 Demo 录屏。
- PGD adversarial training、TRADES 等更强鲁棒训练。
- SimpleCNN / ResNet stem 等跨模型结构消融。

当前结果汇总见 [reports/result_summary.md](reports/result_summary.md)。  
报告草稿见 [reports/draft_report.md](reports/draft_report.md)。  
图表索引见 [reports/figure_table_index.md](reports/figure_table_index.md)。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

运行数据集检查并生成报告素材：

```bash
python -m src.data.check_gtsrb --config configs/dataset_check.yaml
```

训练基础识别模型：

```bash
python -m src.train_classifier --config configs/baseline_resnet18.yaml
```

评估 FGSM / PGD 对抗攻击：

```bash
python -m src.evaluate_attacks --config configs/attack_fgsm_pgd.yaml
```

评估输入预处理防御：

```bash
python -m src.evaluate_input_defense --config configs/defense_input_preprocessing.yaml
```

进行 FGSM 对抗训练防御：

```bash
python -m src.train_adversarial --config configs/defense_adversarial_training.yaml
```

运行补充鲁棒性验证：

```bash
python -m src.visualization.gradcam_analysis --config configs/explainability_gradcam.yaml
python -m src.evaluate_jpeg_ablation --config configs/defense_jpeg_ablation.yaml
python -m src.evaluate_per_class_failure --config configs/per_class_failure_analysis.yaml
python -m src.evaluate_adaptive_jpeg_attack --config configs/adaptive_jpeg_attack.yaml
python -m src.benchmark_runtime --config configs/runtime_benchmark.yaml
```

输出位置：

- 完整结果：`results/00_dataset_check/`
- 报告图片：`reports/figures/`
- 报告表格：`reports/tables/`

## 结果保存原则

每个实验都必须保存：

- 配置文件：模型、数据、超参数、随机种子。
- 日志文件：训练日志、测试日志、异常记录。
- 指标表：CSV 或 JSON，便于画图和写报告。
- 中间图片：原图、扰动图、对抗样本、防御后样本、预测结果图。
- 精选报告素材：复制或导出到 `reports/figures/` 和 `reports/tables/`。

详见 [结果保存规范](docs/result_saving_guide.md)。
