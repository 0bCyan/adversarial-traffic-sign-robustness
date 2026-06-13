# E06 additional validation plan

## Goal

Add the missing experiments that make the robustness study more complete:

- Grad-CAM case studies for clean/adversarial/defended inputs.
- JPEG compression quality ablation at Q50/Q75/Q90.
- Runtime benchmark for inference, attack generation, and JPEG defense.
- Full-test key configuration validation for FGSM/PGD epsilon=0.03 and JPEG under PGD epsilon=0.03.
- Streamlit demo for interactive attack and defense visualization.

## Run order

1. Ensure the baseline checkpoint exists:

   ```bash
   python -m src.train_classifier --config configs/baseline_resnet18.yaml
   ```

2. Run additional lightweight validation:

   ```bash
   python -m src.visualization.gradcam_analysis --config configs/explainability_gradcam.yaml
   python -m src.evaluate_jpeg_ablation --config configs/defense_jpeg_ablation.yaml
   python -m src.benchmark_runtime --config configs/runtime_benchmark.yaml
   ```

3. Optional full-test key checks:

   ```bash
   python -m src.evaluate_attacks --config configs/attack_fgsm_pgd_full_test.yaml
   python -m src.evaluate_input_defense --config configs/defense_input_preprocessing_full_test.yaml
   ```

4. Optional model-level defense:

   ```bash
   python -m src.train_adversarial --config configs/defense_adversarial_training.yaml
   ```

5. Demo:

   ```bash
   streamlit run src/demo/streamlit_app.py
   ```

## Expected outputs

- `results/04_explainability/gradcam/`
- `results/03_defense/jpeg_quality_ablation/`
- `results/06_runtime/`
- `results/02_attack/fgsm_pgd_full_test/`
- `results/03_defense/input_preprocessing_full_test/`
- `results/05_demo/`
