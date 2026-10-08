# Results

Data split: test_clean n=9487, test_official n=10000. Numbers come from `runs/*/metrics.json` produced by `python -m notmnist.evaluate`.

| run | params | val acc | test_clean acc [95% CI] | test_official acc | macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18_probe_s42 | 11,181,642 | 81.25% | 87.28% [86.59–87.93] | 87.65% | 0.8731 | 0.4361 | 0.0433 |
| legacy_keras_mlp | 203,530 | — | 92.82% [92.28–93.32] | 93.10% | 0.9275 | 0.2519 | 0.0075 |
| mlp_baseline_s42 | 203,530 | 87.40% | 93.20% [92.68–93.69] | 93.46% | 0.9316 | 0.2414 | 0.0059 |
| cnn_baseline_s42 | 462,858 | 90.26% | 94.62% [94.15–95.06] | 94.85% | 0.9458 | 0.2703 | 0.0287 |
| legacy_keras_cnn | 462,858 | — | 94.91% [94.45–95.33] | 95.12% | 0.9486 | 0.2880 | 0.0304 |
| resnet18_scratch_s42 | 11,181,642 | 91.71% | 95.88% [95.46–96.26] | 96.02% | 0.9583 | 0.1329 | 0.0051 |
| resnet18_finetune_s42 | 11,181,642 | 92.15% | 96.02% [95.60–96.39] | 96.18% | 0.9598 | 0.1308 | 0.0042 |
| cnn_improved_s42 | 288,618 | 92.79% | 96.72% [96.34–97.06] | 96.83% | 0.9668 | 0.1049 | 0.0047 |

## McNemar ladder (test_clean)

b01 = a wrong, b right; b10 = a right, b wrong; p-value is a two-sided exact binomial test.

| a | b | b01 | b10 | p-value |
|---|---|---:|---:|---:|
| legacy_keras_mlp | mlp_baseline_s42 | 199 | 163 | 0.06568 |
| mlp_baseline_s42 | cnn_baseline_s42 | 312 | 177 | 1.077e-09 |
| cnn_baseline_s42 | cnn_improved_s42 | 279 | 80 | 6.139e-27 |
| cnn_improved_s42 | resnet18_finetune_s42 | 94 | 161 | 3.248e-05 |
| resnet18_probe_s42 | resnet18_finetune_s42 | 895 | 66 | 1.5e-186 |
| resnet18_scratch_s42 | resnet18_finetune_s42 | 168 | 155 | 0.5044 |
