# Experiment 3: Quantization & QLoRA Profiling Results

| Config                  |   Allocated VRAM (MB) |   Reserved VRAM (MB) |   Avg Latency (ms) |   Avg Generated Tokens |
|:------------------------|----------------------:|---------------------:|-------------------:|-----------------------:|
| Baseline FP16           |                 942.3 |                  954 |             3093.7 |                     40 |
| INT8 Quantization       |                 610.4 |                  992 |            18390   |                     40 |
| INT4 / NF4 Double Quant |                 444.6 |                  970 |             3929.9 |                     40 |
