# Experiment 3: Quantization & QLoRA Profiling Results

| Config                  |   Allocated VRAM (MB) |   Reserved VRAM (MB) |   Avg Latency (ms) | Quality   |
|:------------------------|----------------------:|---------------------:|-------------------:|:----------|
| Baseline FP16           |                  1050 |                 1200 |              145.2 | High      |
| INT8 Quantization       |                   620 |                  750 |              188.5 | High      |
| INT4 / NF4 Double Quant |                   382 |                  490 |              218.4 | High      |