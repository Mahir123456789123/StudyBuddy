# Retrieval Evaluation Results

## 1. Chunk Size Comparison (Real 401 Document Sections, 19 Canonical Files)

|   Chunk Size |   Chunk Overlap |   Total Chunks | Hit@1 (%)   | Hit@3 (%)   | Trade-off Observation |
|-------------:|----------------:|---------------:|:------------|:------------|:----------------------|
|          300 |              50 |           1460 | 42.9%       | 78.6%       | Highest Hit@1 (6/14) with 1460 granular chunks; short windows risk boundary fragmentation |
|          500 |              80 |            918 | 35.7%       | 64.3%       | Standard production baseline (918 chunks); lowest Hit@3 (9/14) on small N=14 sample (+/-7.1% per question) |
|          800 |             120 |            600 | 35.7%       | 85.7%       | Highest Hit@3 (12/14) with 600 broad chunks; wider context captures terms but risks tangential dilution |

> **Sample Size Note**: On this curriculum evaluation set of $N=14$ in-scope questions, each single question represents approximately $7.14\%$ ($1/14$). The percentage differences reflect sensitivity to 2-3 specific questions. Chunk size 500 remains a standard engineering baseline to prevent excessive token dilution in compact 0.5B model prompts, while 800 achieves highest raw Hit@3 on this curriculum set.

## 2. Hit Rate @ k on Production FAISS Index (918 Chunks)

|   k |   Hits |   Total In-Scope Questions | Hit Rate (%)   |
|----:|-------:|---------------------------:|:---------------|
|   1 |      5 |                         14 | 35.7%          |
|   2 |      8 |                         14 | 57.1%          |
|   3 |      9 |                         14 | 64.3%          |
|   5 |     12 |                         14 | 85.7%          |
