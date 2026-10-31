# Retrieval Evaluation Results

## 1. Chunk Size Comparison (Real 401 Document Sections, 19 Canonical Files)

|   Chunk Size |   Chunk Overlap |   Total Chunks | Hit@1 (%)   | Hit@3 (%)   | Trade-off Observation                        |
|-------------:|----------------:|---------------:|:------------|:------------|:---------------------------------------------|
|          300 |              50 |           1460 | 42.9%       | 78.6%       | Granular but fragmented context              |
|          500 |              80 |            918 | 35.7%       | 64.3%       | Optimal balance of coherence and specificity |
|          800 |             120 |            600 | 35.7%       | 85.7%       | Broad context, risk of tangential dilution   |

## 2. Hit Rate @ k on Production FAISS Index (918 Chunks)

|   k |   Hits |   Total In-Scope Questions | Hit Rate (%)   |
|----:|-------:|---------------------------:|:---------------|
|   1 |      5 |                         14 | 35.7%          |
|   2 |      8 |                         14 | 57.1%          |
|   3 |      9 |                         14 | 64.3%          |
|   5 |     12 |                         14 | 85.7%          |

