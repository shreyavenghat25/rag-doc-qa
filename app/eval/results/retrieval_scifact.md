**BEIR scifact** · 5,183 documents · 300 test queries · 20 candidates per query

| Configuration | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | Recall@20 | Latency / query |
|---|---|---|---|---|---|---|
| bm25 | 0.560 | 0.524 | 0.616 | 0.686 | 0.737 | 8 ms |
| dense | 0.717 | 0.681 | 0.762 | 0.842 | 0.875 | 18 ms |
| hybrid | 0.679 | 0.640 | 0.734 | 0.823 | 0.882 | 21 ms |
| hybrid+rerank | 0.700 | 0.666 | 0.746 | 0.832 | 0.882 | 301 ms |
