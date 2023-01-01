This is the pytorch version of Bert-LWAN which is from paper *An Empirical Study on Large-Scale Multi-Label Text Classification Including Few and Zero-Shot Labels*

The model architecture is referrenced on CAML which is from paper *Explainable Prediction of Medical Codes from ClinicalText*

the result is:

|mertric      |pytorch version      |tensorflow version      |the result of the paper      |
| ---------- | ---------- | ---------- | ---------- |
| Harmony | RP@5: 0.784 nDCG@5: 0.810 | RP@5: 0.781 nDCG@5: 0.805 | RP@5: 0.803 nDCG@5: 0.829 |
| Frequent | RP@5: 0.822 nDCG@5: 0.832 | RP@5: 0.821 nDCG@5: 0.830 | RP@5: 0.843 nDCG@5: 0.854 |
| Few | RP@5: 0.644 nDCG@5: 0.602 | RP@5: 0.648 nDCG@5: 0.613 | RP@5: 0.699 nDCG@5: 0.650 |
| Zero | RP@5: 0.011 nDCG@5: 0.006 | RP@5: 0.045 nDCG@5: 0.020 | RP@5: - nDCG@5: - |
