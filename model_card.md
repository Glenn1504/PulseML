# Model Card — PulseML Deterioration Detector

## Model Details
- **Type**: XGBoost + LSTM ensemble (XGBoost-only in this release)
- **Version**: v1.0.0
- **Prediction task**: Binary classification — will this ICU patient deteriorate within 6 hours?
- **Output**: Probability ∈ [0,1] + risk level (LOW / MEDIUM / HIGH)

## Training Data
- **Source**: Synthetically generated vital sign time series (not real patient data)
- **Size**: 368,443 training windows from 3,500 simulated patients
- **Positive rate**: ~2.9% (windows where deterioration occurs within 6h)
- **Features**: 42 engineered features from 6-hour sliding windows over HR, SBP, DBP, SpO2, RR, Temperature

## Performance
| Metric | Value |
|---|---|
| AUROC | 0.836 (95% CI: 0.827–0.844) |
| AUPRC | 0.375 (95% CI: 0.356–0.393) |
| Sensitivity | 37.0% @ threshold 0.36 |
| Specificity | 98.6% @ threshold 0.36 |

## Intended Use
**This model is a research prototype and must not be used for real clinical decisions.**
It is intended to demonstrate end-to-end ML system design for portfolio purposes.

## Limitations
- Trained on synthetic data — real ICU data (e.g. MIMIC-III) would be required for clinical validity
- No prospective validation
- LSTM component not trained in this release (CPU constraint)
- No fairness evaluation across demographic subgroups

## Ethical Considerations
- Predictions must be reviewed by qualified clinicians
- False negatives (missed deteriorations) are clinically costly — threshold should be tuned with domain experts
- Model requires regulatory approval (FDA 510(k) or De Novo) before clinical deployment in the US

## How to Reproduce
```bash
make data && make pipeline && make train && make eval
```