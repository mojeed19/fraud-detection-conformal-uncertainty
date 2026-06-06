# Conformal + Causal Uncertainty Quantification for Real-Time Fraud Detection under Concept Drift

## Abstract

This project implements a robust fraud detection pipeline that combines cost‑sensitive XGBoost with conformal prediction and causal‑inspired drift detection. The model is evaluated on a real‑world credit card fraud dataset (0.1727% fraud rate). The cost‑sensitive XGBoost achieves an Average Precision (AP) of 0.7857, a Brier score of 0.0027 (excellent calibration), and an expected cost of $7,490 (FN cost $500, FP cost $10). Split conformal prediction exceeds 99% coverage (over‑conservative), while Adaptive Conformal Inference (ACI) also shows >99% coverage on the first 1000 streaming points. The naive conformal decision rule (flag only when prediction set = {fraud}) performs worse than the cost‑optimised threshold, with an expected cost of $8,600. Drift detection via Maximum Mean Discrepancy (MMD) and change point detection identifies three distinct shifts in the data distribution.

## Aim and Objectives

**Aim:** To develop and evaluate an uncertainty‑aware fraud detection system that remains robust under concept drift, incorporating cost sensitivity and conformal prediction.

**Objectives:**
1. Automatically detect the target column and preprocess features (log‑transform `Amount`, drop `Time`).
2. Train a cost‑sensitive XGBoost classifier with an optimal decision threshold minimising expected cost (FN = $500, FP = $10).
3. Implement split conformal prediction to construct prediction sets with 95% coverage.
4. Apply Adaptive Conformal Inference (ACI) to maintain coverage in an online streaming setting.
5. Detect concept drift using Maximum Mean Discrepancy (MMD) with sliding windows and change point detection (PELT).
6. Evaluate the model using average precision, Brier score, expected cost, and visualise results (calibration curve, ACI coverage, SHAP analysis).

## Data Source

The dataset used is the **Credit Card Fraud Detection dataset** from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains 284,807 transactions with 29 anonymised features (`V1` to `V28`), `Amount`, and `Time`. The target variable `Class` indicates fraud (1) or legitimate (0). The fraud rate is 0.1727%, making it highly imbalanced and realistic for fraud detection.

## Mathematical Methodology

### 1. Cost‑Sensitive XGBoost

XGBoost is trained with `scale_pos_weight` = (number of non‑fraud)/(number of fraud) to handle class imbalance. The optimal threshold $\tau^*$ is found by minimising the expected cost on a validation set:

$$
\min_{\tau} \left[ \text{FN} \times C_{\text{FN}} + \text{FP} \times C_{\text{FP}} \right]
$$

where $C_{\text{FN}} = 500$, $C_{\text{FP}} = 10$, FN = false negatives, FP = false positives.

### 2. Split Conformal Prediction

- Nonconformity score: $s_i = 1 - P(y_i^{\text{true}} \mid x_i)$
- Quantile level: $\hat{q} = \left\lceil (1-\alpha)(n_{\text{cal}}+1) \right\rceil / (n_{\text{cal}}+1)$
- Prediction set for a new instance: $\mathcal{T}(x) = \{ y : 1 - P(y \mid x) \le \hat{q} \}$
- Target coverage: $1-\alpha = 0.95$.

### 3. Adaptive Conformal Inference (ACI)

Maintains a running quantile $q_t$ updated online:

$$
q_{t+1} = q_t + \gamma \left( \alpha - \frac{1}{t} \sum_{i=1}^t \mathbf{1}(s_i \le q_t) \right)
$$

where $\gamma = 0.005$ is the learning rate, and $s_i$ is the nonconformity score of the $i$-th online point.

### 4. Drift Detection (MMD + Change Point)

Maximum Mean Discrepancy (MMD) between two sliding windows $X$ and $Y$:

$$
\text{MMD}^2(X,Y) = \frac{1}{m^2}\sum_{i,j} k(x_i,x_j) + \frac{1}{n^2}\sum_{i,j} k(y_i,y_j) - \frac{2}{mn}\sum_{i,j} k(x_i,y_j)
$$

with RBF kernel $k(x,y)=\exp(-\gamma \|x-y\|^2)$, $\gamma=0.1$. Change points are detected using the PELT algorithm (`ruptures` library).

## Results and Findings

| Metric | Value |
|--------|-------|
| Average Precision (AP) | 0.7857 |
| Brier score | 0.0027 |
| XGBoost + optimal threshold cost | $7,490 (FP=199, FN=11) |
| Conformal (static) decision cost | $8,600 (FP=10, FN=17) |
| Split conformal coverage (static) | 0.999 (target 0.95) |
| ACI coverage (first 1000 online) | 0.998 |
| Drift change points | 45, 85, 424 |

**Key findings:**
- The cost‑sensitive threshold (0.340) outperforms the naive conformal decision rule, reducing expected cost by ≈14.8%.
- Both static and adaptive conformal methods are overly conservative, achieving >99% coverage (much higher than the 95% target).
- The model has moderate discriminative power (AP=0.7857) but excellent probability calibration (Brier=0.0027).
- MMD detected three distinct data distribution shifts, confirming the presence of concept drift.

## Recommendations

1. **Improve discriminative performance:** Experiment with gradient‑boosted trees (LightGBM, CatBoost), neural networks, or anomaly detection methods (Isolation Forest) to raise AP above 0.9.
2. **Tune ACI parameters:** Reduce the learning rate $\gamma$ (e.g., 0.001) to avoid over‑adaptation and maintain coverage closer to 95%.
3. **Use a cost‑aware conformal decision rule:** Instead of flagging only when the set is exactly `{fraud}`, flag when the set includes fraud and the predicted probability exceeds a cost‑optimised threshold. This reduces false negatives without incurring excessive false positives.
4. **Retrain after detected change points:** The drift detector identified change points at indices 45, 85, and 424. In a real system, retrain or recalibrate the model after such events.
5. **Replace the static conformal rule** with a cost‑sensitive reject option: defer decisions when the prediction set contains both classes and the cost of uncertainty is high.

## Conclusion

This project demonstrates a comprehensive uncertainty‑aware fraud detection system that integrates cost‑sensitive learning, conformal prediction, adaptive online inference, and drift detection. While the cost‑optimised XGBoost achieves reasonable financial performance ($7,490 expected cost), the conformal decision rule as implemented is financially suboptimal. The drift detection module successfully identifies distribution shifts, providing actionable triggers for model maintenance. Future work should focus on improving the base model’s ranking ability (AP) and designing cost‑aware conformal rules that balance false positives and false negatives more effectively. The pipeline is modular and can be applied to any credit card fraud dataset with minimal adaptation.
