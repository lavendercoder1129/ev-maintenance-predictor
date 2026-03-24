# How Confidence Is Calculated

## Overview

The web app runs two real trained models entirely in the browser (no server). Their outputs are combined into a single **ensemble confidence score**.

---

## Step 1 — Logistic Regression probability

The LR model outputs a probability between 0 and 1 using the sigmoid function:

```
score = intercept + (coef_usage × usage) + (coef_downtime × downtime) + (coef_cost × cost)
LR_probability = 1 / (1 + e^(-score))
```

**Trained coefficients (unscaled):**

| Parameter | Value |
|---|---|
| Intercept | -12.181263 |
| Daily Usage | 0.001478 |
| Downtime Hours | 2.023720 |
| Maintenance Cost | 0.003582 |

Downtime has by far the strongest pull — a 1-hour increase in downtime raises the log-odds by ~2.02 points.

---

## Step 2 — Random Forest vote ratio

The RF model is an ensemble of 50 decision trees (max depth 6), each trained on a bootstrap sample of the 500-row dataset. For a given input, each tree traverses its nodes and lands on a leaf that predicts 0 or 1.

```
RF_vote_ratio = (number of trees predicting class 1) / 50
```

All 50 trees are exported as compact arrays and traversed in JavaScript, producing results identical to scikit-learn's `predict_proba`.

---

## Step 3 — Ensemble

```
confidence = (LR_probability + RF_vote_ratio) / 2
```

This simple average gives equal weight to both models. Since both achieved 100% accuracy on the held-out test set, no further calibration weighting was applied.

**Decision threshold:** confidence ≥ 0.50 → Maintenance Required

---

## Feature influence bars

The influence bars shown after each prediction represent each feature's **weighted contribution to the LR log-odds score**:

```
contribution_i = coef_i × value_i
```

These are absolute (always positive) and normalised against the largest contributor so they display as relative percentages. The bar color indicates direction:
- **Red** — pushes toward maintenance required
- **Green** — pushes toward all clear

Since all LR coefficients are positive (higher values of all three features correlate with needing maintenance), bars are always red in practice — but the relative lengths show which feature is driving the decision most strongly.

---

## Why both models agree so consistently

The dataset was generated with clear distributional separation between the two classes:

| Feature | Class 0 (No maintenance) | Class 1 (Maintenance) |
|---|---|---|
| Daily Usage | ~280 sessions | ~418 sessions |
| Downtime Hours | 1–2 hrs | 3–7 hrs |
| Maintenance Cost | ₹500–1,900 | ₹1,600–4,500 |

The boundary is clean enough that both a linear model (LR) and a non-linear model (RF) find it perfectly — hence 100% test accuracy on both.
