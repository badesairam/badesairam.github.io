# Loss Correction for Imbalanced Regression

Real world regression problems are often highly imbalanced. In medical applications, many health metrics are continuous and have skewed distributions. Imbalanced regression is under-explored compared to imbalanced classification. This work can be considered as an extension to the one proposed for classification in **Long-Tail Learning via Logit Adjustment**.

## Problem Setup

In normal regression, we try to minimize MSE loss but with imbalance in the training distribution, the model might trivially learn to predict just the high shot cases for all the data. So the metric that we are interested in minimizing is the **balanced error**. This can also be thought of as reducing error on a uniformly distributed test set.

Consider:
- `P_train(x,y)` is the distribution from which training data is drawn with `P_train(y)` is imbalanced
- `P_bal(x,y)` is the distribution from which uniformly distributed test set is drawn with `P_bal(y)` is uniform
- `P(x|y)` can be considered the same for all the scenarios

Our aim is to learn `P_bal(x,y)` during training to reduce the error on the uniform test set.

## Loss Derivation

```
LOSS = -log(P_train(y|x))
     = log(P_tr(x|y) * P_tr(y) / P_tr(x))
     = log(P_bal(y|x) * P_train(y) / Sum_y'(P_bal(y'|x) * P_train(y')))
     = -log(P_bal(y|x)) - log(P_train(y)) + log(Sum_y'(P_bal(y'|x) * P_train(y')))
     = -log(P_bal(y|x)) + log(Sum_y'(P_bal(y'|x) * P_train(y')))
```

### Example
- X, y_gt = 4, f(x) = y, s = 3.1, 1.3
- **First term** = MSE(4, 3.1, sigma=1.3)
- **Second term** = Sum_y'(MSE(4, y', sigma=?))

In general regressors, we assume `P(y|x)` as a normal distribution with `y_pred` as mean and a unit normal Gaussian error: `N(y, y_pred, σ)`.

Note: `-log(P_train(y))` can be ignored in minimization (derivation provided at the bottom).

## Loss Function Components

The loss function contains:

1. **P_bal(y|x)** - A model to learn the balanced distribution which is what our model should do finally

2. **Correction term** which penalizes the imbalance. `P_train(y')` can be estimated by binning the training data. `Y'` are the bins. The summation term can be considered as a penalty for high probable bins as just an MSE-based model would try to predict these high probable targets.

3. We can use **uncertainty-based estimation** for the model which is `P_bal(y|x)` that is predict both `ŷ` as well as `σ`.

4. For **low shot and few shot learning**, we can use KDE to estimate `P_train(y')`.

## Implementation Notes

![Loss correction derivation](/assets/images/imbalanced_loss.png)

Substitute equation 2 in equation 1 to get the relation between the components. 



