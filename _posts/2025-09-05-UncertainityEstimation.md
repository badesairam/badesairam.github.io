# Uncertainty Estimation in Healthcare AI

Building on our previous discussion of imbalanced regression in clinical AI for clinically reliable AI, we now turn to another critical challenge: uncertainty estimation. While addressing data imbalance helps models pay attention to rare cases, we must also ensure that when our AI makes predictions, it can honestly communicate how confident it is in those predictions. This work is detailed in our paper, [HypUC: A Framework for Handling Imbalanced Regression in Healthcare](https://arxiv.org/abs/2311.13821) published in TMLR 08/2023.

When an AI model analyzes a patient's ECG and predicts a potassium level of 5.5 mmol/L, clinicians need to know: is the model 95% confident in this prediction, or is it just making its best guess? This distinction is crucial in healthcare, where overconfident predictions could lead to wrong treatments, while underconfident ones might trigger unnecessary tests. The AI must "know what it doesn't know" – a principle that forms the foundation of trustworthy clinical AI.

In our HypUC framework (Hyperfine Uncertainty Calibration), we developed techniques that go beyond simple point predictions. Instead of just outputting a single number, our models provide a full probability distribution that quantifies uncertainty. More importantly, we rigorously calibrate this uncertainty so it reflects reality. In this article, we'll explore why uncertainty calibration is essential for clinical AI and how we achieved reliable uncertainty estimates that clinicians can trust with patients' lives.

## Why Uncertainty in AI Predictions is Crucial

Consider a model analyzing an ECG to predict a patient's blood potassium level. If it predicts 5.5 mmol/L, is it certain this indicates hyperkalemia, or could the actual value be 4.5 mmol/L with some measurement error? The clinical response depends entirely on the model's confidence level. A careful clinician might order a confirmatory lab test if the model expresses uncertainty, whereas high confidence might prompt immediate intervention.

This scenario highlights a fundamental problem in traditional machine learning: many models, especially deep neural networks, are poorly calibrated. Their predicted probabilities or confidence intervals don't match actual error rates. A model might claim "90% confidence" but only be correct 70% of the time—a dangerous overconfidence that could lead to clinical errors.

In our work, we address this by implementing probabilistic regression: instead of predicting a single number, our models predict a full probability distribution (e.g., a Gaussian with mean and variance) for the target variable. This provides an initial uncertainty estimate—wider distributions indicate greater uncertainty. 

### Probabilistic Regression Framework

In probabilistic regression, the model predicts the parameters of a probability distribution for each input, allowing it to express uncertainty in its predictions. A common choice is the Gaussian distribution, where the model predicts both the mean $\mu(\mathbf{x})$ and the variance $\sigma^2(\mathbf{x})$ for a given input $\mathbf{x}$ (ECG in our case).

The objective function is typically the negative log-likelihood of the observed data under the predicted distribution. For a Gaussian distribution, this is given by:

$$\mathcal{L}_{\text{reg}}(\mu(\mathbf{x}_i), \sigma^2(\mathbf{x}_i), y_i) = \frac{1}{2} \log(2\pi\sigma^2(\mathbf{x}_i)) + \frac{(y_i - \mu(\mathbf{x}_i))^2}{2\sigma^2(\mathbf{x}_i)}$$

where:
- $y_i$ is the true target value for the $i$-th sample
- $\mu(\mathbf{x}_i)$ is the predicted mean for the $i$-th sample
- $\sigma^2(\mathbf{x}_i)$ is the predicted variance for the $i$-th sample

#### Explanation of the Components:

1. **Logarithmic Term**: $\frac{1}{2} \log(2\pi\sigma^2(\mathbf{x}_i))$
   - This term accounts for the uncertainty in the prediction. A larger variance $\sigma^2(\mathbf{x}_i)$ implies more uncertainty, which increases the log-likelihood penalty.

2. **Squared Error Term**: $\frac{(y_i - \mu(\mathbf{x}_i))^2}{2\sigma^2(\mathbf{x}_i)}$
   - This term measures how far the predicted mean is from the actual target value, scaled by the predicted variance. If the model is confident (small $\sigma^2(\mathbf{x}_i)$), deviations from the mean are penalized more heavily.

The goal of the objective function is to find the parameters of the model that minimize the negative log-likelihood across all training samples. This ensures that the model not only predicts accurate means but also provides reliable estimates of uncertainty. By optimizing this function, the model learns to balance the trade-off between fitting the data well and expressing appropriate uncertainty.


Beyond statistical metrics, the practical impact of calibrated uncertainty is profound: it enables safer integration of AI into clinical workflows. Our model can flag predictions as "uncertain" when the predicted distribution is too broad or has high entropy, alerting clinicians to step in or suggesting follow-up diagnostic tests. Conversely, when the model expresses high confidence, it can enable faster clinical decisions.

Regulators and clinicians are far more comfortable with AI that can honestly express uncertainty. It's analogous to a junior doctor saying "I'm not entirely sure, let's double-check"—a humble approach that's infinitely better than unfounded certainty.


## Our Approach: Global + Hyperfine Uncertainty Calibration

However, we discovered that this raw uncertainty often requires calibration to align with reality, particularly in the distribution tails or underrepresented regions. Uncertainty calibration involves adjusting the model's predictive distribution so that its uncertainty statements are statistically valid. If the model claims "I am 95% sure the true value lies between X and Y," then in practice, the true value should fall within that range about 95% of the time. We quantify calibration quality using metrics like Uncertainty Calibration Error (UCE), which measures the deviation between predicted confidence intervals and actual outcomes. Our goal is to minimize UCE, ensuring our model's uncertainty estimates are trustworthy and reliable.

Predicting uncertainty is one thing; calibrating it is another challenge. We took a two-stage approach in HypUC: global calibration followed by "hyperfine" calibration.

**Global uncertainty calibration**: First, after training our probabilistic regression model, we find a single scaling factor to adjust its predicted variances (uncertainties) overall. We do this by evaluating the model on a validation set and solving for a factor that makes the 90% prediction interval actually contain ~90% of true values. This is analogous to temperature scaling in classification, but for regression uncertainty. It corrects any overall tendency of the model to be over- or under-confident in general. However, one scaling factor can't capture all the nuanced variations across different prediction ranges.

**Hyperfine calibration**: We partition predictions into bins based on their value range and compute specialized scaling factors for each bin. This addresses the issue that models might be well-calibrated on average but still miscalibrated for specific ranges—too uncertain for moderate values, not uncertain enough for extremes.

The result is a calibrated prediction like "ECG analysis predicts potassium = 5.5 ± 0.3" where we have high confidence that ±0.3 truly represents the 1-standard deviation range. The entire process is post-hoc, meaning we adjust outputs without retraining the neural network. We use a held-out validation set (or an "augmented" validation with data-driven augmentation) to learn the calibration parameters, building on prior methods like Laves et al. (2020) but extending to multi-factor hyperfine calibration for superior performance on large, imbalanced datasets.

## Evidence: Reliable Uncertainty, Verified

How do we know our approach worked? We evaluated HypUC's uncertainty calibration on multiple clinical tasks (age prediction, survival time prediction, serum potassium, LVEF) using our test sets, measuring metrics like UCE and the percentage of true values falling within the model's predicted intervals. The calibrated model achieved very low UCE, indicating excellent alignment between predicted and actual uncertainties.

For example, prior to calibration, a certain task's 90% confidence interval might only cover 80% of true values (UCE off by 10%). After our two-stage calibration, it covered ~90% as intended, with appropriately tight interval widths.

Another way to visualize our results is shown in the figure below, which compares the model's predicted distribution against the ground truth distribution across many examples. Each subplot shows the relationship between predicted values and true values. The red line/area represents the peak and spread of the ground truth distribution, while the blue line/area represents the peak and spread of the model's calibrated predicted distribution (HypUC). Ideally, the blue (prediction) region should overlap the red (actual) region.

![Uncertainty Estimation](/assets/images/hypuc/uncertainity.png)

We see that for Age, Survival, Serum Potassium, and LVEF, the HypUC predictive distributions tightly encompass the true values (blue covers red), indicating that the model's uncertainty estimates are well-calibrated. In other words, when HypUC claims 95% confidence, the outcome indeed falls within its predicted range about 95% of the time—across all these varied clinical tasks. This level of calibration is unprecedented in clinical ML models, which have historically struggled with reliable uncertainty measures.

We also tested how calibrated uncertainties can be used in clinical workflows. By flagging and removing the most uncertain predictions, we found that model performance on the remaining cases improved significantly. When we drop the 10% of test samples with highest uncertainty, the model's error rates on the remaining 90% drop substantially. This demonstrates that our uncertainty metric effectively identifies the trickiest cases where the model was likely to err. In clinical practice, this means the AI can handle routine cases automatically while flagging challenging ones for human review—a safe and intelligent paradigm.

Critically, our enormous dataset makes the uncertainty calibration approach feasible and robust. Learning fine-grained calibration corrections from sparse data would be unreliable; but with millions of samples spanning different hospitals, patient populations, and modalities, our calibration is smooth and trustworthy even in the far tails of the distribution. This approach is a prime example of how having massive, diverse, longitudinal data is a game-changer for building clinically reliable AI.

## Building Trustworthy AI, One Uncertainty at a Time

Calibrating uncertainty is not merely a technical exercise—it's about building trustworthy AI for healthcare. By ensuring our models can honestly express their limitations, we make them significantly more valuable and clinically safe. Our work on HypUC demonstrated that it's possible to achieve highly reliable uncertainty estimates even in complex deep learning models operating on noisy biosignals. This sets the stage for a new paradigm of AI assistance: one that can be transparent about its confidence levels. We believe this transparency is fundamental to the future of AI in medicine—it's how we transform opaque algorithms into tools that clinicians can confidently rely on.

For engineers and researchers reading this, if you're excited by the idea of making AI not just accurate but honestly trustworthy, we invite you to explore opportunities with our team. We're working on some of the hardest—and most rewarding—problems at the intersection of machine learning and medicine. By combining state-of-the-art algorithms, an unparalleled dataset, and a mission that matters, we're pushing the frontier of what AI can do in healthcare. 

### References
1. [HypUC: A Framework for Handling Imbalanced Regression in Healthcare](https://arxiv.org/abs/2311.13821)
2. [Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning](https://proceedings.mlr.press/v121/laves20a.html)