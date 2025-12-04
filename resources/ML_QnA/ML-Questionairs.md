# Bias and Variance

**What do you understand by the terms bias and variance in machine learning?**
- Bias represents the systematic error that occurs when your model makes simplifying assumptions about the data. In machine learning terms, bias measures how far off your model's predictions are from the true values, on average, across different possible training sets.
- Variance tells a completely different story. While bias is about being systematically wrong, variance is about being inconsistent. Variance measures how much your model's predictions change when you train it on slightly different datasets.

**Can you give a real-world example of each?**
- *Bias(example)* Consider a real-world scenario where you're trying to predict house prices. If you use a simple linear regression model that only considers the square footage of a house, you're introducing bias into your system. This model assumes that house prices have a perfectly linear relationship with size, ignoring crucial factors like location, neighborhood quality, age of the property, and local market conditions. Your model might consistently undervalue houses in premium neighborhoods and overvalue houses in less desirable areas—this systematic error is bias.
- *Variance (Example)* In our house price prediction example, imagine you're using a very deep decision tree instead of linear regression. This complex model might perform brilliantly on your training data, capturing every nuance and detail. But here's the problem: if you collect a new set of training data from the same market, your decision tree might look completely different. One day it might predict that houses with pools are worth 50% more, but with slightly different training data, it might conclude that pools actually decrease property value. This sensitivity to training data variations is variance.

**Explain the bias-variance tradeoff**
- The bias-variance tradeoff is the fundamental principle that you cannot simultaneously minimize both bias and variance. As you make your model more complex to reduce bias (better fit to training data), you inevitably increase variance (sensitivity to training data changes). The goal is finding the optimal balance where total error is minimized. This tradeoff is crucial because it guides every major decision in model selection, from choosing algorithms to tuning hyperparameters.

**How do bias and variance contribute to the overall prediction error?**
- The total expected error of any machine learning model can be mathematically decomposed into three components: Total Error = Bias² + Variance + Irreducible Error. Bias squared represents systematic errors from model assumptions, variance captures the model's sensitivity to training data variations, and irreducible error is the inherent noise in the data that no model can eliminate. Understanding this decomposition helps you identify which component to focus on when improving model performance.

**What is Irreducible error? Can it be minimised?**
- Irreducible error is the part of the prediction error that no model can reduce, because it comes from noise or randomness in the data itself.
- It's the 'unavoidable' error in any system.
- Irreducible error reprsents:
  - Random flactuations in the real world.
  - Measurement mistakes
  - Missing variables that affect the outcomes but are not in the dataset
  - Human errors, sensor noise, data collection inconsistencies
- For example:
  - Suppose you want to predict the house prices using size and location, but the actual price depends on: owner's urgency to sell, negotiation between buyer & seller, market mood or minor property issues not recorder in data.
  - These variables aren't in your dataset -> they create noise.
- You *cannot remove irreducible error using modelling techniques*.
  - but you can reduce it by: improved data quality like better instruments or sensors, consistent data collection or reducing measurement noise.
  - Adding more relevant features, cleaner labels and more controlled environment.

**How would you detect if your model has high bias or high variance?**
- High bias manifests as poor performance on both training and test datasets, with similar error levels on both. Your model consistently underperforms because it's too simple to capture the underlying patterns. High variance shows excellent training performance but poor test performance - a large gap between training and validation errors. You can diagnose these issues using learning curves, cross-validation results, and comparing training versus validation metrics.

**Which machine learning algorithms are prone to high bias vs high variance?**
- High bias algorithms include linear regression, logistic regression, and Naive Bayes - they make strong assumptions about data relationships. High variance algorithms include deep decision trees, k-nearest neighbors with low k values, and complex neural networks - they can model intricate patterns but are sensitive to training data changes. Balanced algorithms like Support Vector Machines and Random Forest (through ensemble averaging) manage both bias and variance more effectively.

**How does model complexity affect the bias-variance tradeoff?**
- Simple models (like linear regression) have high bias because they make restrictive assumptions, but low variance because they're stable across different training sets. Complex models (like deep neural networks) have low bias because they can approximate any function, but high variance because they're sensitive to training data specifics. The relationship typically follows a U-shaped curve where optimal complexity minimizes the sum of bias and variance.

**What techniques can you used to reduce high bias in a model?**
- To combat high bias, you need to increase your model's capacity to learn complex patterns. Use more sophisticated algorithms (switch from linear to polynomial regression), add more relevant features through feature engineering, reduce regularization constraints that oversimplify the model, or collect more diverse training data that better represents the problem's complexity. Sometimes the solution is recognizing that your feature set doesn't adequately capture the problem's nuances.

**When would you choose a biased model over an unbiased one?**
- A biased model over an unbiased one when the biased model gives better generalization performance due to lower variance, more stability, or less overfitting. In practice, an unbiased model is not always the best model.
- Some situation when it can be conidered
  - When the unbaised model has very high variance: High complexit, low-bias models (e.g., deep trees, high-degree polynomials) can fit the training data perfectly - but perform poorly on unseen data.
  - When you have limited data: with small datasets, complex model overfit quickly. A simple biased models can perform better. For example, using linear regression instead of a neural network for only 200 samples.
  - When the problem is noisy (high irreducible error): If the target has a lot of randomness (noise), complex models chase the noise. A baised, simpler model ignores the noise -> better performance. For example: Predicting stock prices, A high-bias, low-variance model (like ridge regrssion) often outperforms high-variance deep nets.
  - For interpretability and stability: biased models are usually simpler, easier to explain and more robust to small data changes. For Example: Logistic regression (biased) vs Random Forest (unbiased, more complex)

**What methods would you employ to reduce high variance without increasing bias?**
- Regularization techniques like L1 (Lasso) and L2 (Ridge) add penalties to prevent overfitting. Cross-validation provides more reliable performance estimates by testing on multiple data subsets. Ensemble methods like Random Forest and bagging combine multiple models to reduce individual model variance. Early stopping prevents neural networks from overfitting, and feature selection removes noisy variables that contribute to variance.

**How do you use learning curves to diagnose bias and variance issues?**
- Learning curves plot model performance against training set size or model complexity. High bias appears as training and validation errors that are both high and converge to similar values - your model is consistently underperforming. High variance shows up as a large gap between low training error and high validation error that persists even with more data. Optimal models show converging curves at low error levels with minimal gap between training and validation performance.

**Explain how regularization techniques help manage the bias-variance tradeoff.**
- Regularization adds penalty terms to the model's cost function to control complexity. L1 regularization (Lasso) can drive some coefficients to zero, effectively performing feature selection, which increases bias slightly but reduces variance significantly. L2 regularization (Ridge) shrinks coefficients toward zero without eliminating them, smoothing the model's behavior and reducing sensitivity to training data variations. The regularization parameter lets you tune the bias-variance tradeoff - higher regularization increases bias but decreases variance.

**How do ensemble methods like Random Forests address variance?**
- Random Forests reduce variance by building many diverse trees and averaging their predictions, which cancels out individual tree overfitting and produces a more stable, accurate model.

**How will you detect Overfitting and Underfitting in practice?**
- *Underfitting* occurs when your model has high bias. The symptoms are unmistakable: poor performance on both training and validation data, with training and validation errors that are similar but both unacceptably high. It's like studying for an exam by only reading the chapter summaries—you'll perform poorly on both practice tests and the real exam because you haven't captured enough detail. In practical terms, if your linear regression model achieves only 60% accuracy on both training and test data when predicting whether emails are spam, you're likely dealing with underfitting. The model isn't complex enough to capture the nuanced patterns that distinguish spam from legitimate emails. You might notice that the model treats all emails with certain keywords the same way, regardless of context.

- *Overfitting* manifests as high variance. The classic symptoms include excellent performance on training data but significantly worse performance on validation or test data. Your model has essentially memorized the training examples rather than learning generalizable patterns. It's like a student who memorizes all the practice problems but can't solve new problems because they never learned the underlying principles. A telltale sign of overfitting is when your training accuracy reaches 95% but your validation accuracy hovers around 70%.

# Linear Regression

**What is Linear Regression?**
- linear regression is about modeling relationships. Imagine you want to predict someone’s weight from their height. Intuitively, taller people tend to weigh more. Linear regression takes this kind of intuition and expresses it in the form of a mathematical equation that describes a straight-line relationship between input variables (features) and an output variable (target).

  - The simplest form is *simple linear regression*, where we have one feature and one target. The equation looks like this:
`y=β0+β1x+ϵ`.
    - y is the target we want to predict,
    - x is the input feature,
    - β0 is the intercept (the value of y when x=0),
    - β1 is the slope (how much y changes when x increases by 1 unit), and
    - ϵ represents the error term — the part of y not explained by our linear model.

  -  When we deal with, we are dealing with more than one input varibales. That's Where *multiple linear regression* come is. The equation generalizes to: `y=β0+β1x1+β2x2+⋯+βnxn+ϵ`
  -  Instead of fitting a straight line, we are fitting a hyperplane in an n-dimensional space. Each coefficient βi​ represents the contribution of a specific feature to the target, while keeping other features constant.
  

**What is ploynomial regression?**
- Polynomial regression is an extension of linear regression where higher-degree terms are added to model non-linear relationships. The general form of the equation for a polynomial regression of degree n is: `y=β0+β1x+β2x^2+…+βnx^n+ϵ`
  - y is the dependent variable.
  - x is the independent variable.
  - β0, β1, β2 are the coefficients of the polynimals terms.
  - n is the degree of the polynomial.
  - ϵ reprsents the error term.
- Choosing the right polynomial degree n is important: a higher degree may fit the data more closely but it can lead to overfitting. The degree should be selected based on the complexity of the data. Once the model is trained, it can be used to make predictions on new data, capturing non-linear relationships and providing a more accurate model for real-world applications.

**What are the key assumptions of linear regression and why do they matter?**
- Linear regression is elegant and powerful, but it comes with a set of assumptions. These assumptions are important because they determine whether the model’s predictions and interpretations are reliable.
- *Linearity* : We assume that the relationship between the features and the target is linear. For example, if you are predicting house prices using square footage, the model assumes that as square footage increases, the price increases or decreases at a constant rate. If the true relationship is curved or more complex, a simple linear regression will underperform.
  
  - *How to test*: Plot the predicted values against the residuals (errors). If the residuals show a random scatter around zero, linearity is likely satisfied. But if you see clear curves or patterns, the relationship might not be linear.
  - *Fixes if violated*: Apply transformations (e.g., log, square root), add polynomial features, or switch to a non-linear model.

- *Independence of errors* : This means that the errors (or residuals) for one data point should not be correlated with those for another. In practice, this assumption is often violated in time-series data where yesterday’s error can influence today’s. If independence is broken, the model may appear more confident than it actually is, leading to misleading conclusions.

  - *How to test*: Use the Durbin-Watson test. Values around 2 suggest independence; values closer to 0 or 4 indicate positive or negative autocorrelation, respectively.
  - *Fixes if violated*: Consider time-series models like ARIMA or add lag variables to capture temporal dependencies.

- *Homoscedasticity* : Which is just a fancy way of saying that the errors have constant variance. In other words, the spread of residuals should be roughly the same across all levels of the predicted values. If the variance of errors grows larger for higher values of the target, we have heteroscedasticity. This can cause the model to give too much weight to certain regions of the data.
  
  - *How to test* : Plot residuals against predicted values. If the spread of residuals is roughly even, the assumption holds. A funnel-shaped pattern (residuals growing larger at one end) suggests heteroscedasticity. You can also use statistical tests like Breusch–Pagan or White’s test.
  - *Fixes if violated* : Transform the dependent variable (e.g., log(y)), or use Weighted Least Squares.

- *Normality of errors* : While the target itself does not need to be normally distributed, we expect the residuals to roughly follow a normal distribution. This matters most when we are performing hypothesis testing or constructing confidence intervals. A strongly non-normal error distribution makes these statistical tests unreliable.
  
  - *How to test*: Plot a histogram or a Q–Q plot of residuals. If residuals follow a straight diagonal line in the Q–Q plot, the assumption is satisfied. You can also use statistical tests like the Shapiro–Wilk test.
  - *Fixes if violated*: With large datasets, this assumption is less critical (Central Limit Theorem helps). Otherwise, consider transformations or non-parametric methods.

- *Multicollinearity* : Linear regression assumes no multicollinearity among the independent variables. Multicollinearity means that two or more features are highly correlated with each other. For example, if you try to predict salary using both “years of experience” and “age,” the model might struggle because these features carry overlapping information. High multicollinearity inflates the variance of coefficient estimates, making them unstable and hard to interpret.
  
  - *How to test* : Compute the Variance Inflation Factor (VIF) for each predictor. A VIF above 5 (sometimes 10) indicates problematic multicollinearity. A correlation heatmap is also a quick way to spot highly correlated features.
  - *Fixes if violated* : Remove one of the correlated variables, combine them into a single feature, or use regularization methods like Ridge or Lasso regression.



