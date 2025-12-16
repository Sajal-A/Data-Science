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

**How Linear Regression optimized? OR How does the model learn?**
- The key idea is to choose the coefficients that make the predictions as close as possible to the actual values in the data. This is where optimization comes in.
The most common method is *Ordinary Least Squares (OLS)*. OLS works by minimizing the sum of squared residuals — the squared difference between the actual value $y^i$ and the predicted value (yhat-i) for each data point. The cost function looks like this: $J(\beta) = \sum_{i=1}^{n} (y_i - \bar{y_i})^2$
- **Why squared residuals?**
- Squaring ensures that positive and negative errors don’t cancel out, and it penalizes larger errors more heavily than smaller ones. By minimizing this function, OLS finds the “line of best fit.”

**Optimzation of Cost Function in Linear Regression**

- **How does ordinary least squares (OLS) estimate coefficients?**
- OLS minimizes the sum of squared residuals by solving a closed-form equation. The solution is: $\beta = (X^TX)^{-1}X^Ty$.
- For relatively small datasets with not too many features, we can directly compute the optimal coefficients using linear algebra.
- Here, X is the feature matrix, y is the target vector, and β-hat​ is the vector of coefficients. This solution comes from setting the derivative of the cost function to zero and solving for coefficients. It’s exact, fast for low-dimensional data, and forms the mathematical foundation of linear regression.
- However, the matrix inversion step becomes computationally expensive when the number of features is very large, and sometimes the matrix is not even invertible (especially with multicollinearity).

- **Gradient Descent (Iterative Method)**
- For larger datasets or high-dimensional problems, gradient descent is often preferred. Instead of solving the equations directly, gradient descent takes small steps in the direction that reduces the cost function the most.
The update rule for each coefficient looks like this:
- $\beta_j := \beta_j - \alpha \frac{\partial J}{\partial b_j}$
- Here, α is the learning rate, which controls how big each step is. If α is too small, training is slow; if it’s too large, the algorithm may overshoot and fail to converge.
- Gradient descent is more flexible than the closed-form solution. It can handle massive datasets, can be parallelized, and forms the basis of how many modern machine learning algorithms (like neural networks) are trained.

**What are the common evaluation metrics used in regression problem?**
- Different evaluation metrics capture different aspects of model performance:
- `Mean Squared Error (MSE)`:
-  MSE calculates the average of the squared differences between the actual values and the predicted values. The formula looks like this: $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2$
- The squaring ensures that larger errors are penalized more heavily than smaller ones. MSE is widely used because it connects directly to the optimization objective in ordinary least squares regression. However, since it squares the errors, it is sensitive to outliers.

- `Root Mean Squared Error (RMSE)`:
- $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
- The advantage of RMSE is that it is in the same units as the target variable, which makes it more interpretable. For example, if you are predicting house prices in dollars, RMSE will also be in dollars. Like MSE, though, it still emphasizes large errors because of the squaring.

- `Mean Absolute Error (MAE)`:
- $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} | (y_i - \hat{y}_i) |$
- MAE treats all errors equally, regardless of their size. This makes it more robust to outliers compared to MSE or RMSE. For instance, if your dataset has a few extreme values that are hard to predict, MAE will give a fairer picture of performance than MSE.

- `Coefficient of Determination, or R-squared (R^2)`
- R-squared measures the proportion of variance in the target variable that is explained by the model. The formula is:
- $R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y})^2 } {\sum_{i=1}^{n} (y_i - \bar{y}_i)^2}$
- Here, the denominator represents the total variance in the data (the differences between actual values and their mean), and the numerator is the unexplained variance (the residual sum of squares). An R^2 of 1 means perfect predictions, while an R^2 of 0 means the model does no better than predicting the mean of the target.
  - One limitation of R^2 is that it always increases when you add more features, even if those features are not truly helpful. 

- `Adjusted R-Squared`
- Adjusted R-squared, which penalizes the addition of irrelevant variables. The formula is:
- $\bar{R}^2 = 1 - (1 - R^2)(\frac{n - 1}{n - P - 1})$
- where n is the number of observations and p is the number of predictors. This adjusted version helps us compare models with different numbers of features more fairly.

**Why might you prefer MAE over RMSE as an evaluation metric?**
- Both metrics measure prediction error, but they behave differently. RMSE squares errors, which means it penalizes large deviations much more heavily. This is desirable when large mistakes are unacceptable, such as in financial forecasting. MAE, on the other hand, treats all errors equally and is more robust to outliers. For example, if your dataset has a few extremely unusual house prices, RMSE might exaggerate their influence, whereas MAE will provide a fairer picture of typical model performance.

**What happens if residuals are not normally distributed?**
- Strictly speaking, linear regression does not require residuals to be normal for coefficient estimation. However, normality is important when you want to perform statistical inference — for example, computing confidence intervals or conducting hypothesis tests on coefficients. If residuals deviate strongly from normality, p-values and confidence intervals become unreliable. In practice, with large datasets, the Central Limit Theorem mitigates this issue. Otherwise, you can use bootstrapping or non-parametric methods to obtain more reliable inference.

**How do you detect and handle heteroscedasticity?**
- Heteroscedasticity occurs when the variance of residuals changes across different levels of predicted values. To detect it, you can plot residuals against predictions — a funnel-shaped pattern indicates heteroscedasticity. Statistical tests like Breusch–Pagan can also confirm it.
- If heteroscedasticity is present, standard errors of coefficients are biased, which affects hypothesis testing. Remedies include transforming the target variable (e.g., log-transform), or using Weighted Least Squares, which gives less weight to data points with higher variance.

**What happens if you include irrelevant variables in a regression model?**
- Including irrelevant predictors increases model complexity without improving predictive power. In OLS, this can inflate the variance of coefficient estimates, making them less stable. It also reduces interpretability since you are estimating more parameters than necessary. From an evaluation standpoint, R² will increase slightly even if the variable adds no true value, which can mislead you. This is why Adjusted R² or regularization methods like Lasso are used to discourage unnecessary features.

**How would you evaluate a regression model on imbalanced error costs?**
- In some applications, not all errors are equal. For example, in predicting demand for a product, underestimating demand may be costlier than overestimating. Standard metrics like MAE or RMSE treat errors symmetrically. In such cases, you might define a custom cost function that penalizes certain types of errors more heavily, or use Quantile Regression, which shifts focus toward under- or over-estimation. This shows that you can adapt regression to real business constraints.

**How do you handle missing data in regression?**
- Missing data can bias coefficient estimates if not addressed properly. Common strategies include imputation with mean, median, or mode, but these may distort variance. A more robust approach is regression imputation or using models like k-Nearest Neighbors to estimate missing values. In high-stakes scenarios, multiple imputation is used to account for uncertainty. Importantly, you must also assess why data is missing — whether it is Missing Completely at Random (MCAR), Missing at Random (MAR), or Missing Not at Random (MNAR) — because the choice of strategy depends on this mechanism.

**How feature scaling is important to Linear Regression?**
- Although linear regression itself does not strictly require features to be on the same scale, scaling becomes essential when you extend linear regression with regularization methods like Ridge or Lasso. Without scaling, features with larger numeric ranges can dominate the penalty terms and bias the coefficients. A good practice is to standardize or normalize features before fitting the model when regularization is involved.

**How to avoid overfitting in regression problem?**
- Overfitting is another challenge in regression. With too many predictors, the model may start capturing noise instead of signal. This is where regularization techniques like Ridge regression and Lasso regression become valuable.
- Ridge regression adds a penalty term proportional to the square of the coefficients:
    - $J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y_i})^2 + \lambda \sum_{j=1}^{p} \beta_j^2$
    - This discourages large coefficients and stabilizes the model in the presence of multicollinearity.
- Lasso regression, on the other hand, adds a penalty based on the absolute value of coefficients:
    - $J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y_i})^2 + \lambda \sum_{j=1}^{p} | \beta_j |$
    - The absolute penalty has the effect of driving some coefficients exactly to zero, which makes Lasso a useful method for feature selection.

**How outliers effect your linear regression model?**
- Outlier is an another pitfall. Since linear regression minimizes squared errors, outliers can disproportionately influence the coefficients. For example, if most house prices in a dataset are between $200k and $500k, but one mansion is $5 million, the fitted line may shift dramatically because of that single point. Checking residual plots and using robust regression techniques are ways to mitigate this issue.

**What is the difference between R² and Adjusted R²?**
- R² measures the proportion of variance in the target explained by the model. However, it always increases when you add more features, even if they are irrelevant. Adjusted R² corrects for this by penalizing the inclusion of unnecessary variables: $\bar{R}^2 = 1 - (1 - R^2)(\frac{n - 1}{n - P - 1})$
- where n is the number of observations and p is the number of predictors. Adjusted R² is therefore more reliable for comparing models with different numbers of features, especially in interview case studies where feature selection is part of the exercise.








