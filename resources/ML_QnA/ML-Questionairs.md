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


# Logistic Regression

*What is Logistic Regression?*

- Logistic regression: a supervised learning algorithm that can be used to classify data into categories, or classes, 
by predicting the probability that an observation falls into a particular class based on its features.

*From Linear Regression to Classification*

- Linear Regression fits a straight line through data to predict continuous outcomes like house prices or sales numbers. 
Mathematically, it looks like this:
	$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon$

- The output y can take any value — from negative infinity to positive infinity.
But what if we’re not trying to predict a continuous value?
What if we want to predict whether something is yes or no, spam or not spam, disease or no disease?

That’s where Linear Regression fails — because probabilities must always lie between 0 and 1, 
while a straight line can easily predict values outside that range.

- *Enter Logistic Regression*
	- Logistic Regression fixes this by transforming the linear output into a probability using a special function called the sigmoid (or logistic) function.
	- The sigmoid function takes any real number and “squeezes” it into a value between 0 and 1:
	$P(y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-z}}$
	$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$
	
	- Here,
		- P represents the probability of the positive class for example, “1” = spam, churn, fraud, etc.).
		- 1−p represents the probability of the negative class (“0”).
		- The coefficients βi​ determine how strongly each feature influences that probability.
		
		- Once we have the probability p, we can easily classify outcomes:
			- If p>0.5, predict 1
			- If p≤0.5, predict 0	
		- This simple threshold converts a smooth probability curve into a crisp decision boundary.

*What do mean by log-Odds (The "Logit")*

- The term Logistic Regression actually comes from modeling the log-odds (also known as the logit) of the probability.
- Odds represent how much more likely an event is to happen than not to happen:
	- Odds=p/(1−p)
	- Taking the logarithm gives us the log-odds:
	- $logit(p) = \log{\frac{1}{1-p}} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$
	- This relationship is linear in the coefficients.

*Why it matters?*
- Logistic Regression is often the first classification model every data scientist learns — and for good reason:
	- It’s mathematically interpretable — you can explain exactly how each feature affects the odds of an outcome.
	- It’s computationally efficient, even on large datasets.
	- It provides probability estimates, not just hard labels.
	- And it forms the foundation for more advanced models like neural networks and generalized linear models (GLMs).

*What are the assumptions of Logistic Regression?*

- The Dependent variable is Binary (or Categorical)
	- Logistic Regression is designed for classification problems — typically binary ones.
	- That means your target variable should represent two possible outcomes:
		- 1 = Event happens (e.g., customer churns, email is spam)
		- 0 = Event does not happen
	- When you have more than two categories, you can use extensions like 
	Multinomial Logistic Regression (for multiple classes) or Ordinal Logistic Regression (for ordered categories).

- Independence of Observations
	- Each observation in your dataset should be independent of others.
		- For example, if you’re predicting whether users will buy a product, each 
		user should be treated as a separate, unrelated case.
	- If the data points are correlated — say, multiple transactions from the same customer or repeated measures from the same patient 
	the model’s standard errors and significance tests become unreliable.
	- In such cases, we often turn to Generalized Estimating Equations (GEE) or Mixed-Effects Logistic Regression, 
	which account for correlated data.

- Linearity between Features and Log-odds
	- This is one of the misunderstood assumptions.
	- Logistic Regression doesn’t assume a linear relationship between predictors and the target variable itself.
	Instead, it assumes a linear relationship between the independent variables and the log-odds of the dependent variable.
	- $logit(p) = \log{\frac{1}{1-p}} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$
	That means each feature’s contribution is additive in the log-odds space, not directly in probability space.
	If this linearity assumption is violated, the model can misrepresent relationships, leading to biased predictions.

- No (or Minimal) Multicollinearity
	- Multicollinearity occurs when independent variables are highly correlated with each other.
	- This makes it difficult for the model to determine the unique effect of each variable — inflating variance in coefficient estimates.

	- *How to detect multicollinearity?*
		- Variance Inflation Factor (VIF) — values above 5 (or 10) suggest trouble.
		- Correlation matrices — to identify redundant features.
		- To fix it, you can remove correlated variables, use Principal Component Analysis (PCA), 
		or apply regularization (like Ridge or Lasso Logistic Regression).

- Sufficiently Large Sample Size
	- Logistic Regression relies on Maximum Likelihood Estimation (MLE) to find the best-fitting coefficients.
	- MLE performs better with larger sample sizes — because it needs enough data to estimate probabilities accurately, 
	especially for the minority class.

*How Logistic Regression is Optimized?*

- Logistic Regression relies on a powerful idea from statistics called Maximum Likelihood Estimation (MLE) — 
it learns the parameters that make the observed data most probable under the model.

- From Probabilities to Likelihood
	- For each data point i, the model predicts the probability of the positive class as:
		- $P(y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-\left(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n\right)}}$
		
		- pi = predicted probability that yi=1
		- xij = value of feature j for sample i
		- Bj = weight(coefficient) the model is trying to learn.
	- Now, depending on whether the true label yi​ is 0 or 1, the probability of observing that label is:
		- $P(y_i) = p_i^{y_i} (1-p_i)^{(1-y_i)}$
			- If yi=1, the probability is pi​.
			- If yi=0, the probability is 1−pi.
			So this formula elegantly handles both cases in one expression.
	- For all m observations, we multiply these probabilities together (assuming independence) to get the likelihood:
		- $\mathcal{L}(\boldsymbol{\beta}) = \prod_{i=1}^{m} p_i^{y_i} (1-p_i)^{(1-y_i)}$
		- This is the probability of the entire dataset given the parameters β. The goal is to find the coefficients that make this overall likelihood as large as possible (Maximum Likelihood Estimation)

- The Log-Likelihood Function
	- Products of many small probabilities can quickly underflow to zero numerically, 
	so we take the logarithm (which turns products into sums) to get the log-likelihood:
		- $\ell(\boldsymbol{\beta}) =\sum_{i=1}^{m}\left[y_i\log{(p_i)}+(1 - y_i)\log{(1 - p_i)}\right]$
		
	- Each data point contributes to the overall score depending on how confident and correct the model was:
		- If the model correctly predicts a high pip_ipi​ for a positive sample, log(pi) is large → high reward.
		- If it predicts a low pip_ipi​ for a positive sample, log(pi) is very negative → strong penalty.
		That’s why the model “learns” by maximizing this function — it’s literally rewarding correct confidence and punishing confident mistakes.
	- The optimization goal is:
		- $max_{\beta} \ell(\boldsymbol{\beta})$
		or equivalently (since maximization and minimization are mirror problems):
		- $max_{\beta} -\ell(\boldsymbol{\beta})$

	- The negative log-likelihood is what’s commonly known as the Binary Cross-Entropy Loss or Log Loss:
		- $\ell(\boldsymbol{\beta}) = -\frac{1}{m} \sum_{i=1}^{m}\left[y_i\log{(p_i)}+(1 - y_i)\log{(1 - p_i)}\right]$
	
	- Log Loss measures how uncertain or wrong the model is.
	- A perfect model has Log Loss = 0.
	- As predictions become uncertain (closer to 0.5), the penalty increases sharply — reflecting the cost of indecision.
	
	- For Learning Mechanism:
		- we use Gradient Descent, an iterative optimization algorithm.
		- Over time, the model “descends” to the set of parameters that minimize the loss — i.e., maximize the likelihood.
	- When features are many or correlated, Logistic Regression can overfit — memorizing patterns that don’t generalize.
		- To fix that, we add regularization, a penalty term that discourages overly large coefficients.
		- L2 Regularization (Ridge Regression)
		- L1 Regualrization (Lasso Regression)
		


- At a High Level:
	- Logistic Regression tries to find parameters that maximize the likelihood of seeing your data.
	- The math turns that problem into minimizing Log Loss.
	- Gradient Descent adjusts parameters step by step to find the best fit.
	- Regularization keeps the model from memorizing noise.


*Explain the Common Evaluation Metrics Used for Classification*

- There are different key evaluation metrics that tell the true story
of model's performance.

- **The Confusion Matrix**
	- Every evaluation metrics stems from one simple concept - the confusion matrix
	- It compares the model's predicted labels with the actual ones.
		-			predicted:1			predicted:0
		- Actual:1		TP					FN
		- Actual:0		FP					TN
		
		- TP(True Positive): Model correctly predicts the positive class.
		- TN(True Negative): Model correctly predicts the negative class.
		- FP(False Positive): Model incorrectly predicts positive class. (model predicts postive but actual is negative)
		- FN(False Negative): Model incorrectly predicts the negative class. (model predicts negative but actual is positve)
- **Accuracy**
	- Accuracy tells you how often the model was correct overall.
	- $Accuracy = \frac{TP+TN}{TP+FP+TN+FN}$
	- **Intution**
		- Accuracy is great when both classes are equally important and balanced.
		But if the dataset is 95% "No Fraud" and 5% "Fraud", model that always predicts
		"No Fraud" will still be 95% accurate - yet completely useless.
		
- **Precision**
	- $Precision = \frac{TP}{TP+FP}$
	- Precison answers the question-Of all the positve predictions the model made,
	how many were actually correct?
	- High precision means when it says "Positive" , it is usualy right. 
	- For example: in fraud detection, high precision means you're not falsely accusing legitimate transactions.

- **Recall(Sensitivity or True Positive Rate)**
	- $Recall = \frac{TP}{TP+FN}$
	- Recall measures how many of the actual positives the model managed to catch.
	- In other words: Out of all the fraudulent transactions, how many did the model detect?
	- High recall means fewer missed positives - but possibly more false alarms.

- **F1-Score: The Balance Between Precision and Recall**
	- $Accuracy = 2 * \frac{Precision * Recall}{Precision + Recall}$
	- F1 is the harmonic mean of precision and recall.
	- It's high only when both are high - making it a balanced metric for imbalanced datasets.
	- Why use the harmonic mean instead of the arithmetic mean?
		- Harmonic mean penalizes extreme imbalances - e.g., high precision but very low recall.

- **ROC Curve and AUC (Area Under Curve)**
	- The ROC curve (Receiver Operating Characteristic curve) plots:
		- X-axis: False Positive Rate (FPR)
		- Y-axis: True Positive Rate (TPR = Recall)
		- $TPR = \frac{TP}{TP+FN}$, $FPR = \frac{FP}{FP+TN}$
	- The AUC (Area Under Curve) summarizes this curve into a single number between 0 and 1.
		- AUC = 0.5 -> model is guessing randomly.
		- AUC = 1.0 -> prefect seperation between classes.
		- AUC = 0.8 -> on average, model ranks a random positive higher than a random negative 80% of the time.
		- AUC is particularly valuable when your data is imbalanced because it measures ranking quality - not just accuracy at one threshold.
		
- **Choosing the Right Metric**
	- Scenario							Best Metric(s)							Why
	- Balanced datasets					Accuracy, F1							All classes equally important
	  Imbalancedd dataset				F1, ROC-AUC, Precision/Recall			Focus on minority class
	  High cost for false positives		Precsion 								e.g., flagging legitimate transactions
	  High cost for false negatives		Recall 									e.g., detecting cancer, fraud 
	  Probabilitic calibration matters	Log Loss								Measure how confident your model model should
	  
	 - Each metric highlights a different aspect of model performance:
		- Accuracy tells you how often you're right.
		- Precision tells you how trustworthy your positive predicitions are.
		- Recall tells you how complete your detection is.
		- F1 balances Precision and Recall
		- AUC tells you how well your model seperates classes overall.
		- Log Loss tells you how confident and calibrated your probabilities are. 
		

**Important Practise and Common pitfalls - how to avoid them.**

- Always Scale or Normalize Continuous Features
	- Standardize or normalize the features before training
		- `from sklean.preprocessing import StandardScaler
		  scaler = StandardScaler()
		  X_scaled = scaler.fit_tranform(X)`
		  
		 - Scaling ensures the optimization landscape is smooth, allowing the gradient
		 descent to converge efficiently without oscillations.
		 
- Handle Categorical Variables Properly
	- Categorical features must be encoded (like city, department, or product-type)
	- Common Encoding methods:
		- One-Hot Encoding: for nominal categories (e.g "Red", "Green", "Blue")
		- Ordinal Encoding: for ordered categories (e.g "Low", "Medium", "High")
		
	- Pitfall to avoid:
		- including all dummy varibales without dropping one can cause prefect multicollinearity - known as the dummy varibale trap
		- pd.get_dummies(df['colo'], drop_first=True)
		
- Watch Out for Multicollinearity
	- Highly correlated predictors make it difficult for the model to estimate the unique effect of each variable — leading to unstable coefficients.
	- Detect it:
		- Use the Variance Inflation Factor (VIF)
		- `from statsmodels.stats.outliers_influence import variance_inflation_factor
			vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])`
		- Fix it:
			- Drop redundant variables
			- Combine correlated features (e.g., take averages)
			- Use L2 regularization (Ridge) to shrink correlated coefficients smoothly
			- Rule of thumb: VIF > 5 (or 10) indicates serious multicollinearity.
- Regualrization
	- add a penalty to control model complexity:
		- L1 (Lasso): useful for sparse models and feature selection
		- L2 (Ridge): preferred when most features are relevant.
		
- Careful with Imbalanced Data
	- In cases like fraud detection or rare disease prediction, the positive class might be <5% of all samples. 
	A model that predicts “no fraud” every time could have 95% accuracy — but be completely useless.

	- Best practices:
		- Use stratified train-test splits so both sets preserve class balance.
		- Evaluate using Precision, Recall, F1, or ROC-AUC, not just accuracy.
		- Adjust class weights or decision thresholds.
		- Try SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic minority examples.
		





