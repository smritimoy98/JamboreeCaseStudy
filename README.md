# Jamboree Graduate Admission Prediction

## Project Overview

Jamboree supports students preparing for graduate study at leading universities. This project analyzes applicant profiles to understand which academic and application factors are associated with admission chances and to build a model that estimates a candidate’s **Chance of Admit**.

The work uses exploratory data analysis, multiple linear regression, regression diagnostics, and a scikit-learn train/test comparison. It is intended as an analytical decision-support exercise: a predicted chance is an estimate based on historical patterns, not a guarantee or an admissions decision.

## Business Questions

- Which applicant characteristics have the strongest relationship with estimated admission chance?
- How do the applicant features relate to one another?
- How well can a linear regression model predict admission chance on held-out data?
- What should be improved before this model is used in a student-facing product?

## Dataset

The repository contains `Jamboree.csv`, with **500 applicant records and 9 columns**. All columns were read as numeric; the notebook reports no missing values.

| Field | Meaning |
| --- | --- |
| `Serial No.` | Unique row identifier; excluded from model features |
| `GRE Score` | GRE score (out of 340) |
| `TOEFL Score` | TOEFL score (out of 120) |
| `University Rating` | University rating (1–5) |
| `SOP` | Statement of Purpose strength (1–5) |
| `LOR` | Letter of Recommendation strength (1–5) |
| `CGPA` | Undergraduate GPA (out of 10) |
| `Research` | Research experience indicator (0 or 1) |
| `Chance of Admit` | Estimated admission chance (0–1); prediction target |

The data contains no categorical text fields. The notebook retains the ordinal ratings and binary research indicator as numeric features.

## Analysis Workflow

### 1. Data inspection and exploratory analysis

The notebook loads the CSV, inspects the first records, column names, data types, null counts, and descriptive statistics. It finds **500 non-null values in every column**. The target has a mean of **0.7217**, a standard deviation of **0.1411**, and a range of **0.34–0.97**.

Histograms and count plots are used to review distributions; box plots are used to inspect potential outliers; and a pair plot is used to explore pairwise relationships among the applicant variables. The notebook does not calculate or display a numeric correlation matrix or correlation heatmap.

### 2. Outlier handling

A custom function calculates IQR bounds and is applied to `LOR` and `Chance of Admit`. In the code, only values below the calculated lower bound are replaced; the calculated upper bound is not applied. This is therefore a lower-tail adjustment, rather than a complete two-sided outlier treatment.

### 3. Train/test preparation

The unique serial number is removed. The remaining seven applicant fields are used as predictors, and `Chance of Admit` is the target. The data is split into **80% training (400 rows)** and **20% testing (100 rows)** with `random_state=2`.

The notebook applies MinMax scaling. It fits one scaler on the training set, but then calls `fit_transform` separately on the test set. For a valid holdout evaluation, the test set should instead be transformed with the scaler fitted on the training data. The saved test score should therefore be treated as provisional and recalculated after correcting this preprocessing step.

### 4. Statsmodels OLS regression

An Ordinary Least Squares (OLS) model is fitted with Statsmodels using all seven scaled predictors. The saved full-model summary reports:

| Statistic | Saved result |
| --- | ---: |
| Training observations | 400 |
| R² | 0.830 |
| Adjusted R² | 0.827 |
| F-statistic | 272.6 |
| F-statistic p-value | 2.51 × 10⁻¹⁴⁶ |

The overall regression is statistically significant in the saved output. The OLS coefficients are based on MinMax-scaled features, so each coefficient represents the modeled change associated with moving a predictor across its training-set scale, holding the other predictors constant.

| Predictor | Coefficient | p-value | Interpretation in the saved full model |
| --- | ---: | ---: | --- |
| GRE Score | 0.1001 | <0.001 | Positive association; statistically significant |
| TOEFL Score | 0.0797 | 0.003 | Positive association; statistically significant |
| University Rating | 0.0193 | 0.237 | Not statistically significant at 0.05 |
| SOP | 0.0088 | 0.652 | Not statistically significant at 0.05 |
| LOR | 0.0650 | <0.001 | Positive association; statistically significant |
| CGPA | 0.3538 | <0.001 | Largest positive coefficient; statistically significant |
| Research | 0.0247 | 0.001 | Positive association; statistically significant |

These are associations in this dataset, not evidence that changing a score will cause a particular admissions outcome. The high multicollinearity described below also makes individual coefficient estimates less stable.

### 5. Multicollinearity review

The initial VIF results show substantial overlap among several academic and application features:

| Predictor | Initial VIF |
| --- | ---: |
| CGPA | 36.46 |
| TOEFL Score | 26.32 |
| GRE Score | 23.47 |
| SOP | 17.95 |
| LOR | 11.31 |
| University Rating | 10.96 |
| Research | 3.31 |

The notebook then removes GRE Score and TOEFL Score and refits a reduced model. The saved adjusted R² falls to **0.8079**. The resulting VIFs are **CGPA 18.70, SOP 17.61, LOR 11.22, University Rating 10.73, and Research 3.02**. Thus, the saved iteration does **not** reach the stated VIF ≤ 5 criterion; five predictors remain above 5. The code also stops when adjusted R² falls below 0.82, which prevents it from continuing until all VIFs meet the threshold.

### 6. Regression diagnostics and saved metrics

The notebook performs residual checks on the reduced OLS model and records the following:

| Check / metric | Saved result | Notes |
| --- | ---: | --- |
| Mean residual | -0.00413 | Close to zero |
| OLS R² | 0.8103 | Reduced model, training data |
| OLS adjusted R² | 0.8079 | Reduced model, training data |
| `results.mse_resid` | 0.003753 | This is residual mean squared error, not RMSE |
| RMSE implied by that MSE | ≈0.0613 | Square root of 0.003753 |
| MAE | 0.10666 | Saved residual-based calculation; conflicts with the MSE-derived RMSE |

The notebook plots predicted versus actual values, residuals versus fitted values, a residual histogram, and a Q–Q plot. It also reports a Shapiro–Wilk p-value of **1.0**. However, the saved full-model OLS summary reports Omnibus p-value **0.000** and Jarque–Bera p-value **5.11 × 10⁻⁵⁰**, which points to non-normal residuals for that model. The MAE is also greater than the RMSE implied by the saved residual MSE, which cannot occur when both metrics are calculated from the same residuals. These conflicts suggest that some notebook outputs may be stale or come from different execution states. Rerun the notebook from top to bottom before treating these residual metrics or normality conclusions as final.

The residual plot is used for visual inspection of linearity and homoscedasticity, but the notebook does not report a formal homoscedasticity test. The mean residual check is close to zero; it does not by itself establish that the other regression assumptions hold.

### 7. Scikit-learn Linear Regression

A separate scikit-learn LinearRegression model is trained with all seven features. The saved scores are:

| Split | R² |
| --- | ---: |
| Training | 0.8296 |
| Testing | 0.7799 |

The test score is about 0.05 below the training score, indicating weaker performance on the holdout data. Because the test data was scaled with a separately fitted scaler, this test R² needs to be recalculated using the training-fitted scaler before drawing a firm conclusion.

The notebook does not calculate MAE or RMSE on the test set. It also does not implement Ridge or Lasso, so there is no regularized-model comparison in the current analysis.

## Key Findings

- **CGPA** has the largest positive coefficient in the saved full OLS model and is statistically significant, followed by GRE and TOEFL scores among the examination measures.
- **Research experience** and **LOR strength** also have positive, statistically significant coefficients in the saved full model.
- **SOP strength** and **University Rating** are not statistically significant at the 0.05 level in that full model, after controlling for the other included features.
- Several predictors are strongly collinear, so coefficient size and individual significance should be interpreted cautiously.
- The saved scikit-learn holdout R² is **0.7799**, but the scaling procedure must be corrected before this is used as a reliable generalization estimate.

## Recommendations

1. **Correct preprocessing and rerun the analysis.** Fit MinMaxScaler on the training data only, then use that fitted scaler to transform the test data.
2. **Recalculate evaluation metrics consistently.** Report MAE, RMSE, R², and adjusted R² for train and test predictions from the same selected model. Compute RMSE as the square root of mean squared error.
3. **Resolve the VIF/model-selection trade-off.** Investigate correlated features and use a transparent selection strategy. Consider Ridge regression to retain useful correlated predictors while reducing coefficient instability; compare Lasso as a feature-selection baseline.
4. **Recheck regression assumptions using a single clean run.** Review residual plots and Q–Q plots for the final model and add a formal heteroscedasticity test if required.
5. **Improve the data before real-world use.** Validate on newer and broader applicant cohorts, and consider features such as intended program, target university, work experience, applicant cohort, and actual admission outcomes.
6. **Use predictions responsibly.** Present estimates with uncertainty and explain that they support student planning rather than guarantee admission or replace university decisions. Monitor errors and model performance over time and across applicant groups.

## Tools and Concepts

- Python, Pandas, NumPy
- Matplotlib and Seaborn
- Statsmodels OLS
- Scikit-learn train/test split, MinMaxScaler, and LinearRegression
- Exploratory Data Analysis, VIF, residual analysis, MAE, RMSE, R², and adjusted R²

## Repository Files

- `Jamboree.csv` — applicant dataset
- `JamboreeProject.ipynb` — exploratory analysis, regression modeling, and diagnostics
- `README.md` — project overview, workflow, saved results, and interpretation
