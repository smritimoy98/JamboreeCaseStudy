# Jamboree Graduate Admission Analysis

## Project overview

Jamboree helps students prepare for graduate study at leading universities. This project explores applicant-profile data and builds a reproducible regression workflow to estimate the dataset's **Chance of Admit** score.

The analysis is designed as an educational and decision-support baseline. The target is a supplied chance estimate, not a verified admission outcome. Predictions should not be presented as admissions guarantees or as proof that a particular applicant will be accepted.

## Business questions

- Which applicant characteristics are associated with the estimated admission chance?
- How strongly are applicant features related to one another?
- How well do regression models estimate chance for applicants held out from training?
- What additional evidence and safeguards are needed before using estimates in a student-facing product?

## Dataset

The repository includes [Jamboree.csv](Jamboree.csv), with **500 applicant records and 9 columns**.

| Feature | Description |
| --- | --- |
| Serial No. | Unique row identifier; excluded from model predictors |
| GRE Score | GRE score, out of 340 |
| TOEFL Score | TOEFL score, out of 120 |
| University Rating | University rating, from 1 to 5 |
| SOP | Statement of Purpose strength, from 1 to 5 |
| LOR | Letter of Recommendation strength, from 1 to 5 |
| CGPA | Undergraduate GPA, out of 10 |
| Research | Research experience indicator (0 or 1) |
| Chance of Admit | Estimated admission chance, from 0 to 1; prediction target |

The three rating fields are ordinal and Research is binary. The notebook keeps them numeric for the linear-regression baseline and documents the assumptions behind that choice.

## Analysis workflow

The [JamboreeProject.ipynb](JamboreeProject.ipynb) notebook is a documented walkthrough for analytics and product stakeholders.

1. **Frame the problem** — define the use case, target, and limits of the prediction.
2. **Load and validate the data** — review dimensions, schema, missing values, duplicate rows and identifiers, summary statistics, and documented ranges.
3. **Explore the data** — inspect score and target distributions, rating counts, box plots, IQR outlier flags, pairwise relationships, and a correlation heatmap.
4. **Prepare the modeling data** — remove the serial identifier, create a reproducible 80/20 train/test split, fit MinMax scaling on training data only, and reuse that fitted scaler on the test set.
5. **Fit and interpret OLS** — use Statsmodels to report fit statistics, feature coefficients, p-values, and confidence intervals.
6. **Review multicollinearity** — calculate VIF, then create a separately reported reduced model by iteratively removing the highest-VIF predictor from training data.
7. **Check regression diagnostics** — assess residual mean, functional form, heteroscedasticity, and residual normality using plots and statistical tests.
8. **Compare model performance** — calculate MAE, RMSE, R², and adjusted R² for train and test predictions.
9. **Compare regularized models** — tune Ridge and Lasso with five-fold cross-validation using pipelines that scale within each training fold.
10. **Translate analysis into recommendations** — discuss uncertainty, data needs, validation, monitoring, and responsible product use.

### Model evaluation

The notebook generates a fresh scorecard for the full OLS model, VIF-reduced OLS model, cross-validated Ridge, and cross-validated Lasso. It reports train and holdout results, including:

- **MAE:** average absolute prediction error.
- **RMSE:** square root of mean squared error; penalizes larger errors more heavily than MAE.
- **R²:** proportion of target variation explained by the fitted model.
- **Adjusted R²:** R² adjusted for the number of predictors and observations.

MAE and RMSE are expressed in Chance of Admit units. For example, an error of 0.05 is five percentage points on the 0–1 target scale. The notebook’s outputs are intentionally cleared; run the analysis from a clean kernel to generate the current plots, diagnostics, and metrics.

## Review of the earlier notebook

The prior saved outputs were internally inconsistent and are retained here only as an audit note, not as verified performance claims:

| Earlier result or issue | What was observed |
| --- | --- |
| Full OLS training fit | R² 0.830; adjusted R² 0.827 |
| Scikit-learn train/test scores | Training R² 0.8296; test R² 0.7799, but the test set was scaled using a separately fitted scaler |
| VIF-reduced OLS | Adjusted R² 0.8079, while several remaining VIFs exceeded 5 |
| Residual error metrics | MAE 0.10666; a residual MSE of 0.003753 was labeled RMSE. Its square root is about 0.0613, which cannot be lower than MAE if both describe the same residuals |
| Residual normality | Shapiro–Wilk p-value 1.0 conflicted with the full OLS summary's Jarque–Bera p-value of 5.11 × 10⁻⁵⁰ |
| Outlier handling | The earlier function calculated an upper IQR fence but applied only a lower-tail replacement, including to the target |
| Missing analysis elements | No duplicate check, numeric correlation matrix, Ridge/Lasso comparison, test-set MAE/RMSE, or formal heteroscedasticity test was reported |

The revised notebook corrects the workflow, clears those old outputs, and calculates metrics consistently from one execution. The historical values above should not be compared directly with the rerun scorecard.

## Interpretation and limitations

- Regression coefficients describe associations in the available dataset, not causal effects.
- Correlated predictors can make individual OLS coefficients unstable even when overall prediction performance is useful.
- This dataset may not represent all universities, programs, applicant cohorts, or time periods.
- Because the target is an estimated chance score rather than confirmed admission outcomes, the analysis does not establish real-world admission accuracy.
- Before product use, validate against verified outcomes, review prediction calibration and uncertainty, assess relevant group-level performance, and monitor errors over time.
- Present estimates as planning support for counseling, never as a guarantee or substitute for university decisions.

## Running the notebook

Open [JamboreeProject.ipynb](JamboreeProject.ipynb) in Jupyter or Google Colab with [Jamboree.csv](Jamboree.csv) in the same folder. Run all cells from top to bottom.

Required Python libraries: Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, SciPy, and Scikit-learn.

Install dependencies with:

```bash
python -m pip install pandas numpy matplotlib seaborn statsmodels scipy scikit-learn jupyter
```

## Repository contents

- [Jamboree.csv](Jamboree.csv) — applicant-profile dataset.
- [JamboreeProject.ipynb](JamboreeProject.ipynb) — documented EDA, regression modeling, diagnostics, and recommendations.
- [README.md](README.md) — project overview, workflow, evaluation measures, and limitations.
