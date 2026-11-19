# Jamboree Graduate Admission Analysis

## Project overview

Jamboree supports students preparing applications to graduate programs. This project explores applicant-profile data and builds a reproducible regression workflow to estimate the dataset's Chance of Admit score.

The analysis is intended as a learning and decision-support baseline. The target is a supplied chance estimate, not a verified admission decision. Predictions should not be presented as admission guarantees or as evidence that a particular applicant will be accepted.

## Business questions

- Which applicant characteristics are associated with the estimated admission chance?
- How strongly are the applicant features related to one another?
- How well do regression models estimate chance for held-out records?
- What data and validation would be needed before a student-facing rollout?

## Dataset

The repository includes Jamboree.csv with **500 records and 9 columns**.

| Field | Description |
| --- | --- |
| Serial No. | Unique row identifier; excluded from modeling |
| GRE Score | GRE score, out of 340 |
| TOEFL Score | TOEFL score, out of 120 |
| University Rating | University rating, from 1 to 5 |
| SOP | Statement of Purpose strength, from 1 to 5 |
| LOR | Letter of Recommendation strength, from 1 to 5 |
| CGPA | Undergraduate GPA, out of 10 |
| Research | Research experience indicator (0 or 1) |
| Chance of Admit | Estimated admission chance, from 0 to 1; modeling target |

## Notebook workflow

JamboreeProject.ipynb provides a stakeholder-oriented, step-by-step analysis:

1. **Business framing and data definitions** — describes the product question, target, and limits of interpretation.
2. **Data loading and quality checks** — inspects schema, missingness, duplicates, identifiers, and documented value ranges.
3. **Exploratory analysis** — reviews score and target distributions, ordinal feature counts, box plots, outlier flags, a correlation heatmap, and pairwise relationships.
4. **Data preparation** — removes the serial identifier, creates a reproducible 80/20 train/test split, and fits scaling on training data only before transforming the holdout set.
5. **OLS modeling** — fits Statsmodels Ordinary Least Squares and presents model statistics, coefficients, p-values, and confidence intervals.
6. **Multicollinearity** — calculates VIF and reports both the full model and a transparent train-only VIF-reduced model.
7. **Regression diagnostics** — reviews residual mean, residual-versus-fitted and Q–Q plots, Ramsey RESET, Breusch–Pagan, Shapiro–Wilk, and Jarque–Bera diagnostics.
8. **Model evaluation** — reports MAE, RMSE, R², and adjusted R² on both train and test splits, with RMSE computed as the square root of mean squared error.
9. **Regularized comparisons** — tunes Ridge and Lasso with five-fold cross-validation using pipelines that scale within each fold.
10. **Interpretation and recommendations** — discusses model use, data gaps, uncertainty, monitoring, and stakeholder safeguards.

## Methodological improvements

The notebook was rebuilt after review of the earlier analysis. It addresses these issues:

- Test data is transformed with the scaler fitted on training data, preventing inconsistent train/test feature scales.
- Duplicate checks and range validations are included.
- Outliers are flagged for review instead of automatically modifying only the lower tail or changing the target.
- A numeric correlation matrix and heatmap are added.
- VIF reduction removes the current highest-VIF feature at each step and reports whether the threshold was reached; full and reduced models remain available for comparison.
- Residual metrics are calculated consistently from predictions. MAE, RMSE, R², and adjusted R² are reported for train and test data.
- Ridge and Lasso are included as regularized comparisons.
- Formal functional-form and heteroscedasticity screens complement visual residual diagnostics.

The earlier notebook contained saved outputs that conflicted with one another, including an MSE labeled as RMSE and a residual MAE that did not agree with the reported residual MSE. Those outputs were cleared in the revised notebook rather than presented as validated final results. **Run all notebook cells from a clean kernel to generate current findings and metrics.**

## Interpretation and limitations

- Regression coefficients describe associations in this dataset, not causal effects.
- Correlated features can make individual OLS coefficients unstable even when overall prediction is useful.
- A small dataset may not represent all programs, universities, applicant cohorts, or time periods.
- The target is an estimated chance score; confirmed admission outcomes and university/program context would be needed to validate actual admission predictions.
- Before product use, review calibration, uncertainty, group-level performance, and error consequences. Present any estimate as support for counseling, not as a guarantee or a substitute for institutional decisions.

## Running the notebook

Open JamboreeProject.ipynb in Jupyter or Google Colab with Jamboree.csv in the same folder. The notebook uses Python with Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, SciPy, and Scikit-learn. Run cells from top to bottom for reproducible tables, plots, diagnostics, and model metrics.

## Repository files

- Jamboree.csv — applicant-profile dataset
- JamboreeProject.ipynb — documented analysis and modeling workflow
- README.md — project overview, methodology, and interpretation guidance
