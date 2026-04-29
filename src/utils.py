import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from typing import Union
from itertools import combinations

from scipy import stats
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV

# This function fits a Partial Least Squares (PLS) model to extract the primary latent component 
# and returns the underlying projection weights for both data modalities.
# Author: Antonio Scardace

def fit_pls(X: np.ndarray, Y: np.ndarray, n_components: int) -> tuple[PLSSVD, np.ndarray, np.ndarray]:
    pls = PLSSVD(n_components=n_components).fit(X, Y)
    return pls, pls.x_weights_[:, 0], pls.y_weights_[:, 0]

# This function computes the statistical residuals for multiple target variables after regressing 
# out the provided covariates, returning them as standardized Z-scores.
# Author: Antonio Scardace

def standardized_residuals(df: pd.DataFrame, y_cols: list[str], covariates: pd.DataFrame, add_constant: bool) -> pd.DataFrame:
    X = sm.add_constant(covariates) if add_constant else covariates.copy()
    res = df[y_cols].apply(lambda y: sm.OLS(y, X).fit().resid)
    return (res - res.mean()) / res.std()

# This function computes the Spearman rank correlation between a given feature and the 
# patient's age, automatically ignoring any missing values in the data.
# Author: Antonio Scardace

def spearman_age(df: pd.DataFrame, feature: str) -> tuple[str, float, float]:
    clean = df[['age', feature]].dropna()
    rho, p = stats.spearmanr(clean['age'], clean[feature])
    return feature, round(rho, 3), round(p, 5)

# This function calculates the average Mutual Information score to quantify the non-linear 
# dependencies between a set of input predictors and multiple target variables.
# Author: Antonio Scardace

def calculate_mi(X_data: Union[pd.Series, pd.DataFrame], Y_data: Union[pd.Series, pd.DataFrame]) -> float:
    X = X_data.to_numpy().reshape(-1, 1) if isinstance(X_data, pd.Series) else X_data.to_numpy()
    Y = Y_data.to_numpy().reshape(-1, 1) if isinstance(Y_data, pd.Series) else Y_data.to_numpy()
    mi_scores = [mutual_info_regression(X, Y[:, i], random_state=42) for i in range(Y.shape[1])]
    return np.mean(mi_scores).item()

# This function removes the effect of confounding covariates from both tau and volumetric data 
# using linear regression, returning their standardized residuals.
# Author: Antonio Scardace

def get_residuals_scaled(df: pd.DataFrame, tau_cols: list[str], vol_cols: list[str], covariates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    target_cols = tau_cols + vol_cols
    res = df[target_cols] - LinearRegression().fit(covariates, df[target_cols]).predict(covariates)
    scaler = StandardScaler()
    return scaler.fit_transform(res[tau_cols]), scaler.fit_transform(res[vol_cols])

# This function evaluates the predictive power of a Ridge regression model through cross-validation, 
# returning the optimal regularization penalty and the overall R-squared score.
# Author: Antonio Scardace

def calculate_multivariate_r2(X_data: Union[pd.Series, pd.DataFrame], Y_data: Union[pd.Series, pd.DataFrame]) -> tuple[float, float]:
    X_arr = X_data.values.reshape(-1, 1) if isinstance(X_data, pd.Series) else X_data.values
    Y_arr = Y_data.values.reshape(-1, 1) if isinstance(Y_data, pd.Series) else Y_data.values
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 100)))
    Y_pred = cross_val_predict(model, X_arr, Y_arr, cv=cv)
    model.fit(X_arr, Y_arr)
    return model.named_steps['ridgecv'].alpha_, max(0.0, r2_score(Y_arr, Y_pred, multioutput='uniform_average'))

# This function performs a statistical mediation analysis using OLS regression to understand how 
# brain atrophy mediates the effect of tau pathology on cognitive decline.
# Author: Antonio Scardace

def get_mediation_stats_controlled(data: pd.DataFrame) -> tuple[float, float, float]:
    X_a = sm.add_constant(data[['tau_idx', 'age', 'sex']])
    X_yc = sm.add_constant(data[['tau_idx', 'atrophy_idx', 'age', 'sex']])
    model_a = sm.OLS(data['atrophy_idx'], X_a).fit()
    model_yc = sm.OLS(data['cdr_sb'], X_yc).fit()
    c_prime = model_yc.params['tau_idx']
    indirect = model_a.params['tau_idx'] * model_yc.params['atrophy_idx']
    return indirect, c_prime, c_prime + indirect

# This function evaluates how well a cognitive metric and hippocampus volume can distinguish 
# between different patient diagnoses using a cross-validated logistic regression.
# Author: Antonio Scardace

def calculate_diagnostic_separability(df_input: pd.DataFrame, cognitive_metric: str) -> None:
    features = ['hippocampus_volume', cognitive_metric]
    df_clean = df_input.dropna(subset=['diagnosis'] + features)
    clf = LogisticRegression(class_weight='balanced', random_state=42)
    
    for d1, d2 in combinations(df_clean['diagnosis'].unique(), 2):
        pair = df_clean[df_clean['diagnosis'].isin([d1, d2])]
        X, y = pair[features], (pair['diagnosis'] == d1).astype(int)
        auc = roc_auc_score(y, clf.fit(X, y).predict_proba(X)[:, 1])
        print(cognitive_metric, '|', d1, 'vs', d2, '=', max(auc, 1.0 - auc))

# This function generates a horizontal grid of histograms with kernel density estimation to 
# visually inspect the distribution of multiple variables side-by-side.
# Author: Antonio Scardace

def plot_histograms(df: pd.DataFrame, columns: list[str], titles: list[str], xlabel: str, bins: int, figsize: tuple, palette: str, binrange: tuple = None, sharex: bool = False, sharey: bool = True) -> None:
    
    _, axes = plt.subplots(1, len(columns), figsize=figsize, sharex=sharex, sharey=sharey)
    for ax, col, title, color in zip(axes, columns, titles, sns.color_palette(palette, len(columns))):
        sns.histplot(data=df, x=col, kde=True, bins=bins, binrange=binrange, ax=ax, color=color)
        ax.set_title(title, fontsize=11, pad=10, weight='bold')
        ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.3)

    axes[0].set_ylabel('Count', fontsize=11, labelpad=10)
    plt.tight_layout()
    plt.show()

# This function uses bootstrap resampling to estimate the stability and statistical significance 
# of the cross-modal interactions between the PLS weights.
# Author: Antonio Scardace

def bootstrap_pls_interactions(X: np.ndarray, Y: np.ndarray, w_x_original: np.ndarray, n_boot: int, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(random_state)
    boot_x, boot_y = np.empty((n_boot, X.shape[1])), np.empty((n_boot, Y.shape[1]))

    for i in range(n_boot):
        X_b, Y_b = resample(X, Y)
        pls_b = PLSSVD(n_components=1).fit(X_b, Y_b)
        sign = np.sign(np.dot(pls_b.x_weights_[:, 0], w_x_original))
        boot_x[i], boot_y[i] = sign * pls_b.x_weights_[:, 0], sign * pls_b.y_weights_[:, 0]
    
    boot_interactions = boot_x[:, :, np.newaxis] * boot_y[:, np.newaxis, :]
    ci_low = np.percentile(boot_interactions, 2.5, axis=0)
    ci_high = np.percentile(boot_interactions, 97.5, axis=0)
    return boot_interactions.mean(axis=0), ci_low, ci_high, (ci_low > 0) | (ci_high < 0)