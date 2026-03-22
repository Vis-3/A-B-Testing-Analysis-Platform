import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

def z_test_proportions(successes_a, total_a, successes_b, total_b, alpha=0.05):
    '''
    Two-sample z-test for proportions
    Returns test statistic, p-value, confidence interval, and difference in proportions
    '''
    counts = np.array([successes_b, successes_a])
    nobs = np.array([total_b, total_a])
    
    stat, pval = proportions_ztest(counts, nobs)
    
    # Confidence intervals
    (lower_b, lower_a), (upper_b, upper_a) = proportion_confint(counts, nobs, alpha=alpha)
    
    diff = (successes_b / total_b) - (successes_a / total_a)
    
    return {
        'z_statistic': stat,
        'p_value': pval,
        'diff_proportions': diff,
        'ci_lower': diff - (upper_b - lower_b), # approximate CI for difference
        'ci_upper': diff + (upper_b - lower_b),
        'significant': pval < alpha
    }

def t_test_continuous(data_a, data_b, alpha=0.05, equal_var=False):
    '''
    T-test for continuous metrics (default Welch's t-test)
    '''
    # Check normality assumption
    _, norm_a_p = stats.shapiro(data_a)
    _, norm_b_p = stats.shapiro(data_b)
    
    # Check variance equality assumption
    _, var_p = stats.levene(data_a, data_b)
    
    if var_p < 0.05 and equal_var:
        equal_var = False # Override if variances clearly unequal
        
    stat, pval = stats.ttest_ind(data_b, data_a, equal_var=equal_var)
    
    # Calculate effect size (Cohen's d)
    n1, n2 = len(data_b), len(data_a)
    var1, var2 = np.var(data_b, ddof=1), np.var(data_a, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(data_b) - np.mean(data_a)) / pooled_se if pooled_se > 0 else 0
    
    # Calculate CI for difference
    se_diff = np.sqrt(var1/n1 + var2/n2) if not equal_var else pooled_se * np.sqrt(1/n1 + 1/n2)
    df = n1 + n2 - 2 if equal_var else (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    diff = np.mean(data_b) - np.mean(data_a)
    ci_lower = diff - t_crit * se_diff
    ci_upper = diff + t_crit * se_diff
    
    return {
        't_statistic': stat,
        'p_value': pval,
        'diff_means': diff,
        'cohens_d': cohens_d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': pval < alpha,
        'normality_passed': (norm_a_p > 0.05) and (norm_b_p > 0.05),
        'equal_variance_passed': var_p > 0.05
    }

def chi_square_test(contingency_table, alpha=0.05):
    '''
    Chi-square test of independence
    '''
    stat, pval, dof, expected = stats.chi2_contingency(contingency_table)
    return {
        'chi2_statistic': stat,
        'p_value': pval,
        'dof': dof,
        'significant': pval < alpha
    }
