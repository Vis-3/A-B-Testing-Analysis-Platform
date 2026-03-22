import numpy as np
import scipy.stats as stats
import statsmodels.stats.power as smp
from statsmodels.stats.proportion import proportion_effectsize

def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.8, is_proportion=True):
    '''
    Calculate sample size needed per group.
    baseline_rate: current conversion rate or mean
    mde: minimum detectable effect (absolute)
    '''
    if is_proportion:
        p1 = baseline_rate
        p2 = baseline_rate + mde
        effect_size = proportion_effectsize(p1, p2)
        n = smp.zt_ind_solve_power(effect_size=effect_size, nobs1=None, alpha=alpha, power=power, alternative='two-sided')
    else:
        # For continuous metrics, effect size is Cohen's d. Assume mde is Cohen's d here if continuous.
        n = smp.tt_ind_solve_power(effect_size=mde, nobs1=None, alpha=alpha, power=power, alternative='two-sided')
    return np.ceil(n)

def calculate_power(n_per_group, baseline_rate, mde, alpha=0.05, is_proportion=True):
    '''
    Calculate statistical power given sample size.
    '''
    if is_proportion:
        p1 = baseline_rate
        p2 = baseline_rate + mde
        effect_size = proportion_effectsize(p1, p2)
        power = smp.zt_ind_solve_power(effect_size=effect_size, nobs1=n_per_group, alpha=alpha, power=None, alternative='two-sided')
    else:
        power = smp.tt_ind_solve_power(effect_size=mde, nobs1=n_per_group, alpha=alpha, power=None, alternative='two-sided')
    return power

def mde_calculator(n_per_group, baseline_rate, alpha=0.05, power=0.8, is_proportion=True):
    '''
    Calculate minimum detectable effect given sample size.
    '''
    if is_proportion:
        # Search for MDE that gives the desired power
        mde_list = np.linspace(0.001, 0.5, 500)
        for mde in mde_list:
            p1 = baseline_rate
            p2 = baseline_rate + mde
            if p2 >= 1.0:
                continue
            effect_size = proportion_effectsize(p1, p2)
            calc_power = smp.zt_ind_solve_power(effect_size=effect_size, nobs1=n_per_group, alpha=alpha, power=None, alternative='two-sided')
            if calc_power >= power:
                return mde
        return None
    else:
        effect_size = smp.tt_ind_solve_power(effect_size=None, nobs1=n_per_group, alpha=alpha, power=power, alternative='two-sided')
        return effect_size # Returns Cohen's d

def apply_multiple_testing_correction(p_values, method='bonferroni'):
    '''
    Apply multiple testing correction.
    method: 'bonferroni' or 'fdr' (Benjamini-Hochberg)
    '''
    from statsmodels.stats.multitest import multipletests
    method_map = {'bonferroni': 'bonferroni', 'fdr': 'fdr_bh'}
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method=method_map.get(method, 'bonferroni'))
    return reject, pvals_corrected
