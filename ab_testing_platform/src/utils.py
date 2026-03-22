import numpy as np
import pandas as pd
from scipy import stats

def generate_binary_experiment_data(n_control, n_treatment, p_control, p_treatment, random_state=42):
    np.random.seed(random_state)
    control_data = np.random.binomial(1, p_control, n_control)
    treatment_data = np.random.binomial(1, p_treatment, n_treatment)
    
    df_control = pd.DataFrame({'group': 'A', 'converted': control_data})
    df_treatment = pd.DataFrame({'group': 'B', 'converted': treatment_data})
    
    return pd.concat([df_control, df_treatment], ignore_index=True)

def generate_continuous_experiment_data(n_control, n_treatment, mean_control, mean_treatment, std_dev, random_state=42):
    np.random.seed(random_state)
    control_data = np.random.normal(mean_control, std_dev, n_control)
    treatment_data = np.random.normal(mean_treatment, std_dev, n_treatment)
    
    df_control = pd.DataFrame({'group': 'A', 'value': control_data})
    df_treatment = pd.DataFrame({'group': 'B', 'value': treatment_data})
    
    return pd.concat([df_control, df_treatment], ignore_index=True)

def check_srm(n_control, n_treatment, expected_ratio=0.5):
    '''Sample Ratio Mismatch check'''
    total = n_control + n_treatment
    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)
    
    _, p_value = stats.chisquare(f_obs=[n_control, n_treatment], f_exp=[expected_control, expected_treatment])
    return p_value

def summarize_experiment(data, value_col, is_proportion=True):
    summary = data.groupby('group')[value_col].agg(['count', 'mean', 'std'])
    return summary
