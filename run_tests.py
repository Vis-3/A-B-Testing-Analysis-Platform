import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "ab_testing_platform/src"))

from frequentist_tests import z_test_proportions, t_test_continuous
from bayesian_tests import bayesian_ab_test_proportions, bayesian_ab_test_continuous
from sequential_tests import sprt
from power_analysis import calculate_sample_size

def test_frequentist():
    print("Testing Frequentist Z-Test...")
    res = z_test_proportions(100, 1000, 130, 1000)
    print(f"Z-Test Result: Significant={res['significant']}, P-value={res['p_value']:.4f}")
    assert res['significant'] == True
    
    print("Testing Frequentist T-Test...")
    data_a = np.random.normal(50, 10, 1000)
    data_b = np.random.normal(52, 10, 1000)
    res_t = t_test_continuous(data_a, data_b)
    print(f"T-Test Result: Significant={res_t['significant']}, P-value={res_t['p_value']:.4f}")

def test_bayesian():
    print("Testing Bayesian Proportions...")
    res = bayesian_ab_test_proportions(100, 1000, 130, 1000)
    print(f"Bayesian Proportion Result: Prob B > A = {res['prob_b_better']:.4f}")
    assert res['prob_b_better'] > 0.95
    
    print("Testing Bayesian Continuous...")
    data_a = np.random.normal(50, 5, 100)
    data_b = np.random.normal(55, 5, 100)
    res_c = bayesian_ab_test_continuous(data_a, data_b)
    print(f"Bayesian Continuous Result: Prob B > A = {res_c['prob_b_better']:.4f}")
    assert res_c['prob_b_better'] > 0.95

def test_sequential():
    print("Testing SPRT...")
    np.random.seed(42)
    data_a = np.random.binomial(1, 0.10, 2000)
    data_b = np.random.binomial(1, 0.15, 2000)
    res = sprt(data_a, data_b, mde=0.05)
    print(f"SPRT Result: Decision={res['decision']}, Samples used={res['samples_used']}")
    assert res['decision'] == 'Accept H1'

def test_power():
    print("Testing Power Analysis...")
    n = calculate_sample_size(0.10, 0.02)
    print(f"Sample size for 10% baseline and 2% MDE: {n}")
    assert n > 0

if __name__ == "__main__":
    test_frequentist()
    test_bayesian()
    test_sequential()
    test_power()
    print("\nAll core statistical tests passed!")
