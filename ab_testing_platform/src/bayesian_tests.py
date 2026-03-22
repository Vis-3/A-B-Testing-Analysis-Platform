import numpy as np
import scipy.stats as stats

def bayesian_ab_test_proportions(successes_a, total_a, successes_b, total_b, prior_alpha=1, prior_beta=1, n_samples=100000):
    '''
    Bayesian A/B test for proportions using Beta-Binomial conjugate model
    Returns posterior distributions, prob(B > A), expected loss, and credible intervals
    '''
    # Posteriors
    posterior_a = stats.beta(prior_alpha + successes_a, prior_beta + total_a - successes_a)
    posterior_b = stats.beta(prior_alpha + successes_b, prior_beta + total_b - successes_b)
    
    # Sampling
    samples_a = posterior_a.rvs(n_samples)
    samples_b = posterior_b.rvs(n_samples)
    
    # Calculate probability B > A
    prob_b_better = np.mean(samples_b > samples_a)
    
    # Expected loss
    loss_b = np.maximum(samples_a - samples_b, 0)
    expected_loss_b = np.mean(loss_b)
    
    loss_a = np.maximum(samples_b - samples_a, 0)
    expected_loss_a = np.mean(loss_a)
    
    # 95% High Density Interval (HDI) approximation
    hdi_b = np.percentile(samples_b, [2.5, 97.5])
    hdi_a = np.percentile(samples_a, [2.5, 97.5])
    
    # Relative lift
    relative_lift = (samples_b - samples_a) / samples_a
    hdi_lift = np.percentile(relative_lift, [2.5, 97.5])
    
    return {
        'prob_b_better': prob_b_better,
        'expected_loss_b': expected_loss_b,
        'expected_loss_a': expected_loss_a,
        'hdi_a': hdi_a,
        'hdi_b': hdi_b,
        'hdi_lift': hdi_lift,
        'mean_a': posterior_a.mean(),
        'mean_b': posterior_b.mean(),
        'samples_a': samples_a, # Return samples for plotting
        'samples_b': samples_b
    }

def bayesian_ab_test_continuous(data_a, data_b, n_samples=100000):
    '''
    Simple Bayesian A/B test for continuous metrics using normal approximation
    For a fully robust approach, PyMC should be used, but this works for large samples.
    '''
    n_a, n_b = len(data_a), len(data_b)
    mean_a, mean_b = np.mean(data_a), np.mean(data_b)
    var_a, var_b = np.var(data_a, ddof=1), np.var(data_b, ddof=1)
    
    # Standard errors
    se_a = np.sqrt(var_a / n_a)
    se_b = np.sqrt(var_b / n_b)
    
    # Normal posteriors
    posterior_a = stats.norm(mean_a, se_a)
    posterior_b = stats.norm(mean_b, se_b)
    
    samples_a = posterior_a.rvs(n_samples)
    samples_b = posterior_b.rvs(n_samples)
    
    prob_b_better = np.mean(samples_b > samples_a)
    
    loss_b = np.maximum(samples_a - samples_b, 0)
    expected_loss_b = np.mean(loss_b)
    
    hdi_a = np.percentile(samples_a, [2.5, 97.5])
    hdi_b = np.percentile(samples_b, [2.5, 97.5])
    
    relative_lift = (samples_b - samples_a) / samples_a
    hdi_lift = np.percentile(relative_lift, [2.5, 97.5])
    
    return {
        'prob_b_better': prob_b_better,
        'expected_loss': expected_loss_b,
        'hdi_a': hdi_a,
        'hdi_b': hdi_b,
        'hdi_lift': hdi_lift,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'samples_a': samples_a,
        'samples_b': samples_b
    }
