import numpy as np
import scipy.stats as stats

def sprt(data_a, data_b, alpha=0.05, beta=0.20, mde=0.05):
    '''
    Sequential Probability Ratio Test (SPRT)
    For binomial data (proportions)
    Returns: decision (Continue, Accept H1, Accept H0), likelihood ratio, boundaries
    '''
    # Assume data_a, data_b are arrays of 1s and 0s
    p0 = np.mean(data_a)  # Baseline probability
    p1 = p0 + mde         # Alternative probability
    
    # Calculate boundaries
    A = (1 - beta) / alpha
    B = beta / (1 - alpha)
    
    log_A = np.log(A)
    log_B = np.log(B)
    
    # Calculate cumulative log-likelihood ratio
    llr = 0
    decision = 'Continue'
    
    n = min(len(data_a), len(data_b))
    
    for i in range(n):
        x = data_b[i]  # Observation from treatment group
        
        # Log-likelihood ratio for Bernoulli distribution
        if x == 1:
            llr += np.log(p1 / p0)
        else:
            llr += np.log((1 - p1) / (1 - p0))
            
        if llr >= log_A:
            decision = 'Accept H1'
            break
        elif llr <= log_B:
            decision = 'Accept H0'
            break
            
    return {
        'decision': decision,
        'llr': llr,
        'lower_bound': log_B,
        'upper_bound': log_A,
        'samples_used': i + 1
    }

def group_sequential_test(p_values, alpha=0.05, spending_function='obrien_fleming'):
    '''
    Group sequential testing using spending functions.
    p_values: list of p-values computed at different interim stages
    spending_function: 'obrien_fleming' or 'pocock'
    '''
    k = len(p_values)
    
    if spending_function == 'obrien_fleming':
        # O'Brien-Fleming boundaries are very conservative early on
        # Approximation for k stages
        z_alpha = stats.norm.ppf(1 - alpha/2)
        boundaries = [z_alpha * np.sqrt(k / (i + 1)) for i in range(k)]
        alpha_boundaries = [2 * (1 - stats.norm.cdf(b)) for b in boundaries]
    elif spending_function == 'pocock':
        # Pocock uses constant boundaries
        # Approximation
        c_p = {1: 1.96, 2: 2.178, 3: 2.289, 4: 2.361, 5: 2.413}
        z_bound = c_p.get(k, 2.5) # Fallback
        boundaries = [z_bound] * k
        alpha_boundaries = [2 * (1 - stats.norm.cdf(b)) for b in boundaries]
    else:
        raise ValueError('Unsupported spending function')
        
    decisions = []
    for i in range(k):
        if p_values[i] <= alpha_boundaries[i]:
            decisions.append('Stop - Reject H0')
        else:
            decisions.append('Continue')
            
    return {
        'alpha_boundaries': alpha_boundaries,
        'decisions': decisions
    }
