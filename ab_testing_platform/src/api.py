from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd

from frequentist_tests import z_test_proportions, t_test_continuous
from sequential_tests import sprt
from power_analysis import calculate_sample_size
from utils import generate_binary_experiment_data, check_srm
from bayesian_tests import bayesian_ab_test_proportions, bayesian_ab_test_continuous

app = FastAPI(title="A/B Testing Platform API", version="1.0.0")

class ExperimentData(BaseModel):
    data_a: List[float]
    data_b: List[float]
    metric_type: str = "proportion" # or "continuous"
    test_type: str = "frequentist" # or "bayesian"

class SampleSizeRequest(BaseModel):
    baseline_rate: float
    mde: float
    alpha: float = 0.05
    power: float = 0.8
    is_proportion: bool = True

class SequentialCheckRequest(BaseModel):
    data_a: List[float]
    data_b: List[float]
    mde: float
    alpha: float = 0.05
    beta: float = 0.20

class SRMCheckRequest(BaseModel):
    n_a: int
    n_b: int
    expected_ratio: float = 0.5

@app.post("/api/experiments/analyze")
def analyze_experiment(request: ExperimentData):
    if request.metric_type == "proportion":
        successes_a = sum(request.data_a)
        total_a = len(request.data_a)
        successes_b = sum(request.data_b)
        total_b = len(request.data_b)
        
        if request.test_type == "frequentist":
            results = z_test_proportions(successes_a, total_a, successes_b, total_b)
            return {"status": "success", "results": results}
        elif request.test_type == "bayesian":
            results = bayesian_ab_test_proportions(successes_a, total_a, successes_b, total_b)
            # Filter out large sample arrays from JSON response
            resp = {k: v for k, v in results.items() if k not in ['samples_a', 'samples_b']}
            return {"status": "success", "results": resp}
        else:
            raise HTTPException(status_code=400, detail="Invalid test type")
            
    elif request.metric_type == "continuous":
        if request.test_type == "frequentist":
            results = t_test_continuous(request.data_a, request.data_b)
            return {"status": "success", "results": results}
        elif request.test_type == "bayesian":
            results = bayesian_ab_test_continuous(request.data_a, request.data_b)
            resp = {k: v for k, v in results.items() if k not in ['samples_a', 'samples_b']}
            return {"status": "success", "results": resp}
        else:
            raise HTTPException(status_code=400, detail="Invalid test type")
    else:
        raise HTTPException(status_code=400, detail="Invalid metric type")

@app.post("/api/experiments/srm-check")
def srm_check(request: SRMCheckRequest):
    p_value = check_srm(request.n_a, request.n_b, request.expected_ratio)
    return {"status": "success", "p_value": p_value, "srm_detected": p_value < 0.01}

@app.post("/api/experiments/sample-size")
def get_sample_size(request: SampleSizeRequest):
    try:
        n = calculate_sample_size(
            baseline_rate=request.baseline_rate, 
            mde=request.mde, 
            alpha=request.alpha, 
            power=request.power, 
            is_proportion=request.is_proportion
        )
        return {"status": "success", "sample_size_per_group": int(n)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/experiments/sequential-check")
def check_sequential(request: SequentialCheckRequest):
    results = sprt(request.data_a, request.data_b, alpha=request.alpha, beta=request.beta, mde=request.mde)
    return {"status": "success", "results": results}

@app.get("/api/experiments/simulation")
def get_simulation(n_control: int = 1000, n_treatment: int = 1000, p_control: float = 0.1, p_treatment: float = 0.12):
    data = generate_binary_experiment_data(n_control, n_treatment, p_control, p_treatment)
    return {"status": "success", "data_sample": data.head(5).to_dict()}
