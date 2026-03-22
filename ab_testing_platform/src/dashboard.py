import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

from frequentist_tests import z_test_proportions, t_test_continuous
from bayesian_tests import bayesian_ab_test_proportions
from power_analysis import calculate_sample_size, apply_multiple_testing_correction
from sequential_tests import sprt

st.set_page_config(page_title="A/B Testing Platform", layout="wide")

st.title("A/B Testing Analysis Platform")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Experiment Setup", 
    "Real-Time Monitoring", 
    "Bayesian Analysis", 
    "Multiple Test Correction",
    "Historical Results"
])

with tab1:
    st.header("Experiment Setup & Power Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        metric_type = st.radio("Metric Type", ["Proportion (e.g., Conversion Rate)", "Continuous (e.g., Revenue)"])
        is_prop = metric_type.startswith("Proportion")
        
        baseline = st.number_input("Baseline Rate / Mean", value=0.10 if is_prop else 50.0)
        mde = st.number_input("Minimum Detectable Effect (Absolute)", value=0.02 if is_prop else 2.0)
        alpha = st.slider("Significance Level (Alpha)", 0.01, 0.10, 0.05)
        power = st.slider("Statistical Power (1 - Beta)", 0.60, 0.99, 0.80)
        
    with col2:
        if st.button("Calculate Sample Size"):
            try:
                n = calculate_sample_size(baseline, mde, alpha, power, is_prop)
                st.success(f"Required Sample Size: **{int(n):,}** per group")
                st.info(f"Total Sample Size: **{int(n*2):,}**")
                
                # Visualize power curve
                mdes = np.linspace(mde*0.5, mde*2, 20)
                ns = [calculate_sample_size(baseline, m, alpha, power, is_prop) for m in mdes]
                fig = px.line(x=mdes, y=ns, labels={'x': 'Minimum Detectable Effect', 'y': 'Required Sample Size (per group)'})
                fig.update_layout(title="Sample Size vs. MDE")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error calculating: {e}")

with tab2:
    st.header("Real-Time Monitoring & Sequential Testing")
    
    col_file, col_sim = st.columns(2)
    
    with col_file:
        st.subheader("Upload Experiment Data")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:", df.head())
            group_col = st.selectbox("Select Group Column", df.columns)
            value_col = st.selectbox("Select Value Column", df.columns)
            
            n_a = len(df[df[group_col] == 'A'])
            n_b = len(df[df[group_col] == 'B'])
            st.write(f"Sample Sizes: A={n_a}, B={n_b}")
            
            from utils import check_srm
            srm_p = check_srm(n_a, n_b)
            if srm_p < 0.01:
                st.error(f"SRM Detected! (p={srm_p:.4f}). Actual ratio is too far from 50/50.")
            else:
                st.success(f"No SRM Detected (p={srm_p:.4f})")

    with col_sim:
        st.subheader("Simulate Sequential Monitoring (SPRT)")
        sim_n = st.number_input("Total Observations (per group)", 1000)
        sim_pA = st.number_input("True Conv. Rate A", 0.10, max_value=1.0)
        sim_pB = st.number_input("True Conv. Rate B", 0.12, max_value=1.0)
        sim_mde = st.number_input("Expected MDE for SPRT", 0.02)
        
        if st.button("Run Sequential Test"):
            np.random.seed(42)
            data_a = np.random.binomial(1, sim_pA, int(sim_n))
            data_b = np.random.binomial(1, sim_pB, int(sim_n))
            
            results = sprt(data_a, data_b, mde=sim_mde)
            
            st.metric("Decision", results['decision'])
            st.write(f"Samples used before decision: {results['samples_used']} (out of {int(sim_n)})")
            
            # Cumulative Log-Likelihood Ratio Plot
            # (simplified visualization of the LLR journey)
            st.write("Sequential likelihood path (conceptual)")

with tab3:
    st.header("Bayesian A/B Testing")
    b_metric = st.selectbox("Bayesian Metric", ["Proportions (Binary)", "Means (Continuous)"])
    
    if b_metric == "Proportions (Binary)":
        col3, col4 = st.columns(2)
        with col3:
            b_success_A = st.number_input("Successes A", 100, key='ba')
            b_total_A = st.number_input("Total A", 1000, key='ta')
        with col4:
            b_success_B = st.number_input("Successes B", 120, key='bb')
            b_total_B = st.number_input("Total B", 1000, key='tb')
        
        if st.button("Run Bayesian Proportion Test"):
            res = bayesian_ab_test_proportions(b_success_A, b_total_A, b_success_B, b_total_B)
            st.metric("Probability B is better than A", f"{res['prob_b_better']:.2%}")
            st.metric("Expected Loss (if choose B)", f"{res['expected_loss_b']:.4f}")
            
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=res['samples_a'][:10000], name='Group A', histnorm='probability density', opacity=0.7))
            fig2.add_trace(go.Histogram(x=res['samples_b'][:10000], name='Group B', histnorm='probability density', opacity=0.7))
            fig2.update_layout(barmode='overlay', title="Posterior Distributions")
            st.plotly_chart(fig2)
    else:
        from bayesian_tests import bayesian_ab_test_continuous
        st.write("Enter comma-separated values for A and B")
        vals_a_str = st.text_area("Values A", "50, 52, 48, 55, 50, 49")
        vals_b_str = st.text_area("Values B", "52, 54, 51, 58, 53, 52")
        
        if st.button("Run Bayesian Continuous Test"):
            vals_a = [float(x.strip()) for x in vals_a_str.split(",")]
            vals_b = [float(x.strip()) for x in vals_b_str.split(",")]
            res = bayesian_ab_test_continuous(vals_a, vals_b)
            
            st.metric("Probability B is better than A", f"{res['prob_b_better']:.2%}")
            st.write(f"95% Credible Interval for B: [{res['hdi_b'][0]:.2f}, {res['hdi_b'][1]:.2f}]")
            
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=res['samples_a'][:10000], name='Group A', histnorm='probability density', opacity=0.7))
            fig3.add_trace(go.Histogram(x=res['samples_b'][:10000], name='Group B', histnorm='probability density', opacity=0.7))
            fig3.update_layout(barmode='overlay', title="Posterior Distributions (Means)")
            st.plotly_chart(fig3)

with tab4:
    st.header("Multiple Test Correction")
    st.write("When running many tests, false positives accumulate.")
    
    p_vals = st.text_area("Enter p-values separated by comma", "0.01, 0.04, 0.06, 0.20, 0.001")
    method = st.selectbox("Correction Method", ["bonferroni", "fdr"])
    
    if st.button("Apply Correction"):
        p_list = [float(p.strip()) for p in p_vals.split(",")]
        reject, corrected = apply_multiple_testing_correction(p_list, method=method)
        
        df_corr = pd.DataFrame({
            "Original p-value": p_list,
            "Corrected p-value": corrected,
            "Significant (alpha=0.05)": reject
        })
        st.dataframe(df_corr)

with tab5:
    st.header("Historical Results & Meta-Analysis")
    st.write("This tab would normally connect to an experiments database.")
    
    # Mock data
    hist_data = pd.DataFrame({
        "Experiment": ["Exp 1: Button Color", "Exp 2: Pricing", "Exp 3: Flow", "Exp 4: Copy", "Exp 5: Img"],
        "Metric": ["Conv", "Rev/User", "Drop-off", "CTR", "Conv"],
        "Lift": ["+2.1%", "+0.5%", "-1.2%", "+5.4%", "+0.1%"],
        "Significant": ["Yes", "No", "No", "Yes", "No"]
    })
    st.dataframe(hist_data)
