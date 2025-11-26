import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t # Used for T-test
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

# --- 1. CONFIGURATION ---
FILE_NAME = 'data/data3.csv'
TREATED_UNIT = 'California'
PRE_TREATMENT_END_YEAR = 2019
POST_TREATMENT_START_YEAR = 2020
MAX_NAN_TO_KEEP = 3 
TEST_YEAR_1 = 2004

# Define time periods
PRE_TREATMENT_YEARS = list(range(TEST_YEAR_1, PRE_TREATMENT_END_YEAR + 1))
POST_TREATMENT_YEARS = list(range(POST_TREATMENT_START_YEAR, 2024))
PRE_TREATMENT_YEARS_STR = [str(y) for y in PRE_TREATMENT_YEARS]
TREATMENT_LINE_X = PRE_TREATMENT_END_YEAR + 0.5

# --- HELPER FUNCTION FOR T-TEST ---

def calculate_did_t_test(ca_pre, ca_post, ctrl_pre, ctrl_post):
    """Performs a simple T-test for the DiD estimate."""
    ca_pre, ca_post = np.asarray(ca_pre), np.asarray(ca_post)
    ctrl_pre, ctrl_post = np.asarray(ctrl_pre), np.asarray(ctrl_post)
    
    ca_pre = ca_pre[~np.isnan(ca_pre)]
    ca_post = ca_post[~np.isnan(ca_post)]
    ctrl_pre = ctrl_pre[~np.isnan(ctrl_pre)]
    ctrl_post = ctrl_post[~np.isnan(ctrl_post)]
    
    n_ca_pre, n_ca_post = len(ca_pre), len(ca_post)
    n_ctrl_pre, n_ctrl_post = len(ctrl_pre), len(ctrl_post)
    
    if n_ca_pre < 1 or n_ca_post < 1 or n_ctrl_pre < 1 or n_ctrl_post < 1:
        return {'did_effect': np.nan, 'se': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'sig': 'N/A'}
    
    mean_ca_pre, mean_ca_post = np.mean(ca_pre), np.mean(ca_post)
    mean_ctrl_pre, mean_ctrl_post = np.mean(ctrl_pre), np.mean(ctrl_post)
    did_effect = (mean_ca_post - mean_ca_pre) - (mean_ctrl_post - mean_ctrl_pre)
    
    var_ca_pre = np.var(ca_pre, ddof=1) if n_ca_pre > 1 else 0
    var_ca_post = np.var(ca_post, ddof=1) if n_ca_post > 1 else 0
    var_ctrl_pre = np.var(ctrl_pre, ddof=1) if n_ctrl_pre > 1 else 0
    var_ctrl_post = np.var(ctrl_post, ddof=1) if n_ctrl_post > 1 else 0

    se2 = (var_ca_post / n_ca_post) + (var_ca_pre / n_ca_pre) + \
          (var_ctrl_post / n_ctrl_post) + (var_ctrl_pre / n_ctrl_pre)
    
    se = np.sqrt(se2)
    
    if se == 0 or np.isnan(se):
        return {'did_effect': did_effect, 'se': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'sig': 'N/A'}

    t_stat = did_effect / se
    df_approx = n_ca_pre + n_ca_post + n_ctrl_pre + n_ctrl_post - 4
         
    p_value = 2 * (1 - t.cdf(np.abs(t_stat), df_approx))
    
    if p_value < 0.01:
        sig = '***'
    elif p_value < 0.05:
        sig = '**'
    elif p_value < 0.10:
        sig = '*'
    else:
        sig = 'Not Significant'
        
    return {'did_effect': did_effect, 'se': se, 't_stat': t_stat, 'p_value': p_value, 'sig': sig}


# --- HELPER FUNCTION FOR DID CALCULATION (GROUP) ---

def calculate_did_for_group(df_wide, treated_unit, pre_years, post_years, control_states):
    """Calculates the aggregate DiD effect and T-test results."""
    present_control_states = [s for s in control_states if s in df_wide.columns]
    
    if treated_unit not in df_wide.columns or not present_control_states:
        return {'CA_change': np.nan, 'Control_change': np.nan, 'did_effect': np.nan, 'control_states_used': [], 'size': 0, 't_test_results': {'sig': 'N/A', 'p_value': np.nan}}

    df_did = df_wide[[treated_unit] + present_control_states].copy()
    df_did['Control_Avg_Outcome'] = df_did[present_control_states].mean(axis=1)

    ca_pre = df_did.loc[df_did.index.isin(pre_years), treated_unit].dropna().values
    ca_post = df_did.loc[df_did.index.isin(post_years), treated_unit].dropna().values
    ctrl_pre = df_did.loc[df_did.index.isin(pre_years), 'Control_Avg_Outcome'].dropna().values
    ctrl_post = df_did.loc[df_did.index.isin(post_years), 'Control_Avg_Outcome'].dropna().values

    CA_pre_avg = np.mean(ca_pre)
    CA_post_avg = np.mean(ca_post)
    Control_pre_avg = np.mean(ctrl_pre)
    Control_post_avg = np.mean(ctrl_post)
    
    CA_change = CA_post_avg - CA_pre_avg
    Control_avg_change = Control_post_avg - Control_pre_avg
    did_effect = CA_change - Control_avg_change
    
    t_test_results = calculate_did_t_test(ca_pre, ca_post, ctrl_pre, ctrl_post)
    
    return {
        'CA_change': CA_change, 
        'Control_change': Control_avg_change, 
        'did_effect': did_effect,
        'control_states_used': present_control_states,
        'size': len(present_control_states),
        't_test_results': t_test_results
    }


# --- HELPER FUNCTION FOR DID CALCULATION (PER STATE) ---

def calculate_did_per_state(df_wide, treated_unit, pre_years, post_years, control_states):
    """Calculates the DiD effect for CA vs each control state individually and includes T-test results."""
    did_per_state = {}
    present_control_states = [s for s in control_states if s in df_wide.columns]

    if treated_unit not in df_wide.columns or not present_control_states:
        return {}

    ca_pre = df_wide.loc[df_wide.index.isin(pre_years), treated_unit].dropna().values
    ca_post = df_wide.loc[df_wide.index.isin(post_years), treated_unit].dropna().values
    
    if len(ca_pre) == 0 or len(ca_post) == 0:
         return {}
         
    CA_change = np.mean(ca_post) - np.mean(ca_pre)
    
    for control_state in present_control_states:
        ctrl_pre = df_wide.loc[df_wide.index.isin(pre_years), control_state].dropna().values
        ctrl_post = df_wide.loc[df_wide.index.isin(post_years), control_state].dropna().values
        
        if len(ctrl_pre) == 0 or len(ctrl_post) == 0:
            did_per_state[control_state] = {'did_effect': np.nan, 'sig': 'N/A', 'p_value': np.nan}
            continue

        t_test_results = calculate_did_t_test(ca_pre, ca_post, ctrl_pre, ctrl_post)
        
        # --- MODIFICATION: INCLUDE P-VALUE ---
        did_per_state[control_state] = {
            'did_effect': t_test_results['did_effect'], 
            'sig': t_test_results['sig'],
            'p_value': t_test_results['p_value'] # Added p-value
        }
        
    return did_per_state


# --- HELPER FUNCTION FOR IMPUTATION ---

def impute_missing_data(df, pre_years_str, max_nan):
    """
    Drops rows with more than max_nan missing values in the pre-treatment period
    and imputes remaining NaNs using linear regression.
    """
    df_temp = df.copy()
    
    pre_cols = [col for col in pre_years_str if col in df_temp.columns]

    df_ca = df_temp[df_temp['State'] == TREATED_UNIT].copy()
    df_donor = df_temp[df_temp['State'] != TREATED_UNIT].copy()
    
    donor_nan_counts = df_donor[pre_cols].isna().sum(axis=1)
    df_clean_subset = df_donor[donor_nan_counts <= max_nan].copy()
    
    for index, row in df_clean_subset.iterrows():
        if row[pre_cols].isna().any():
            imputation_df = row[pre_cols].to_frame(name='Outcome').reset_index().rename(columns={'index': 'Year'})
            imputation_df['Year'] = pd.to_numeric(imputation_df['Year'])
            observed_data = imputation_df.dropna()
            missing_data = imputation_df[imputation_df['Outcome'].isna()]
            
            if len(observed_data) >= 2:
                model = LinearRegression()
                model.fit(observed_data[['Year']], observed_data['Outcome'])
                predicted_values = model.predict(missing_data[['Year']])
                
                for i, year in enumerate(missing_data['Year']):
                    df_clean_subset.loc[index, str(year)] = predicted_values[i]
            else:
                mean_val = observed_data['Outcome'].mean()
                if not pd.isna(mean_val):
                    for i, year in enumerate(missing_data['Year']):
                        df_clean_subset.loc[index, str(year)] = mean_val
                    
    return pd.concat([df_ca, df_clean_subset])


# --- 2. DATA LOADING AND CLEANING ---

df = pd.read_csv(FILE_NAME)
year_cols = [col for col in df.columns if col != 'State']
for col in year_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ***CRITICAL FIX: Get the original donor state order from the raw CSV data***
df_donor_original_order = df[df['State'] != TREATED_UNIT]['State'].tolist()

df_clean = impute_missing_data(df, PRE_TREATMENT_YEARS_STR, MAX_NAN_TO_KEEP)

df_long = pd.melt(df_clean, id_vars=['State'], value_vars=year_cols,
                  var_name='Year', value_name='Outcome')
df_long['Year'] = pd.to_numeric(df_long['Year']).astype(int)
df_long_clean = df_long.drop_duplicates(subset=['State', 'Year'], keep='first')
df_wide = df_long_clean.pivot(index='Year', columns='State', values='Outcome')


# --- 3. SYNTHETIC CONTROL OPTIMIZATION (SCM) ---

df_pre = df_wide.loc[df_wide.index.isin(PRE_TREATMENT_YEARS)].dropna(axis=1)
Y_ca_pre = df_pre[TREATED_UNIT].values
donor_states = [col for col in df_pre.columns if col != TREATED_UNIT]
X_donor_pre = df_pre[donor_states].values

def objective(W):
    synthetic_outcome = X_donor_pre @ W
    diff = Y_ca_pre - synthetic_outcome
    return np.sum(diff**2)

bounds = [(0, 1)] * len(donor_states)
constraints = ({'type': 'eq', 'fun': lambda W: np.sum(W) - 1})
W_initial = np.ones(len(donor_states)) / len(donor_states)

result = minimize(objective, W_initial,
                  method='trust-constr',
                  bounds=bounds,
                  constraints=constraints)

W_optimal = result.x
RMSD = np.sqrt(result.fun / len(PRE_TREATMENT_YEARS))


# --- 4. RESULTS CALCULATION (SCM PATH) ---

X_donor_full = df_wide[donor_states].values
Y_synthetic_full = X_donor_full @ W_optimal

df_comparison_scm = pd.DataFrame({
    'Year': df_wide.index,
    'CA_Outcome': df_wide[TREATED_UNIT].values,
    'Synthetic_CA_Outcome': Y_synthetic_full
})

df_comparison_scm['Treatment_Effect'] = df_comparison_scm['CA_Outcome'] - df_comparison_scm['Synthetic_CA_Outcome']

scm_t_test = calculate_did_t_test(
    df_comparison_scm[df_comparison_scm['Year'].isin(PRE_TREATMENT_YEARS)]['CA_Outcome'].values,
    df_comparison_scm[df_comparison_scm['Year'].isin(POST_TREATMENT_YEARS)]['CA_Outcome'].values,
    df_comparison_scm[df_comparison_scm['Year'].isin(PRE_TREATMENT_YEARS)]['Synthetic_CA_Outcome'].values,
    df_comparison_scm[df_comparison_scm['Year'].isin(POST_TREATMENT_YEARS)]['Synthetic_CA_Outcome'].values
)
did_effect_scm = scm_t_test['did_effect']


# --- 4B. CONDITIONAL DiD CALCULATION (USES CSV ORDER) ---

# Filter original order list to only include states that survived SCM filtering
surviving_donor_states_in_csv_order = [s for s in df_donor_original_order if s in donor_states]

is_single_state_scm = (W_optimal >= 0.99).any()
did_results_combined = {}
did_per_state = {} 

if is_single_state_scm:
    # 1. First 3 states from the CSV order
    did_group_3 = surviving_donor_states_in_csv_order[:3]
    did_results_3 = calculate_did_for_group(df_wide, TREATED_UNIT, PRE_TREATMENT_YEARS, POST_TREATMENT_YEARS, did_group_3)
    did_results_combined['Group_3'] = did_results_3

    # 2. First 10 states from the CSV order
    did_group_10 = surviving_donor_states_in_csv_order[:10]
    did_results_10 = calculate_did_for_group(df_wide, TREATED_UNIT, PRE_TREATMENT_YEARS, POST_TREATMENT_YEARS, did_group_10)
    did_results_combined['Group_10'] = did_results_10

    # 3. All remaining states (the full surviving list)
    did_group_all = donor_states
    did_results_all = calculate_did_for_group(df_wide, TREATED_UNIT, PRE_TREATMENT_YEARS, POST_TREATMENT_YEARS, did_group_all)
    did_results_combined['Group_All'] = did_results_all
    
    # Calculate DiD per state for the "First 3 States" group
    did_per_state = calculate_did_per_state(df_wide, TREATED_UNIT, PRE_TREATMENT_YEARS, POST_TREATMENT_YEARS, did_group_3)
    
    
# --- 5. DISPLAY NUMERIC RESULTS ---

print("\n--- CAUSAL ANALYSIS RESULTS ---")
print(f"File Used: {FILE_NAME}")
print(f"Treated Unit: {TREATED_UNIT}")
print(f"Pre-Treatment Period: {PRE_TREATMENT_YEARS[0]}-{PRE_TREATMENT_END_YEAR}")
print(f"Post-Treatment Period: {POST_TREATMENT_START_YEAR}-{POST_TREATMENT_YEARS[-1]}\n")

# Display SCM Results
print("--- 1. SYNTHETIC CONTROL METHOD (SCM) RESULTS ---")
print(f"Optimization Success: {result.success}")
print(f"Sum of Optimal Weights: {np.sum(W_optimal):.4f}")
print(f"Average Pre-Treatment Fit (RMSD): {RMSD:.2f}")

# --- MODIFICATION: INCLUDE P-VALUE IN SCM RESULT ---
p_val_scm = scm_t_test['p_value']
print("\nSCM Difference-in-Differences Estimate:")
print(f"   SCM DiD Estimate (Causal Effect): {did_effect_scm:.2f} {scm_t_test['sig']} (p={p_val_scm:.3f})")

# Optimal Weights DataFrame
W_optimal_df = pd.DataFrame({'State': donor_states, 'Weight': W_optimal})
W_optimal_df = W_optimal_df[W_optimal_df['Weight'] > 1e-4].sort_values(by='Weight', ascending=False)
W_optimal_df.to_csv('synthetic_control_weights_data3_conditional.csv', index=False)
print("\nOptimal Non-Zero Weights (Weights saved to 'synthetic_control_weights_data3_conditional.csv'):")
print(W_optimal_df)

# Display Conditional DiD Results
# --- 5. DISPLAY NUMERIC RESULTS ---

print("\n--- CAUSAL ANALYSIS RESULTS ---")
print(f"File Used: {FILE_NAME}")
print(f"Treated Unit: {TREATED_UNIT}")
print(f"Pre-Treatment Period: {PRE_TREATMENT_YEARS[0]}-{PRE_TREATMENT_END_YEAR}")
print(f"Post-Treatment Period: {POST_TREATMENT_START_YEAR}-{POST_TREATMENT_YEARS[-1]}\n")

# Display SCM Results
print("--- 1. SYNTHETIC CONTROL METHOD (SCM) RESULTS ---")
print(f"Optimization Success: {result.success}")
print(f"Sum of Optimal Weights: {np.sum(W_optimal):.4f}")
print(f"Average Pre-Treatment Fit (RMSD): {RMSD:.2f}")

# --- MODIFICATION: INCLUDE P-VALUE IN SCM RESULT ---
p_val_scm = scm_t_test['p_value']
# Change 1: Modified p-value format from :.3f to :.2g
print("\nSCM Difference-in-Differences Estimate:")
print(f"   SCM DiD Estimate (Causal Effect): {did_effect_scm:.2f} {scm_t_test['sig']} (p={p_val_scm:.2g})")

# Optimal Weights DataFrame
W_optimal_df = pd.DataFrame({'State': donor_states, 'Weight': W_optimal})
W_optimal_df = W_optimal_df[W_optimal_df['Weight'] > 1e-4].sort_values(by='Weight', ascending=False)
W_optimal_df.to_csv('synthetic_control_weights_data3_conditional.csv', index=False)
print("\nOptimal Non-Zero Weights (Weights saved to 'synthetic_control_weights_data3_conditional.csv'):")
print(W_optimal_df)

# Display Conditional DiD Results
if is_single_state_scm:
    print("\n--- 2. CONDITIONAL DIFFERENCE-IN-DIFFERENCES (DiD) RESULTS ---")
    
    ca_change_val = did_results_combined['Group_3']['CA_change']
    print(f"CA Change (Post - Pre): {ca_change_val:.2f}")

    for group_key, results in did_results_combined.items():
        if group_key == 'Group_3':
            label = "First 3 States (CSV Order)"
        elif group_key == 'Group_10':
            label = "First 10 States (CSV Order)"
        elif group_key == 'Group_All':
            label = "All SCM Donor States"
        else:
            continue
        
        sig = results['t_test_results']['sig']
        p_val = results['t_test_results']['p_value'] # Get p-value
        
        print(f"\nAggregate DiD Estimate for {label}: (Control Group size: {results['size']})")
        print(f"   Control Group Avg Change (Post - Pre): {results['Control_change']:.2f}")
        # Change 2: Modified p-value format from :.3f to :.2g
        print(f"   DiD Estimate (Causal Effect): {results['did_effect']:.2f} {sig} (p={p_val:.2g})")
    
    # Display DiD per state for the first group for detail
    if did_per_state:
        states_to_print = ", ".join(did_group_3[:3])
        print(f"\nDiD Estimate Per State for the '{states_to_print}' group:")
        for state, data in did_per_state.items():
            sig = data['sig']
            p_val = data['p_value']
            # Change 3: Modified p-value format from :.3f to :.2g
            print(f"   {state:15}: {data['did_effect']:.2f} {sig} (p={p_val:.2g})")


# --- 6. PLOTTING ---
# ... (rest of the code for plotting remains unchanged)


# --- 6. PLOTTING ---

# Plot 1: Actual CA and Synthetic CA Outcomes (SCM)
plt.figure(figsize=(10, 6))
plt.plot(df_comparison_scm['Year'], df_comparison_scm['CA_Outcome'], label='Actual CA Outcome', marker='o')
plt.plot(df_comparison_scm['Year'], df_comparison_scm['Synthetic_CA_Outcome'], label='Synthetic CA (SCM Counterfactual)', linestyle='--', marker='x')
plt.axvline(x=TREATMENT_LINE_X, color='red', linestyle=':', label='Treatment Starts (2020)')
plt.title('Actual CA vs. Synthetic CA Outcome (SCM Fit)')
plt.xlabel('Year')
plt.ylabel('Outcome Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ca_synthetic_control_plot_data3_conditional.png')
print("\nPlot 1 (Actual vs. Synthetic SCM) saved as 'ca_synthetic_control_plot_data3_conditional.png'")

# Plot 2: The gap (treatment effect) (SCM)
plt.figure(figsize=(10, 3))
plt.bar(df_comparison_scm['Year'], df_comparison_scm['Treatment_Effect'],
        color=['blue' if y in PRE_TREATMENT_YEARS else 'orange' for y in df_comparison_scm['Year']])
plt.axvline(x=TREATMENT_LINE_X, color='red', linestyle=':', label='Treatment Starts (2020)')
plt.axhline(y=0, color='black', linestyle='-')
plt.title('SCM Treatment Effect Gap (CA - Synthetic CA)')
plt.xlabel('Year')
plt.ylabel('Effect (Gap)')
plt.xticks(df_comparison_scm['Year'])
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('treatment_effect_plot_data3_gap_conditional.png')
print("Plot 2 (SCM Treatment Effect Gap) saved as 'treatment_effect_plot_data3_gap_conditional.png'")