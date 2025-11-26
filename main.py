import pandas as pd
import numpy as np 
import statsmodels.formula.api as smf
import numpy as np


def data_1(file_path):
    
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        # return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")

    data_1 = data[['Year', 
                             '# of Active US Investors', 
                             '# of Active US 1st Round Investors', 
                             '# of Active US Life Science Investors', 
                             '# Active US VC Investors', 
                             '# Active US VC 1st Round Investors', 
                             '# Active US VC Life Science Investors']
                        ]
    
    for col in data_1.columns:
        data_1[col] = data_1[col].astype(str)
        data_1[col] = data_1[col].str.replace(',', '')
        data_1[col] = pd.to_numeric(data_1[col], errors='coerce').fillna(0).astype(int)

    return data_1

def active_investors_analyze(data):

    data['# US Tech'] = data['# of Active US Investors'] - data['# of Active US Life Science Investors']

    df_long = pd.melt(
        data,
        id_vars=['Year'],
        value_vars=['# US Tech', '# of Active US Life Science Investors'],
        var_name='Group_Name',
        value_name='Outcome_Value'
    )

    TREATMENT_YEAR = 2020

    TREATED_GROUP_NAME = '# US Tech' 
    df_long.loc[:, 'TREATMENT_GROUP'] = np.where(df_long['Group_Name'] == TREATED_GROUP_NAME, 1, 0)
    df_long.loc[:, 'AFTER_PERIOD'] = np.where(df_long['Year'] >= TREATMENT_YEAR, 1, 0)
    df_long.loc[:, 'INTERACTION'] = df_long['TREATMENT_GROUP'] * df_long['AFTER_PERIOD']

    model = smf.ols('Outcome_Value ~ TREATMENT_GROUP + AFTER_PERIOD + INTERACTION', data=df_long).fit()

    ALPHA = 0.05

    did_estimate = model.params['INTERACTION']
    p_value = model.pvalues['INTERACTION']

    is_significant = p_value < ALPHA
    significance_text = "is statistically significant" if is_significant else "is NOT statistically significant"

    print("\n--- DiD Conclusion ---")
    print(f"DiD Estimate (Estimated Effect on Tech Group post-2020): {did_estimate:,.2f} investors")
    print(f"P-value for Interaction Term: {p_value:.4f}")
    print(f"Conclusion: The estimated treatment effect {significance_text} at the {ALPHA*100:.0f}% level.")

def vc_investors_analyze(data):
    
    data['# US VC Tech'] = data['# Active US VC Investors'] - data['# Active US VC Life Science Investors']

    df_long = pd.melt(
        data,
        id_vars=['Year'],
        value_vars=['# US VC Tech', '# Active US VC Life Science Investors'],
        var_name='Group_Name',
        value_name='Outcome_Value'
    )
    
    # Crucial step: Reset index after melt to ensure a simple 0..N index
    df_long = df_long.reset_index(drop=True)

    TREATMENT_YEAR = 2020
    
    # --- CHANGE: Define the NEW Treated Group (VC Tech) to make Life Science the Control ---
    TREATED_GROUP_NAME = '# US VC Tech' 
    
    # Use .loc for explicit assignment when creating dummy variables
    # The 'Who' is now the VC Tech Investor group
    df_long.loc[:, 'TREATMENT_GROUP'] = np.where(df_long['Group_Name'] == TREATED_GROUP_NAME, 1, 0)
    
    # The 'When' remains 2020 and later
    df_long.loc[:, 'AFTER_PERIOD'] = np.where(df_long['Year'] >= TREATMENT_YEAR, 1, 0)
    
    # Interaction term remains (Treated * After)
    df_long.loc[:, 'INTERACTION'] = df_long['TREATMENT_GROUP'] * df_long['AFTER_PERIOD']

    # FIX: Use .copy() just before the final regression call. This ensures 
    # the DataFrame passed to smf.ols is a simple, contiguous block of data 
    # with a basic integer index, resolving the shape (40, 40) ValueError.
    df_final = df_long[['Outcome_Value', 'TREATMENT_GROUP', 'AFTER_PERIOD', 'INTERACTION']].copy()
    
    model = smf.ols('Outcome_Value ~ TREATMENT_GROUP + AFTER_PERIOD + INTERACTION', data=df_final).fit()

    # START: MOVED PRINT LOGIC FROM main()
    ALPHA = 0.05

    did_estimate = model.params['INTERACTION']
    p_value = model.pvalues['INTERACTION']

    is_significant = p_value < ALPHA
    significance_text = "is statistically significant" if is_significant else "is NOT statistically significant"

    print("\n--- DiD Conclusion ---")
    print(f"DiD Estimate (Estimated Effect on Tech Group post-2020): {did_estimate:,.2f} investors")
    print(f"P-value for Interaction Term: {p_value:.4f}")
    print(f"Conclusion: The estimated treatment effect {significance_text} at the {ALPHA*100:.0f}% level.")
    
    # return model

    return 

def main():

    data = data_1('data/data1.csv')
    data_nonVC = active_investors_analyze(data)

    data_VC = vc_investors_analyze(data)

    # from synthetic import data_read_data2
    # from synthetic import synthetic_analysis_t1, synthetic_analysis_t2

    # data2 = data_read_data2('data/data2.csv')
    # data2.to_csv('data/data2_5.csv', index=False) 
    # synthetic_analysis_t2(data2)
    # synthetic_analysis_t1(data2)


main()