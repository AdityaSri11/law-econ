import pandas as pd
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

    import statsmodels.formula.api as smf

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
    
    return 

def main():

    data = data_1('data/data1.csv')
    data_1 = active_investors_analyze(data)

    data_2 = vc_investors_analyze(data)
main()