from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def data_read_data2(file_path):
    import pandas as pd
    
    try:
        data = pd.read_csv(file_path, index_col=0, dtype=str)
        print("Data loaded successfully.")
        # return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        # return None
    
    df_clean_strings = data.apply(lambda col: col.str.replace('"', '').str.replace(',', ''), axis=0)
    df_numeric = df_clean_strings.replace('-', np.nan).apply(pd.to_numeric, errors='coerce')
    # print(df_numeric)
    
    rows_to_keep_mask = df_numeric.isnull().sum(axis=1) <= 2
    data_cleaned = df_numeric[rows_to_keep_mask].copy()
    # print(data_cleaned)

    ############ Linear Regression
    model = LinearRegression()
    years = data_cleaned.columns.astype(int).values.reshape(-1, 1)
    # print(years)

    for s in data_cleaned.index:
        row = data_cleaned.loc[s]
        
        known_mask = row.notna()
        
        X_train = years[known_mask]        
        y_train = row[known_mask].values     
        
        missing_mask = row.isna()
        X_predict = years[missing_mask]     

        if len(y_train) < 2 or len(X_predict) == 0:
            continue

        model.fit(X_train, y_train)
        
        predicted_values = model.predict(X_predict)

        data_cleaned.loc[s, data_cleaned.columns[missing_mask]] = predicted_valuesrow = data_cleaned.loc[s]

        known_mask = row.notna()
        
        X_train = years[known_mask]          
        y_train = row[known_mask].values     
        
        missing_mask = row.isna()
        X_predict = years[missing_mask]      

        if len(y_train) < 2 or len(X_predict) == 0:
            continue
        model.fit(X_train, y_train)
        
        predicted_values = model.predict(X_predict)
        
        data_cleaned.loc[s, data_cleaned.columns[missing_mask]] = predicted_values

    data_imputed = data_cleaned
    return data_imputed

def synthetic_analysis_t2(data):
    from synthdid.model import SynthDID
    
    return

def synthetic_analysis_t1(data):
    
    print(data)

    states = ["California", "New York", "Massachusetts", "Florida", "Texas", "Maryland", "Illinois", "Washington", "Virginia", "District of Columbia", "Connecticut", "Pennsylvania", "Colorado", "Georgia", "Utah", "Michigan", "Ohio", "North Carolina", "Missouri", "Tennessee", "Wisconsin", "Indiana", "Oregon"]
    data_scm = pd.DataFrame(data, index=states)
    data_scm.columns = data_scm.columns.astype(int)
    # num_data_rows = len(data)
    # print(num_data_rows)
    # print(len(idx_states))

    df_long = data_scm.reset_index().rename(columns={'index': 'State'})
    df_long = pd.melt(df_long, id_vars='State', var_name='Year', value_name='Value')


    ######
    UNIT_VAR = 'State'        
    TIME_VAR = 'Year'         
    OUTCOME_VAR = 'Value'     
    TREATMENT_UNIT = 'California'
    TREATMENT_START_YEAR = 2020
    CONTROL_POOL = states
    TIME_START = df_long['Year'].min() # e.g., 2008

    # Time period used to optimize the weights (pre-treatment period)
    TIME_PRE_TREATMENT = range(TIME_START, TREATMENT_START_YEAR)

    import pysyncon
    CONTROL_IDENTIFIERS = [s for s in CONTROL_POOL if s != TREATMENT_UNIT]
    dp = pysyncon.Dataprep(
        foo=df_long, # Note: Dataprep requires the DataFrame to be passed as 'foo'
        predictors=[OUTCOME_VAR],
        predictors_op='mean', 
        dependent=OUTCOME_VAR,
        unit_variable=UNIT_VAR,
        time_variable=TIME_VAR,
        treatment_identifier=TREATMENT_UNIT,
        controls_identifier=CONTROL_IDENTIFIERS,
        time_predictors_prior=TIME_PRE_TREATMENT, 
        time_optimize_ssr=TIME_PRE_TREATMENT

    )

    # The Synth constructor accepts the Dataprep object as its main input.
    scm = pysyncon.Synth()
    scm.dataprep = dp
    scm.fit(dp)

    import matplotlib.pyplot as plt

    # scm.path_plot()

    summary_results = scm.summary()
    # print(summary_results)
    print(scm.weights())