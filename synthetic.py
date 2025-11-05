from sklearn.linear_model import LinearRegression
import numpy as np

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
    print(data_imputed)

