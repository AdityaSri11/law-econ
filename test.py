import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# 1. Create a sample DataFrame (using Card and Krueger (1994) logic)
data = {
    'outcome': [10, 12, 11, 15, 20, 25, 22, 28],
    'group': [0, 0, 0, 0, 1, 1, 1, 1], # 0 for control, 1 for treatment
    'time': [0, 0, 1, 1, 0, 0, 1, 1]    # 0 for pre-treatment, 1 for post-treatment
}
df = pd.DataFrame(data)

# Create the interaction term (Treatment * After)
df['interaction'] = df['group'] * df['time']

# 2. Fit the OLS (Ordinary Least Squares) regression model
# The formula specifies the outcome variable 'outcome' is a function of 
# 'group', 'time', and the 'interaction' term.
model = smf.ols('outcome ~ group + time + interaction', data=df).fit()

# 3. View the results
print(model.summary())

# The coefficient for 'interaction' is the DiD estimate
did_estimate = model.params['interaction']
print(f"\nDiD Estimate (Treatment Effect): {did_estimate:.2f}")
