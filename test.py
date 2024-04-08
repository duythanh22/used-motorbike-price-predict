import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

encoded_columns = joblib.load('models/encoded_columns.pkl')
model = joblib.load('models/decision_tree_model.pkl')
df = pd.DataFrame({
        'Reg_Date': 2022.0,
        'Km': 11111.0,
        'Capacity': 1,
        'Type': 0,
        'Brand': 'Honda',
        'Model': 'Sh'
    }, index=[0])

encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_columns = ['Brand', 'Model']
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.str.lower())
encoded_data = pd.get_dummies(df, columns=categorical_columns)
new_data_encoded = encoded_data.reindex(columns=encoded_columns, fill_value=0)
print(model.predict(new_data_encoded))