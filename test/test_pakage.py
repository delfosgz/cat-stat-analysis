import pandas as pd
import cat_stat.main  as cstt

data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Purchase': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

cstt.get_analysis(df, 'Gender', 'Purchase')