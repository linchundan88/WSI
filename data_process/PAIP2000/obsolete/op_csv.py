import pandas as pd
file_csv = r'C:\PAPI2020\training_data\traning_data_MSI.csv'

df = pd.read_csv(file_csv)

for _, row in df.iterrows():
    print(row['WSI_ID'], row['MSI-H Prediction'])

print('OK')