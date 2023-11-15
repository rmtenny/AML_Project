import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mstats
import torch
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data/equity_chars_ret.csv', index_col = 0)
df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
characteristics = list(set(df.columns).difference({'permno','DATE', 'sic2', 'RET'}))

df_top = df.sort_values('mvel1',ascending=False).groupby('DATE').head(1000).reset_index(drop=True)
df_bot = df.sort_values('mvel1',ascending=False).groupby('DATE').tail(1000).reset_index(drop=True)

# missing data before filling
print("missing values:")
df.isnull().sum()

df = df.dropna(subset = 'RET')
# df.isnull().sum()

# Select duplicates based on 'permno' and 'DATE'
duplicate_mask = df.duplicated(subset=['permno', 'DATE'], keep=False)
duplicates = df[duplicate_mask]

# Display the duplicate rows
print('Duplicated Data:')
print(duplicates)

# Calculate missing percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Calculate correlation matrix
correlation_matrix = df[characteristics].corr()

# Flatten the correlation matrix
flat_corr = correlation_matrix.unstack()

# Remove self-correlations and duplicates
flat_corr = flat_corr[flat_corr != 1].sort_values(ascending=False)

# Filter correlations with absolute value greater than 0.7
high_corr = flat_corr[(flat_corr.abs() > 0.7) & (flat_corr.abs() < 1)]

# Print the sorted correlations and corresponding feature names
for idx, (features, corr) in enumerate(high_corr.items()):
    feature1, feature2 = features
    print(f"{idx + 1}. Correlation: {corr:.4f} - Features: {feature1} and {feature2}")

np.unique(df['sic2'])

len(np.unique(df['permno']))

# Convert 'RET' to numeric, coerce non-numeric values to NaN
df['RET'] = pd.to_numeric(df['RET'], errors='coerce')

# Create empty lists to store feature names, correlations, and p-values
features = []
correlations = []
p_values = []

# Loop through each feature and calculate correlation with 'RET'
for feature in characteristics:
    if feature != 'RET':  # Skip the 'RET' column
        # Calculate correlation and p-value
        df_tmp = df[[feature] + ['RET']].dropna()
        correlation, p_value = pearsonr(df_tmp[feature], df_tmp['RET'])

        # Append results to lists
        features.append(feature)
        correlations.append(correlation)
        p_values.append(p_value)

# Create a DataFrame with the results
correlation_df_all = pd.DataFrame({'Feature': features, 'Correlation': correlations, 'P-Value': p_values})
correlation_df_all['Correlation'] *= 100

# Sort DataFrame by absolute correlation in descending order
correlation_df_all = correlation_df_all.reindex(correlation_df_all['Correlation'].abs().sort_values(ascending=False).index)

# Significance color mapping
significance_color = np.where(correlation_df_all['P-Value'] < 0.05, 'rgba(0, 100, 0, 0.8)', 'rgba(139, 0, 0, 0.8)')

# Specify the threshold for P-Value and missing percentage
p_value_threshold = 0.05
missing_percentage_threshold = 30

# Select rows based on the conditions
if 'Feature' in correlation_df_all.columns:
  correlation_df_all.set_index('Feature', inplace=True)
correlation_df_all['missing_per'] = missing_percentage

selected_characteristics = correlation_df_all[(correlation_df_all['P-Value'] > p_value_threshold) & (correlation_df_all['missing_per'] > missing_percentage_threshold)].index
selected_characteristics1 = correlation_df_all[(correlation_df_all['P-Value'] > p_value_threshold)].index
selected_characteristics2 = correlation_df_all[(correlation_df_all['missing_per'] > missing_percentage_threshold)].index

# correlation_df_all

# print(selected_characteristics)
# print(selected_characteristics1)
# print(selected_characteristics2)
# characteristics = selected_characteristics


for sic, df_sic in df.groupby('sic2'):

    # Create empty lists to store feature names, correlations, and p-values
    features = []
    correlations = []
    p_values = []

    # Loop through each feature and calculate correlation with 'RET'
    for feature in characteristics:
        if feature != 'RET':  # Skip the 'RET' column
            # Calculate correlation and p-value
            df_tmp = df_sic[[feature] + ['RET']].dropna()

            # Check if the feature has no variation
            if np.isclose(np.min(df_tmp[feature]), np.max(df_tmp[feature])):
                # print(f"Warning: {feature} has no variation in SIC {sic}. Skipping.")
                break

            if len(df_tmp) > 5000:
              correlation, p_value = pearsonr(df_tmp[feature], df_tmp['RET'])

              # Append results to lists
              features.append(feature)
              correlations.append(correlation)
              p_values.append(p_value)

    if len(features) > 90:
        # Create a DataFrame with the results
        correlation_df = pd.DataFrame({'Feature': features, 'Correlation': correlations, 'P-Value': p_values})
        correlation_df['Correlation'] *= 100

        # # Sort DataFrame by absolute correlation in descending order
        # correlation_df = correlation_df.reindex(correlation_df['Correlation'].abs().sort_values(ascending=False).index)

        # # Significance color mapping
        # significance_color = np.where(correlation_df['P-Value'] < 0.05, 'rgba(0, 100, 0, 0.8)', 'rgba(139, 0, 0, 0.8)')

# fill na with cross-sectional median
for feature in characteristics:
     df[feature] = df.groupby('DATE')[feature].transform(lambda x: x.fillna(x.median()))

df.isnull().sum()

# Calculate missing percentage
missing_percentage2 = (df[characteristics].isnull().sum() / len(df)) * 100

def fill_median(data, characteristics):
    for ch in characteristics:
         data[ch] = data.groupby('DATE')[ch].transform(lambda x: x.fillna(x.median()))
    return data

df_top = fill_median(df_top, characteristics)
df_bot = fill_median(df_bot, characteristics)
df = fill_median(df, characteristics)

len(np.unique(df['sic2']))

# get dummies for SIC code
def get_sic_dummies(data):
    sic_dummies = pd.get_dummies(data['sic2'].fillna(999).astype(int),prefix='sic').drop('sic_999',axis=1)
    data = pd.concat([data, sic_dummies],axis=1)
    data.drop(['sic2'],inplace=True,axis=1)
    return data

df = get_sic_dummies(df)
df_top = get_sic_dummies(df_top)
df_bot = get_sic_dummies(df_bot)

## Macro Predictors 
# load macroeconomic predictors data
df_ma = pd.read_csv('data/PredictorData2022.csv')
df_ma = df_ma[(df_ma['yyyymm']>=200001)&(df_ma['yyyymm']<=202112)].reset_index(drop=True)

# construct predictor
ma_predictors = ['dp_sp','ep_sp','bm_sp','ntis','tbl','tms','dfy','svar']
df_ma['Index'] = df_ma['Index'].str.replace(',','').astype('float64')
df_ma['dp_sp'] = df_ma['D12']/df_ma['Index']
df_ma['ep_sp'] = df_ma['E12']/df_ma['Index']
df_ma.rename({'b/m':'bm_sp'},axis=1,inplace=True)
df_ma['tms'] = df_ma['lty']-df_ma['tbl']
df_ma['dfy'] = df_ma['BAA']-df_ma['AAA']
df_ma = df_ma[['yyyymm']+ma_predictors]
# df_ma['yyyymm'] = pd.to_datetime(df_ma['yyyymm'],format='%Y%m')+pd.offsets.MonthEnd(0)
# # Convert datetime to integer in the format 'yyyymm'
# df_ma['yyyymm'] = df_ma['yyyymm'].dt.strftime('%Y%m%d').astype(int)

df_ma.head()

def interactions(data, data_ma, characteristics, ma_predictors):
    # construct interactions between firm characteristics and macroeconomic predictors
    # Convert 'Date' to datetime
    data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m%d')

    # Extract year and month and convert them to 'yyyymm'
    data['DATE'] = data['DATE'].dt.to_period('M').dt.strftime('%Y%m').astype(int)

    # Merge the dataset
    data_ma_long = pd.merge(data[['DATE']], data_ma,left_on='DATE', right_on='yyyymm', how='left').reset_index(drop=True)
    # data = data.reset_index(drop=True)
    # data_ma_long = data_ma_long.reset_index(drop=True)

    # Generate interaction features
    interactions_df = pd.DataFrame()  # Initialize an empty DataFrame for interactions
    for fc in characteristics:
        for mp in ma_predictors:
            interactions_df[f'{fc}*{mp}'] = data[fc] * data_ma_long[mp]

    # Concatenate the original DataFrame and the interactions DataFrame
    data = pd.concat([data, interactions_df], axis=1)

    features = list(set(data.columns).difference({'permno','DATE','RET'})) # a list storing all 920 features used
    print(f"# of feature is {len(features)}")
    return data, features

def split_by_fix_date(data, train_start_date, val_start_date, test_start_date, test_end_date):
    train_dataset = data[((data['DATE'] >= train_start_date) & (data['DATE'] < val_start_date))]
    val_dataset = data[(data['DATE'] >= val_start_date) & (data['DATE'] < test_start_date)]
    test_dataset = data[(data['DATE'] >= test_start_date) & (data['DATE'] <= test_end_date)]
    return train_dataset, val_dataset, test_dataset

def transform(data, features, minmax=True):
    if minmax:
        scaler = MinMaxScaler((-1, 1))
        data[features] = scaler.fit_transform(data[features])
    else:
        pass
    print(f"The shape of the data is: {data.shape}")
    return data

# Save unmodified data
features = characteristics
Macro = False
if Macro:
    # Create features by mutiplying with macro characteristics
    df, features = interactions(df, df_ma, characteristics, ma_predictors)
    df_top, features = interactions(df_top, df_ma, characteristics, ma_predictors)
    df_bot, features = interactions(df_bot, df_ma, characteristics, ma_predictors)   

    train_start_date = 200001
    val_start_date = 201501
    test_start_date = 201801
    test_end_date = 202112

    train_dataset, val_dataset, test_dataset = split_by_fix_date(df, train_start_date, val_start_date, test_start_date, test_end_date)
    train_dataset_top, val_dataset_top, test_dataset_top = split_by_fix_date(df_top, train_start_date, val_start_date, test_start_date, test_end_date)
    train_dataset_bot, val_dataset_bot, test_dataset_bot = split_by_fix_date(df_bot, train_start_date, val_start_date, test_start_date, test_end_date)

    train_dataset = transform(train_dataset, features, minmax=True)
    val_dataset = transform(val_dataset, features, minmax=True)
    test_dataset = transform(test_dataset, features, minmax=True)

    train_dataset_top = transform(train_dataset_top, features, minmax=True)
    val_dataset_top = transform(val_dataset_top, features, minmax=True)
    test_dataset_top = transform(test_dataset_top, features, minmax=True)

    train_dataset_bot = transform(train_dataset_bot, features, minmax=True)
    val_dataset_bot = transform(val_dataset_bot, features, minmax=True)
    test_dataset_bot = transform(test_dataset_bot, features, minmax=True)

    train_dataset['DATE'] = train_dataset['DATE'].astype('int')

    val_dataset['DATE'] = val_dataset['DATE'].astype('int')
    test_dataset['DATE'] = test_dataset['DATE'].astype('int')

    train_dataset_top['DATE'] = train_dataset_top['DATE'].astype('int')
    val_dataset_top['DATE'] = val_dataset_top['DATE'].astype('int')
    test_dataset_top['DATE'] = test_dataset_top['DATE'].astype('int')

    train_dataset_bot['DATE'] = train_dataset_bot['DATE'].astype('int')
    val_dataset_bot['DATE'] = val_dataset_bot['DATE'].astype('int')
    test_dataset_bot['DATE'] = test_dataset_bot['DATE'].astype('int')

    ## Save Datasets as CSV files
    print("Saving Data")
    train_dataset.to_csv("data/train_dataset.csv")
    val_dataset.to_csv("data/val_dataset.csv")
    test_dataset.to_csv("data/test_dataset.csv")

    pd.DataFrame(features).to_csv("data/features.csv")
    train_dataset_top.to_csv("data/train_dataset_top.csv")
    val_dataset_top.to_csv("data/val_dataset_top.csv")
    test_dataset_top.to_csv("data/test_dataset_top.csv")
    train_dataset_bot.to_csv("data/train_dataset_bot.csv")
    val_dataset_bot.to_csv("data/val_dataset_bot.csv")
    test_dataset_bot.to_csv("data/test_dataset_bot.csv")
else:
    train_start_date = 20000131
    val_start_date = 20150131
    test_start_date = 20180131
    test_end_date = 20211231

    train_dataset, val_dataset, test_dataset = split_by_fix_date(df, train_start_date, val_start_date, test_start_date, test_end_date)
    train_dataset_top, val_dataset_top, test_dataset_top = split_by_fix_date(df_top, train_start_date, val_start_date, test_start_date, test_end_date)
    train_dataset_bot, val_dataset_bot, test_dataset_bot = split_by_fix_date(df_bot, train_start_date, val_start_date, test_start_date, test_end_date)


    train_dataset = transform(train_dataset, features, minmax=True)
    val_dataset = transform(val_dataset, features, minmax=True)
    test_dataset = transform(test_dataset, features, minmax=True)

    train_dataset_top = transform(train_dataset_top, features, minmax=True)
    val_dataset_top = transform(val_dataset_top, features, minmax=True)
    test_dataset_top = transform(test_dataset_top, features, minmax=True)

    train_dataset_bot = transform(train_dataset_bot, features, minmax=True)
    val_dataset_bot = transform(val_dataset_bot, features, minmax=True)
    test_dataset_bot = transform(test_dataset_bot, features, minmax=True)

    train_dataset['DATE'] = train_dataset['DATE'].astype('int')

    val_dataset['DATE'] = val_dataset['DATE'].astype('int')
    test_dataset['DATE'] = test_dataset['DATE'].astype('int')

    train_dataset_top['DATE'] = train_dataset_top['DATE'].astype('int')
    val_dataset_top['DATE'] = val_dataset_top['DATE'].astype('int')
    test_dataset_top['DATE'] = test_dataset_top['DATE'].astype('int')

    train_dataset_bot['DATE'] = train_dataset_bot['DATE'].astype('int')
    val_dataset_bot['DATE'] = val_dataset_bot['DATE'].astype('int')
    test_dataset_bot['DATE'] = test_dataset_bot['DATE'].astype('int')

    ## Save Datasets as CSV files
    print("Saving Data")
    train_dataset.to_csv("dataP/train_dataset.csv")
    val_dataset.to_csv("dataP/val_dataset.csv")
    test_dataset.to_csv("dataP/test_dataset.csv")

    pd.DataFrame(features).to_csv("dataP/features.csv")
    train_dataset_top.to_csv("dataP/train_dataset_top.csv")
    val_dataset_top.to_csv("dataP/val_dataset_top.csv")
    test_dataset_top.to_csv("dataP/test_dataset_top.csv")
    train_dataset_bot.to_csv("dataP/train_dataset_bot.csv")
    val_dataset_bot.to_csv("dataP/val_dataset_bot.csv")
    test_dataset_bot.to_csv("dataP/test_dataset_bot.csv")