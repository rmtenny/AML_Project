# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 02:15:50 2023

@author: Administrator
"""


## Data Preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('equity_chars_ret.csv', index_col = 0)
df = df[df['DATE'] >= 20100100]

df['RET'] = pd.to_numeric(df['RET'], errors='coerce')

# get 94 features name
characteristics = list(set(df.columns).difference({'permno','DATE', 'sic2', 'RET'}))

# Pick out Top 1000 and Bottom 1000 Firms Next, let's pick out the top 1000 and bottom 1000 firms
# with respect to market capitalization to see the differnce of predictability between big firms and small firms.

df_top = df.sort_values('mvel1',ascending=False).groupby('DATE').head(1000).reset_index(drop=True)
df_bot = df.sort_values('mvel1',ascending=False).groupby('DATE').tail(1000).reset_index(drop=True)


### Missing Characteristics
# - Delete the missing ret
# - Check duplicate rows

df = df.dropna(subset = 'RET')

# Select duplicates based on 'permno' and 'DATE'
duplicate_mask = df.duplicated(subset=['permno', 'DATE'], keep=False)
duplicates = df[duplicate_mask]


import plotly.express as px

# Calculate missing percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Create a plot for missing percentage with color scale
fig = px.bar(missing_percentage, x=missing_percentage.index, y=missing_percentage.values,
             labels={'index': 'Features', 'y': 'Missing Percentage'},
             title='Feature Missing Percentage',
             template='plotly_dark',
             color=missing_percentage.values,  # Use the values for color scale
             color_continuous_scale='Viridis',  # Specify the color scale
             )

# Show the plot
fig.show()



# - Calculate Features Correlation Matrix and Plot Heatmap

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
correlation_matrix = df[characteristics].corr()

# Plot heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", vmin=-1, vmax=1)
plt.title('Features Correlation Matrix')
plt.show()


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


flat_corr.to_csv(r"flat_corr.csv")

from scipy.stats import pearsonr, mstats

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

# Plot using plotly
fig = px.bar(correlation_df_all, x='Feature', y='Correlation', color=significance_color,
             labels={'Correlation': 'Correlation with RET (%)', 'P-Value': 'P-Value'},
             title='Correlation of Features with Return and Significance',
             template='plotly_dark')

# Show the plot
fig.show()

# - Feature Selection:
#  Select the features where p-value > 0.05 or missing percentage > 30%

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


from scipy.stats import pearsonr
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

        # Plot using plotly
        fig = px.bar(correlation_df, x='Feature', y='Correlation', color='P-Value',
                    labels={'Correlation': 'Correlation with RET (%)', 'P-Value': 'P-Value'},
                    title=f'sic2 = {sic}: Correlation of Features with Return and Significance',
                    template='plotly_dark')

        # Show the plot
        fig.show()
        
        
# - Replaced the missing data with cross-sectional median.       
# fill na with cross-sectional median
for feature in characteristics:
     df[feature] = df.groupby('DATE')[feature].transform(lambda x: x.fillna(x.median()))
     
# Calculate missing percentage
missing_percentage2 = (df[characteristics].isnull().sum() / len(df)) * 100

# Create a plot for missing percentage with color scale
fig = px.bar(missing_percentage2, x=missing_percentage2.index, y=missing_percentage2.values,
             labels={'index': 'Features', 'y': 'Missing Percentage'},
             title='Feature Missing Percentage After Fill by Cross-Sectional Median',
             template='plotly_dark',
             color=missing_percentage2.values,  # Use the values for color scale
             color_continuous_scale='Viridis',  # Specify the color scale
             )

# Show the plot
fig.show()    

# - Do the same process to top and bottom 1000 firms data.
def fill_median(data, characteristics):
    for ch in characteristics:
         data[ch] = data.groupby('DATE')[ch].transform(lambda x: x.fillna(x.median()))
    return data

df_top = fill_median(df_top, characteristics)
df_bot = fill_median(df_bot, characteristics)


### Transform SIC Code into Dummies

# get dummies for SIC code
def get_sic_dummies(data):
    sic_dummies = pd.get_dummies(data['sic2'].fillna(999).astype(int),prefix='sic').drop('sic_999',axis=1)
    data = pd.concat([data, sic_dummies],axis=1)
    data.drop(['sic2'],inplace=True,axis=1)
    return data

df = get_sic_dummies(df)
df_top = get_sic_dummies(df_top)
df_bot = get_sic_dummies(df_bot)



### Macroeconomic Predictors Data

# The eight macroeconomic predictors follows the definitions by Welch and Goyal (2008, RFS). The data are available on Prof Goyal's [website](https://sites.google.com/view/agoyal145).

df_ma = pd.read_csv('PredictorData2022.csv')
df_ma = df_ma[(df_ma['yyyymm']>=201001)&(df_ma['yyyymm']<=202112)].reset_index(drop=True)

# construct predictor
ma_predictors = ['dp_sp','ep_sp','bm_sp','ntis','tbl','tms','dfy','svar']
df_ma['Index'] = df_ma['Index'].str.replace(',','').astype('float64')
df_ma['dp_sp'] = df_ma['D12']/df_ma['Index']
df_ma['ep_sp'] = df_ma['E12']/df_ma['Index']
df_ma.rename({'b/m':'bm_sp'},axis=1,inplace=True)
df_ma['tms'] = df_ma['lty']-df_ma['tbl']
df_ma['dfy'] = df_ma['BAA']-df_ma['AAA']
df_ma = df_ma[['yyyymm']+ma_predictors]

df_ma.head()

### Get All Features, Split, and Transform
# - Construct interaction terms
# - Split the dataset
# - Transform the data into (-1, 1)

from sklearn.preprocessing import MinMaxScaler

def interactions(data, data_ma, characteristics, ma_predictors):
    # construct interactions between firm characteristics and macroeconomic predictors
    # Convert 'Date' to datetime
    data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m%d')

    # Extract year and month and convert them to 'yyyymm'
    data['DATE'] = data['DATE'].dt.to_period('M').dt.strftime('%Y%m').astype(int)

    # Merge the dataset
    data_ma_long = pd.merge(data[['DATE']], data_ma,left_on='DATE', right_on='yyyymm', how='left').reset_index(drop=True)
    data = data.reset_index(drop=True)
    data_ma_long = data_ma_long.reset_index(drop=True)

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
    train_dataset = data[(data['DATE'] >= train_start_date) & (data['DATE'] < val_start_date)]
    val_dataset = data[(data['DATE'] >= val_start_date) & (data['DATE'] < test_start_date)]
    test_dataset = data[(data['DATE'] >= test_start_date) & (data['DATE'] <= test_end_date)]

    train_dataset['DATE'] = train_dataset['DATE'].astype('int')
    val_dataset['DATE'] = val_dataset['DATE'].astype('int')
    test_dataset['DATE'] = test_dataset['DATE'].astype('int')

    return train_dataset, val_dataset, test_dataset


def transform(data, minmax=True):
    if minmax:
        scaler = MinMaxScaler((-1, 1))
        features = list(set(data.columns).difference({'permno','DATE','RET'}))
        data[features] = scaler.fit_transform(data[features])
    else:
        pass
    print(f"The shape of the data is: {data.shape}")
    return data


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

df, features = interactions(df, df_ma, characteristics, ma_predictors)

train_start_date = 201001
val_start_date = 201601
test_start_date = 201801
test_end_date = 202112

train_dataset, val_dataset, test_dataset = split_by_fix_date(df, train_start_date, val_start_date, test_start_date, test_end_date)

train_dataset = transform(train_dataset, minmax=True)
val_dataset = transform(val_dataset, minmax=True)
test_dataset = transform(test_dataset, minmax=True)

train_dataset.to_csv(r"train_dataset.csv")
val_dataset.to_csv(r"val_dataset.csv")
test_dataset.to_csv(r"test_dataset.csv")

pd.DataFrame(features).to_csv(r"features.csv")

# Delete the datasets to free up memory
del train_dataset
del val_dataset
del test_dataset

import gc
gc.collect()


df_top, features_top = interactions(df_top, df_ma, characteristics, ma_predictors)
train_dataset_top, val_dataset_top, test_dataset_top = split_by_fix_date(df_top, train_start_date, val_start_date, test_start_date, test_end_date)

train_dataset_top = transform(train_dataset_top, minmax=True)
val_dataset_top = transform(val_dataset_top, minmax=True)
test_dataset_top = transform(test_dataset_top, minmax=True)

train_dataset_top.to_csv(r"train_dataset_top.csv")
val_dataset_top.to_csv(r"val_dataset_top.csv")
test_dataset_top.to_csv(r"test_dataset_top.csv")

pd.DataFrame(features_top).to_csv(r"features_top.csv")

# Delete the datasets to free up memory
del train_dataset_top
del val_dataset_top
del test_dataset_top

import gc
gc.collect()

df_bot, features_bot = interactions(df_bot, df_ma, characteristics, ma_predictors)
train_dataset_bot, val_dataset_bot, test_dataset_bot = split_by_fix_date(df_bot, train_start_date, val_start_date, test_start_date, test_end_date)

train_dataset_bot = transform(train_dataset_bot, minmax=True)
val_dataset_bot = transform(val_dataset_bot, minmax=True)
test_dataset_bot = transform(test_dataset_bot, minmax=True)

pd.DataFrame(features_bot).to_csv(r"features_bot.csv")

train_dataset_bot.to_csv(r"train_dataset_bot.csv")
val_dataset_bot.to_csv(r"val_dataset_bot.csv")
test_dataset_bot.to_csv(r"test_dataset_bot.csv")



