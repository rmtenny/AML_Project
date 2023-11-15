import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mstats
import torch
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('equity_chars_ret.csv', index_col = 0)
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

# Create a plot for missing percentage with color scale
fig = px.bar(missing_percentage, x=missing_percentage.index, y=missing_percentage.values,
             labels={'index': 'Features', 'y': 'Missing Percentage'},
             title='Feature Missing Percentage',
             template='plotly_dark',
             color=missing_percentage.values,  # Use the values for color scale
             color_continuous_scale='Viridis',  # Specify the color scale
             )

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

# Plot using plotly
fig = px.bar(correlation_df_all, x='Feature', y='Correlation', color=significance_color,
             labels={'Correlation': 'Correlation with RET (%)', 'P-Value': 'P-Value'},
             title='Correlation of Features with Return and Significance',
             template='plotly_dark')

# Show the plot
fig.show()

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

correlation_df_all

print(selected_characteristics)
print(selected_characteristics1)
print(selected_characteristics2)


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

        # Plot using plotly
        fig = px.bar(correlation_df, x='Feature', y='Correlation', color='P-Value',
                    labels={'Correlation': 'Correlation with RET (%)', 'P-Value': 'P-Value'},
                    title=f'sic2 = {sic}: Correlation of Features with Return and Significance',
                    template='plotly_dark')

        # Show the plot
        fig.show()
# fill na with cross-sectional median
for feature in characteristics:
     df[feature] = df.groupby('DATE')[feature].transform(lambda x: x.fillna(x.median()))

df.isnull().sum()

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



