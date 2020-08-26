# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import boto3
import os
import s3fs
import numpy as np
import matplotlib

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)
plt.style.use('ggplot')
# %%
df_dist_year = pd.read_csv('s3://cbh-capstone1-texasrrc/group_by_district_yr.csv')
# %%
## Load in larger data set
s3 = boto3.client('s3')

# %%
df = pd.read_csv('s3://cbh-capstone1-texasrrc/merged_flare_og.csv')
# %%
df.drop('Unnamed: 0', axis=1, inplace=True)
# %%
df.to_csv('s3://cbh-capstone1-texasrrc/merged_flare_og.csv')
# %%
a = sns.lineplot(x=df_dist_year['YEAR'],
            y=df_dist_year['TOTAL_LEASE_FLARE_VOL'],
            hue=df_dist_year['DISTRICT_NO'],
            palette='coolwarm',
            legend='full'
            )
a.set_title('Flare Volumes by District')
# %%
b = sns.pairplot(df_dist_year.loc[:,['DISTRICT_NO','LEASE_OIL_PROD_VOL',
                            'LEASE_GAS_PROD_VOL',
                            'LEASE_COND_PROD_VOL',
                            'LEASE_CSGD_PROD_VOL',
                            'TOTAL_LEASE_FLARE_VOL']],
                  hue = 'DISTRICT_NO',
                  palette='coolwarm')
# %%
districts = range(1,15)
fig, ax = plt.subplots(7,2)
for d, subplot in zip(districts, ax.flatten()):
  sns.pairplot(df_dist_year[df_dist_year['DISTRICT_NO'] == d].loc[:,['LEASE_OIL_PROD_VOL',
                            'LEASE_GAS_PROD_VOL',
                            'LEASE_COND_PROD_VOL',
                            'LEASE_CSGD_PROD_VOL',
                            'TOTAL_LEASE_FLARE_VOL']],
                        palette='coolwarm')


# %%
df_2008_plus = df_dist_year[df_dist_year['YEAR'] >= 2008]
df_2008_plus = df_2008_plus[df_dist_year['YEAR'] < 2020]
df_2007_less = df_dist_year[df_dist_year['YEAR'] < 2008]
# %%
c = sns.lineplot(x=df_2008_plus['YEAR'],
            y=df_2008_plus['TOTAL_LEASE_FLARE_VOL'],
            hue=df_2008_plus['DISTRICT_NO'],
            palette='coolwarm',
            legend='full'
            )
c.set_title('Flare Volumes by District (2008-2019)')
# %%
df_2008_plus.head()
# %%
d = sns.lineplot(x=df_2007_less['YEAR'],
            y=df_2007_less['TOTAL_LEASE_FLARE_VOL'],
            hue=df_2007_less['DISTRICT_NO'],
            palette='coolwarm',
            legend='full'
            )
d.set_title('Flare Volumes by District (1993-2008)')
# %%
df['CYCLE_YEAR_MONTH'] = pd.to_datetime(df['CYCLE_YEAR_MONTH'], yearfirst=True, format='%Y%m')
# %%
df['MONTH'] = df['CYCLE_YEAR_MONTH'].dt.month
df['YEAR'] = df['CYCLE_YEAR_MONTH'].dt.year
df.head()
# %%
df_2000_plus = df[df['YEAR'] >= 2000]
# %%
df_2000_plus.info()
# %%
df_2000_plus.drop(['CYCLE_YEAR_MONTH', 'OPERATOR_NO_y', 'OPERATOR_NAME_y'], axis=1, inplace=True)
# %%
df_2000_plus.columns
# %%
df_2000_plus = df_2000_plus[['DISTRICT_NO', 'LEASE_NO',
                              'MONTH', 'YEAR','OPERATOR_NO_x',
                              'OPERATOR_NAME_x', 'LEASE_OIL_PROD_VOL',
                              'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL',
                              'LEASE_CSGD_PROD_VOL', 'GAS_FLARED',
                              'CASINGHEAD_GAS_FLARED', 'TOTAL_LEASE_FLARE_VOL' ]]
# %%
df_2000_plus.to_pickle('2000_plus_full.pkl')
# %%
df_2000_plus.head()

# %%
oil_price = pd.read_csv('price_of_oil.csv', )
# %%
oil_price.drop('Unnamed: 0', axis=1, inplace=True)
# %%
oil_price['Month'] = pd.to_datetime(oil_price['Month'])
oil_price['YEAR'] = oil_price['Month'].dt.year
oil_price['MONTH'] = oil_price['Month'].dt.month
# %%
oil_price.drop('Month', axis=1, inplace=True)
# %%
oil_price.head()

# %%
oil_price.info()

# %%
df_2000_plus = pd.merge_ordered(df_2000_plus, oil_price, on=['YEAR', 'MONTH'], how='inner')
# %%
df_2000_plus.info()

# %%
operators_df = df_2000_plus.groupby(['OPERATOR_NAME_x', 'OPERATOR_NO_x'])['LEASE_NO'].count().reset_index()
operators_df.head()
# %%
operators_df['LEASE_NO'].nlargest(100, keep='all')
# %%
operators_df.sort_values(['LEASE_NO'], axis=0, ascending=False)
# %%
e = sns.lineplot(x=df_2008_plus['YEAR'],
            y=(df_2008_plus['TOTAL_LEASE_FLARE_VOL'] / df_2008_plus['LEASE_OIL_PROD_VOL']),
            hue=df_2008_plus['DISTRICT_NO'],
            palette='coolwarm',
            legend='full'
            )
e.set_title('Flare / Oil Ratio by District (2008-2019')
# %%
e = sns.lineplot(x=df_2008_plus['YEAR'],
            y=(df_2008_plus['TOTAL_LEASE_FLARE_VOL'] / df_2008_plus['LEASE_CSGD_PROD_VOL']),
            hue=df_2008_plus['DISTRICT_NO'],
            palette='coolwarm',
            legend='full'
            )
e.set_title('Flare / CSGD Ratio by District (2008-2019')
# %%
operators_df_flare = df_2000_plus.groupby(['OPERATOR_NAME_x', 'OPERATOR_NO_x'])['LEASE_OIL_PROD_VOL',
                                                                  'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL',
                                                                  'LEASE_CSGD_PROD_VOL', 'TOTAL_LEASE_FLARE_VOL'].sum().reset_index()
# %%
operators_df_flare.head()

# %%
operators_df = pd.merge(operators_df, operators_df_flare, on=['OPERATOR_NAME_x', 'OPERATOR_NO_x'], how='inner')
# %%
operators_df.columns
# %%
op_bins = pd.cut(operators_df['LEASE_NO'], bins=3, retbins=True)
# %%
operators_df['FLARE-OIL-BBL RATIO'] = operators_df['TOTAL_LEASE_FLARE_VOL'] / operators_df['LEASE_OIL_PROD_VOL']
# %%
operators_df['AVG_FLARE_PER_LEASE'] = operators_df['TOTAL_LEASE_FLARE_VOL'] / operators_df['LEASE_NO']
# %%
operators_df.head()
# %%
lease_df = df_2000_plus.groupby(['LEASE_NO'])['YEAR', 'MONTH'].min().reset_index()
# %%

lease_df['MONTH_DT'] = (pd.to_datetime(lease_df['MONTH'], infer_datetime_format=False, format='%m'))
lease_df['YEAR_DT'] = (pd.to_datetime(lease_df['YEAR'], infer_datetime_format=False, format='%Y'))
# %%
lease_df['FIRST_PROD_REPORT'] = lease_df['MONTH'].map(str) + '-' + lease_df['YEAR'].map(str)
# %%
lease_df['FIRST_PROD_REPORT'] = pd.to_datetime(lease_df['FIRST_PROD_REPORT'], yearfirst=False, format='%m-%Y').dt.to_period('M')
# %%
lease_df.head()
# %%
lease_df_max = df_2000_plus.groupby(['LEASE_NO'])['YEAR', 'MONTH'].max().reset_index()
# %%
lease_df_max['LAST_REPORT'] = lease_df_max['MONTH'].map(str) + '-' + lease_df_max['YEAR'].map(str)
# %%
lease_df_max['LAST_REPORT'] = pd.to_datetime(lease_df_max['LAST_REPORT'], yearfirst=False, format='%m-%Y').dt.to_period('M')
# %%
lease_df_max.head()
# %%
lease_df.head()

# %%
df_2000_plus = pd.read_pickle('2000_plus_full.pkl')
# %%
df_2000_plus = pd.merge_ordered(df_2000_plus, lease_df[['LEASE_NO', 'FIRST_PROD_REPORT']], on=['LEASE_NO'], how='left')
# %%
df_2000_plus = pd.merge_ordered(df_2000_plus, lease_df_max[['LEASE_NO', 'LAST_REPORT']], on=['LEASE_NO'], how='left')
# %%

# %%
df_2000_plus['REPORT_DATE'] = df_2000_plus['MONTH'].map(str) + '-' + df_2000_plus['YEAR'].map(str)
df_2000_plus['REPORT_DATE'] = pd.to_datetime(df_2000_plus['REPORT_DATE'], yearfirst=False, format='%m-%Y').dt.to_period('M')
# %%
df_2000_plus.info()
# %%
df_2000_plus.head()
# %%
df_2000_plus['MONTHS_FROM_FIRST_REPORT'] = df_2000_plus['REPORT_DATE'].astype('int') - df_2000_plus['FIRST_PROD_REPORT'].astype('int')
# %%
df_2000_plus.head()
# %%
df_2000_plus = pd.merge(df_2000_plus, oil_price, on=['YEAR', 'MONTH'], how='left')
# %%
pd.to_pickle(df_2000_plus, 'df_2020_plus_rptdate.pkl')
# %%
fig, axs = plt.subplots(2)

axs[0] = sns.lineplot(x=df_2000_plus['YEAR'],
            y=(df_2000_plus['TOTAL_LEASE_FLARE_VOL'] / df_2000_plus['LEASE_OIL_PROD_VOL']),
            hue=df_2000_plus['DISTRICT_NO'],
            palette='coolwarm',
            legend='full'
            )

axs[1] = sns.lineplot(x=df_2000_plus['YEAR'],
            y=(df_2000_plus['Price of Oil']),
            palette='coolwarm',
            legend='full')

axs[0].set_title('Flare / Oil Ratio by District (2000-2019')
axs[1].set_title('Price of Oil (2000-2019)')
# %%
dist_year_grouped = df_2000_plus.groupby(['DISTRICT_NO', 'YEAR','Price of Oil','MONTHS_FROM_FIRST_REPORT'])['LEASE_OIL_PROD_VOL',
                            'LEASE_GAS_PROD_VOL',
                            'LEASE_COND_PROD_VOL',
                            'LEASE_CSGD_PROD_VOL',
                            'TOTAL_LEASE_FLARE_VOL'].sum().reset_index()
# %%
## group operators and year, sum up total leases for year and other factors

operators_df_year = df_2000_plus.groupby(['OPERATOR_NAME_x',
                                          'OPERATOR_NO_x', 'YEAR']).agg({'LEASE_OIL_PROD_VOL': 'sum',
                                                                          'LEASE_GAS_PROD_VOL': 'sum',
                                                                          'LEASE_COND_PROD_VOL' : 'sum',
                                                                          'LEASE_CSGD_PROD_VOL' : 'sum',
                                                                          'TOTAL_LEASE_FLARE_VOL' : 'sum',
                                                                          'LEASE_NO': 'count'}).reset_index()
# %%
operators_df_year.sort_values(by='YEAR')
# %%
year_arr = np.array(operators_df_year['YEAR'])
lease_count_arr = np.array(operators_df_year['LEASE_NO'])
# %%
operators_df_year[operators_df_year['YEAR'] == 2000]

# %%%
# df_2000_plus.head()
pd.to_pickle(df_2000_plus, 's3://cbh-capstone1-texasrrc/df_2000_plus.pkl')

# %%
