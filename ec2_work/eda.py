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
sns.set_style(style="whitegrid")
plt.style.use('ggplot')
# %%
df_dist_year = pd.read_csv('s3://cbh-capstone1-texasrrc/group_by_district_yr.csv')
# %%
## Load in larger data set
s3 = boto3.client('s3')

# %%
df = pd.read_pickle('s3://cbh-capstone1-texasrrc/df_2000_plus.pkl')
# %%
df.head()
# %%
df.to_pickle('df.pkl')
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

# %%

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
# df_2000 = pd.read_pickle('s3://cbh-capstone1-texasrrc/df_2000_plus.pkl')
df_2000 = pd.read_pickle('df_2000.pkl')

# %%
df_2000['WASTE_RATIO'].fillna(0)
# %%
df_district_yr = df_2000.groupby(['DISTRICT_NO', 'YEAR']).agg({
                        'LEASE_OIL_PROD_VOL' : 'sum',
                        'LEASE_GAS_PROD_VOL': 'sum',
                        'LEASE_COND_PROD_VOL' : 'sum',
                        'LEASE_CSGD_PROD_VOL' : 'sum',
                        'TOTAL_LEASE_FLARE_VOL': 'sum',
                        'LEASE_NO' : 'count',
                        'Price of Oil' : 'mean'}).reset_index()
# %%
df_district_yr.head()
# %%
ratio = df_district_yr['TOTAL_LEASE_FLARE_VOL'] / df_district_yr['LEASE_OIL_PROD_VOL']
g = sns.lineplot(x=df_district_yr['YEAR'], y=ratio,
                  hue=df_district_yr['DISTRICT_NO'], palette='coolwarm',
                  legend='full')
g.set_title('Flare Vol per Oil Bbl Produced (2000-2020)')
plt.show()
h = sns.lineplot(x=df_district_yr['YEAR'], y=df_district_yr['Price of Oil'],
                    )
h.set_title('Price of Oil')
plt.show()

# %%
districts = list(df_district_yr['DISTRICT_NO'].value_counts().index)
df_district_yr['Flare-Oil Ratio'] = df_district_yr['TOTAL_LEASE_FLARE_VOL'] / df_district_yr['LEASE_OIL_PROD_VOL']
df_district_yr[df_district_yr['DISTRICT_NO'] == 1].head()


# %%
oil_kwh = 1700 #per bbl
gas_kwh = 293 #per mcf
cond_kwh = 1589 #per bbl (0.935 bbl oil = 1 bbl condensate)

df_2000['LEASE_OIL_PROD_ENERGY (GWH)'] = (df['LEASE_OIL_PROD_VOL'] * oil_kwh) / 1000000
df_2000['LEASE_GAS_PROD_ENERGY (GWH)'] = (df['LEASE_GAS_PROD_VOL'] * gas_kwh) / 1000000
df_2000['LEASE_CSGD_PROD_ENERGY (GWH)'] = (df['LEASE_CSGD_PROD_VOL'] * gas_kwh) / 1000000
df_2000['LEASE_COND_PROD_ENERGY (GWH)'] = (df['LEASE_COND_PROD_VOL'] * cond_kwh) / 1000000
df_2000['LEASE_FLARE_ENERGY (GWH)'] = (df['TOTAL_LEASE_FLARE_VOL'] * gas_kwh) / 1000000
df_2000['TOTAL_ENERGY_PROD (GWH)'] = (df_2000['LEASE_COND_PROD_ENERGY (GWH)'] +
                                    df_2000['LEASE_CSGD_PROD_ENERGY (GWH)'] +
                                    df_2000['LEASE_GAS_PROD_ENERGY (GWH)'] +
                                    df_2000['LEASE_OIL_PROD_ENERGY (GWH)'])

df_2000['WASTE_RATIO'] = df_2000['LEASE_FLARE_ENERGY (GWH)'] / df_2000['TOTAL_ENERGY_PROD (GWH)']
## Drop inf values w/ np.where(WASTE_RATIO) = np.inf
# %%
inf_idx = np.where(df_2000['WASTE_RATIO'] == np.inf)[0]
df_2000.drop(inf_idx, axis=0, inplace=True)
# %%
for d in districts:
  num_bins = int(1 + 3.332*np.log(len(df_2000[df_2000['DISTRICT_NO'] == 1]['WASTE_RATIO'])))*2
  j = sns.distplot(df_2000[df_2000['DISTRICT_NO'] == d]['WASTE_RATIO'] , kde=False, axlabel='Flare Ratio',
                                   label=f'District {d}', norm_hist=True,
                                   bins=num_bins)
  j.set_title(f'District Waste Ratio Distribution', )
  j.set(xlim=(-.01,0.2))
  j.legend()
plt.show(j)


# %%



# %%
operators_df = df_2000.groupby(['OPERATOR_NO_x', 'OPERATOR_NAME_x']).agg({'TOTAL_ENERGY_PROD (GWH)' : 'sum',
                        'LEASE_FLARE_ENERGY (GWH)': 'sum',
                        'WASTE_RATIO': 'mean', 'LEASE_NO' : 'count'}).reset_index()
# %%
k = sns.distplot(operators_df['WASTE_RATIO'], kde=False,
                  bins=100
                  , norm_hist=True)
k.set_title(f'Operator Waste Ratio Distribution')
k.set(xlim=(-.01,0.2))
plt.show(k)
# %%
l = sns.distplot(operators_df['LEASE_NO'], kde=False,
                  bins=200, norm_hist=True)
l.set(xlim=(-100, 40000))
l.set_title(f'Operator Lease Number Distribution')
plt.show(l)
# %%
sns.pairplot(operators_df[['TOTAL_ENERGY_PROD (GWH)', 'LEASE_NO','LEASE_FLARE_ENERGY (GWH)' ]])
# %%
lease_df = df_2000.groupby(['LEASE_NO', 'MONTHS_FROM_FIRST_REPORT'])['WASTE_RATIO', 'TOTAL_ENERGY_PROD (GWH)'].sum().reset_index()
# %%
lease_df_sample = lease_df.sample(frac=0.003) #0.3% sample
lease_df_sample.shape
# %%
m = sns.scatterplot(x=lease_df_sample['MONTHS_FROM_FIRST_REPORT'], y=lease_df_sample['TOTAL_ENERGY_PROD (GWH)'])
m.set(ylim=(0,1000))
m.set_title('Energy Production After First Report')
plt.show(m)

n = sns.scatterplot(x=lease_df_sample['MONTHS_FROM_FIRST_REPORT'], y=lease_df_sample['WASTE_RATIO'])
n.set(ylim=(0,1.25))
n.set_title('Waste Ratio After First Report')
plt.show(n)

# %%
df_2000['WASTE_RATIO'].nlargest(20)
# %%
df_2000.to_pickle('df_2000.pkl')
# %%
lease_df.to_pickle('lease_df.pkl')
operators_df.to_pickle('operators_df.pkl')
# %%
df_2000 = pd.read_pickle('df_2000.pkl')

# %%
largest = df_2000['WASTE_RATIO'].nlargest(100).index
largest_df = df_2000.loc[largest, :]
# %%
np.std(df_2000['WASTE_RATIO'])
largest_df.head()
# %%
sns.distplot(largest_df['WASTE_RATIO'], kde=False)
# %%
largest_df['OPERATOR_NAME_x'].value_counts()[:25]
# %%
largest_df.head()
# %%
top_100 = list(df_2000.groupby('OPERATOR_NAME_x')['WASTE_RATIO'].mean().nlargest(100).index)
# %%
df_2000.iloc[largest, :]
# %%
bins= [0,2,14900]
test_cute = pd.cut(df_2000['WASTE_RATIO'], bins=bins, include_lowest=True)


# %%
test_df = pd.concat(df_2000, test_cute)
# %%
# %%
df_2000['BIN'] = test_cute

# %%
company_dums = pd.get_dummies(df_2000['BIN'], prefix='COMPANY_CAT')
# %%
df_2000.drop('BIN', axis=1, inplace=True)
# %%
df_2000 = pd.concat((df_2000, company_dums), axis=1)
# %%
df_2000.to_pickle('df_2000.pkl')
# %%
df_2000 = pd.read_pickle('df_2000.pkl')
# %%
df_2000.info()

# %%
samp_df = df_2000.sample(frac=0.25, random_state=42)

# %%
sns.boxplot(x=samp_df[samp_df['COMPANY_CAT_(-0.001, 2.0]'] == 1]['DISTRICT_NO'],
                      y=samp_df[samp_df['COMPANY_CAT_(-0.001, 2.0]'] == 1]['WASTE_RATIO'], data=samp_df)
plt.ylim(-0.5, 1.5)
plt.show()

# %%
sns.boxplot(x=samp_df[samp_df['COMPANY_CAT_(-0.001, 2.0]'] == 0]['DISTRICT_NO'],
                      y=samp_df[samp_df['COMPANY_CAT_(-0.001, 2.0]'] == 0]['WASTE_RATIO'], data=samp_df)
plt.ylim(-0.5, 100)
plt.show()
# %%
sns.distplot(samp_df['WASTE_RATIO'], kde=False, bins=25)

# %%
# stats = pd.DataFrame(index=['Overall'], columns=['Min', '25th', 'Median', '75th', 'Max', 'SD'])
quartiles = np.percentile(df_2000['TOTAL_LEASE_FLARE_VOL'], [25,50,75])
min_, max_ = df_2000['WASTE_RATIO'].min(), df_2000['WASTE_RATIO'].max()
std_ = np.std(df_2000['WASTE_RATIO'])
districts = [1,2,3,4,5,6,7,8,9,10,11,13,14]
for d in districts:
  quartiles = np.percentile(df_2000[df_2000['DISTRICT_NO']==d]['TOTAL_LEASE_FLARE_VOL'], [25,50,75])
  min_, max_ = df_2000[df_2000['DISTRICT_NO']==d]['TOTAL_LEASE_FLARE_VOL'].min(), df_2000[df_2000['DISTRICT_NO']==d]['TOTAL_LEASE_FLARE_VOL'].max()
  std_ = np.std(df_2000[df_2000['DISTRICT_NO']==d]['TOTAL_LEASE_FLARE_VOL'])
  print(f'District {d}:{round(min_,4)},{round(quartiles[0],4)},{round(quartiles[1],4)}, {round(quartiles[2],4)}, {round(max_,4)}, STD: {round(std_,4)}')
  print(df_2000[df_2000['DISTRICT_NO']==d]['TOTAL_LEASE_FLARE_VOL'].describe())
  
  # print(f'Max: {round(max_,2)}')
  # print(f'25th Percentile: {round(quartiles[0],2)}')
  # print(f'Median: {round(quartiles[1],2)}')
  # print(f'75th Percentile: {round(quartiles[2],2)}')
  # print(f'Max: {round(max_,2)}')
  # print(f'STD: {round(std_,2)}')
# %%
df_2000['WASTE_RATIO'].describe()
# %%
