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

import io
import boto3
import os
import s3fs
from tqdm import tqdm

import clean_data

%matplotlib inline
from pandas.plotting import register_matplotlib_converters
from matplotlib.font_manager import FontProperties
register_matplotlib_converters()
matplotlib.rcParams['figure.figsize'] = (12,8)
sns.set(font_scale=1.5, style="whitegrid")
plt.style.use('ggplot')
pd.plotting.register_matplotlib_converters()

# %%
s3 = boto3.client('s3')
df = pd.read_pickle('s3://cbh-capstone1-texasrrc/clean_df.pkl')

# %%

def plot_districts(data):
  df_dist = df.groupby(['DISTRICT_NO', 'REPORT_DATE'])['TOTAL_LEASE_FLARE_VOL',
            'LEASE_OIL_PROD_VOL', 'LEASE_CSGD_PROD_VOL', 'LEASE_GAS_PROD_VOL'].sum().reset_index()

  df_dist['REPORT_DATE'] = df_dist['REPORT_DATE'].dt.to_timestamp()
  x_vals = np.arange(0, len(df_dist[df_dist['DISTRICT_NO']==1]),1)
  titles= ['Flare Volumes by District (MMcf)',
          'Oil Production by District (bbl)',
          'Gas Production by District (MMcf)',
          'Flare Gas - Gas Production Ratio by District']

  y_vals = [df_dist['TOTAL_LEASE_FLARE_VOL'] / 1000,
              df_dist['LEASE_OIL_PROD_VOL'] / 1000,
              (df_dist['LEASE_CSGD_PROD_VOL'] +
              df_dist['LEASE_GAS_PROD_VOL']) / 1000,
              (df_dist['TOTAL_LEASE_FLARE_VOL'] /
              df_dist['LEASE_CSGD_PROD_VOL'])]

  y_labels = ['Volume FLared (MMcf)', 'Oil Produced (1000 bbl)',
              'Gas Produced (MMcf)', 'Flare Gas to Produced Gas Ratio']

  for title, y, y_label in zip(titles, y_vals, y_labels):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    sns.lineplot(x=df_dist['REPORT_DATE'],y=y,
                hue=df_dist['DISTRICT_NO'],palette='deep',
                legend='full')
    plt.title(title)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
            borderaxespad=0.)
    ax.set_xticklabels(labels=df_dist['REPORT_DATE'].dt.to_period('M'),
                      rotation = 45,
                      ha='right')
    ax.set_ylabel(y_label)
    plt.show()
  plt.tight_layout()

# %%
plot_districts(df)
# %%
def oil_price_plotting(df):
  oil_price = pd.read_csv('s3://cbh-capstone1-texasrrc/price_of_oil.csv', index_col=0)
  df = pd.merge_ordered(df, oil_price, on=['CYCLE_MONTH', 'CYCLE_YEAR'], how='left')
  df_dist = df.groupby(['DISTRICT_NO', 'REPORT_DATE', 'OIL_PRICE'])['TOTAL_LEASE_FLARE_VOL',
            'LEASE_OIL_PROD_VOL', 'LEASE_CSGD_PROD_VOL', 'LEASE_GAS_PROD_VOL'].sum().reset_index()
  df_dist['REPORT_DATE'] = df_dist['REPORT_DATE'].dt.to_timestamp()
  titles= ['Flare Volumes by District (MMcf)',
          'Oil Production by District (bbl)',
          'Gas Production by District (MMcf)',
          'Flare Gas - Gas Production Ratio by District',
          'Flare Gas - Oil Production Ratio by District']

  y_vals = [df_dist['TOTAL_LEASE_FLARE_VOL'] / 1000,
              df_dist['LEASE_OIL_PROD_VOL'] / 1000,
              (df_dist['LEASE_CSGD_PROD_VOL'] +
              df_dist['LEASE_GAS_PROD_VOL']) / 1000,
              (df_dist['TOTAL_LEASE_FLARE_VOL'] /
              (df_dist['LEASE_CSGD_PROD_VOL'] + df_dist['LEASE_GAS_PROD_VOL'])),
              (df_dist['TOTAL_LEASE_FLARE_VOL'] /
              df_dist['LEASE_OIL_PROD_VOL']) ]

  y_labels = ['Volume FLared (MMcf)', 'Oil Produced (1000 bbl)',
              'Gas Produced (MMcf)', 'Flare Gas to Produced Gas Ratio',
              'Flare Gas to Oil Produced Ratio (Mcf / bbl)']

  for title, y, y_label in zip(titles, y_vals, y_labels):
    # x_vals = np.arange(0, len(df_dist['REPORT_DATE']),1)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlim(df_dist['REPORT_DATE'].min(), df_dist['REPORT_DATE'].max())
    sns.lineplot(x=df_dist['REPORT_DATE'],y=y,
                hue=df_dist['DISTRICT_NO'],palette='deep',
                legend='full')
    ax2 = ax.twinx()
    sns.lineplot(x=df_dist['REPORT_DATE'], y=df_dist['OIL_PRICE'], color='g',
                  ax=ax2, style=True, dashes=[(2,2)], legend='brief')
    plt.title(title)
    plt.grid(True)
    ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left',
            borderaxespad=0.)
    ax2.legend(bbox_to_anchor=(1.1, 0.5), loc='lower left',
            borderaxespad=0.)
    ax2.legend(['Price of Oil'])
    # ax.set_xticklabels(labels=df_dist['REPORT_DATE'].dt.to_period('Y'),
                      # rotation = 45,
                      # ha='right')
    # ax2.set_xticklabels(labels=df_dist['REPORT_DATE'].dt.to_period('M'),
                      # rotation = 45,
                      # ha='right')
    ax.set_ylabel(y_label)
    plt.show()
  plt.tight_layout()

# %%
oil_price_plotting(df)

# %%
def district_boxplot(df):
  df_dist = df.groupby(['DISTRICT_NO', 'REPORT_DATE'])['TOTAL_LEASE_FLARE_VOL',
            'LEASE_OIL_PROD_VOL', 'LEASE_CSGD_PROD_VOL', 'LEASE_GAS_PROD_VOL'].sum().reset_index()
  titles= ['Flare Volumes by District (MMcf)',
          'Oil Production by District (bbl)',
          'Gas Production by District (MMcf)',
          'Flare Gas - Gas Production Ratio by District',
          'Flare Gas - Oil Production Ratio by District']

  y_vals = [df_dist['TOTAL_LEASE_FLARE_VOL'] / 1000,
              df_dist['LEASE_OIL_PROD_VOL'] / 1000,
              (df_dist['LEASE_CSGD_PROD_VOL'] +
              df_dist['LEASE_GAS_PROD_VOL']) / 1000,
              (df_dist['TOTAL_LEASE_FLARE_VOL'] /
              (df_dist['LEASE_CSGD_PROD_VOL'] + df_dist['LEASE_GAS_PROD_VOL'])),
              (df_dist['TOTAL_LEASE_FLARE_VOL'] /
              df_dist['LEASE_OIL_PROD_VOL']) ]

  y_labels = ['Volume FLared (MMcf)', 'Oil Produced (1000 bbl)',
              'Gas Produced (MMcf)', 'Flare Gas to Produced Gas Ratio',
              'Flare Gas to Oil Produced Ratio (Mcf / bbl)']

  for title, y, y_label in zip(titles, y_vals, y_labels):
    # x_vals = np.arange(0, len(df_dist['REPORT_DATE']),1)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    sns.boxplot(x=df_dist['DISTRICT_NO'],y=y, width=0.8,
                hue=df_dist['DISTRICT_NO'],palette='deep', dodge=False)
    plt.title(title)
    plt.grid(True)
    ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left',
            borderaxespad=0., title='DISTRICT_NO')
    # ax.set_xlim(0, max(df_dist['DISTRICT_NO'].unique()))
    # ax.set_xticklabels(labels=df_dist['DISTRICT_NO'].unique())
    ax.set_ylabel(y_label)
    # ax.set_xticks()
    plt.savefig('boxplot'+ title)
    plt.show()

  plt.tight_layout()

# %%
district_boxplot(df)
# %%

df_dist = df.groupby(['DISTRICT_NO', 'REPORT_DATE'])['TOTAL_LEASE_FLARE_VOL',
            'LEASE_OIL_PROD_VOL', 'LEASE_CSGD_PROD_VOL', 'LEASE_GAS_PROD_VOL'].sum().reset_index()
df_dist['DISTRICT_NO'].unique()
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
# %%

# %%

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

df_2000['WASTE_RATIO'] = df_2000['LEASE_FLARE_ENERGY (GWH)'] / df_2000['TOTAL_ENERGY_PROD (GWH)']
## Drop inf values w/ np.where(WASTE_RATIO) = np.inf
# %%
inf_idx = np.where(df_2000['WASTE_RATIO'] == np.inf)[0]
df_2000.drop(inf_idx, axis=0, inplace=True)
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
x = sns.distplot(largest_df['WASTE_RATIO'], kde=False)
x.set_title('Waste Ratio Distribution of Top 100')
plt.show(x)
# %%
largest_df['OPERATOR_NAME_x'].value_counts()
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

# %%
df_2000['WASTE_RATIO'].describe()
# %%
lease_df[lease_df['WASTE_RATIO'] ==lease_df['WASTE_RATIO'].max()]
# %%
lease_df.iloc[6892846, :]
# %%
worst_lease = df_2000[df_2000['LEASE_NO'] == 43213]
# %%
sns.lineplot(x=worst_lease['MONTHS_FROM_FIRST_REPORT'], y=worst_lease['WASTE_RATIO'])
# %%
worst_lease[['LEASE_NO', 'MONTH', 'YEAR', 'OPERATOR_NO_x', 'OPERATOR_NAME_x', 'LEASE_FLARE_ENERGY (GWH)', 'TOTAL_ENERGY_PROD (GWH)', 'WASTE_RATIO']]
# %%
largest_df.head()
# %%
if __name__ == '__main__':
  flare_cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR' , 'CYCLE_MONTH',
                    'OPERATOR_NO','OPERATOR_NAME',
                    'LEASE_CSGD_DISPCDE04_VOL', 'LEASE_GAS_DISPCD04_VOL']

  prod_cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR' , 'CYCLE_MONTH',
                    'OPERATOR_NO','OPERATOR_NAME', 'LEASE_OIL_PROD_VOL',
                    'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL',
                    'LEASE_CSGD_PROD_VOL']


  data = clean_data.s3_to_df('cbh-capstone1-texasrrc',
                'OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv',
                'OG_LEASE_CYCLE_DATA_TABLE.dsv',
                flare_cols, prod_cols,
                chunk_data=True, chunksize=15000000, year=2015)
# %%
data.head()