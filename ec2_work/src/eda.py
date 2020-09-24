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
# df = clean_data.main()
# %%
df = pd.read_pickle('s3://cbh-capstone1-texasrrc/clean_df.pkl')
df.head()
# %%

def plot_districts(data):
  '''
  Plots lineplots that focus on district by district compairson

  Parameters:
  Pandas dataframe

  Returns:
  None
  '''

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
  '''
  Plots various ratios of production to flaring and oil price to
  depict relationship between price, production and flaring.

  Parameters:
  Pandas dataframe

  Returns:
  None
        '''
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
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlim(df_dist['REPORT_DATE'].min(), df_dist['REPORT_DATE'].max())
    sns.lineplot(x=df_dist['REPORT_DATE'],y=y,
                palette='deep',
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
    ax.set_ylabel(y_label)
    plt.show()
  plt.tight_layout()

# %%
oil_price_plotting(df)

# %%
def district_boxplot(df):

  '''Boxplots of range and IQR of vairous informative ratios
        Parameters:
        Pandas dataframe

        Returns:
        None
   '''
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
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    sns.boxplot(x=df_dist['DISTRICT_NO'],y=y, width=0.8,
                hue=df_dist['DISTRICT_NO'],palette='deep', dodge=False)
    plt.title(title)
    plt.grid(True)
    ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left',
            borderaxespad=0., title='DISTRICT_NO').set_visible(False)
    ax.set_ylabel(y_label)
    # plt.savefig('boxplot'+ title)
    plt.show()

  plt.tight_layout()

# %%
district_boxplot(df)
# %%
# %%
def operator_eda(data):
  op_df = data.groupby(['OPERATOR_NAME'])['CYCLE_YEAR','TOTAL_LEASE_FLARE_VOL',
            'LEASE_OIL_PROD_VOL', 'LEASE_CSGD_PROD_VOL', 'LEASE_GAS_PROD_VOL'].reset_index()
  print(op_df.head())
  top_100_names = list(op_df.groupby('OPERATOR_NAME')['TOTAL_LEASE_FLARE_VOL'].sum().nlargest(100, keep='first').index)
  print(top_100_names)
  top_100_mask = op_df.index.isin([top_100_names])
  top_100 = op_df.loc[top_100_names, :]
  print(top_100.shape)
  else_df = op_df[~top_100_mask]
  print(else_df.shape)
  top_vals = []
  else_vals = []
  x_vals = op_df['CYCLE_YEAR'].unique()
  for y in x_vals:
    top_vals.append(top_100[top_100['CYCLE_YEAR']== y ]['TOTAL_LEASE_FLARE_VOL'].sum() / 1000)
    else_vals.append(else_df[else_df['CYCLE_YEAR']== y ]['TOTAL_LEASE_FLARE_VOL'].sum() / 1000)
  print(len(else_vals))
  print(len(top_vals))
  y_vals = [top_vals, else_vals]
  plt.stackplot(x_vals, y_vals, labels=['Top 100 Operators', 'Everyone Else'])
  # plt.xlabel(np.unique(op_df['CYCLE_YEAR']))
  plt.legend(loc='upper left')
  plt.show()

# %%
test = df.groupby(['OPERATOR_NAME'])['TOTAL_LEASE_FLARE_VOL',
            'LEASE_OIL_PROD_VOL', 'LEASE_CSGD_PROD_VOL', 'LEASE_GAS_PROD_VOL'].sum().reset_index()
# top_100 = op_df.nlargest(100, 'TOTAL_LEASE_FLARE_VOL', keep='all')
test_100 = test[['OPERATOR_NAME','TOTAL_LEASE_FLARE_VOL']].nlargest(100, 'TOTAL_LEASE_FLARE_VOL')
test_100.head()
# %%
test_100['OPERATOR_NAME']
# %%
operator_eda(df)
# %%

df['OPERATOR_NAME'].value_counts()
# %%
df.shape
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