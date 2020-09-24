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
    plt.show()

  plt.tight_layout()


def operator_eda(data, num_ops):
  '''investigates the contributions of the
    top n operators contibuting to flaring'''

  op_df = data.groupby(['OPERATOR_NAME'])['TOTAL_LEASE_FLARE_VOL',
            'LEASE_OIL_PROD_VOL', 'LEASE_CSGD_PROD_VOL', 'LEASE_GAS_PROD_VOL'].sum()
  n = op_df.shape[0]
  top_100_names = list(op_df['TOTAL_LEASE_FLARE_VOL'].nlargest(num_ops, keep='first').index)
  other_names = list(op_df['TOTAL_LEASE_FLARE_VOL'].nsmallest(n-num_ops, keep='first').index)
  df = data.groupby(['OPERATOR_NAME', 'CYCLE_YEAR'])['TOTAL_LEASE_FLARE_VOL',
            'LEASE_OIL_PROD_VOL', 'LEASE_CSGD_PROD_VOL',
            'LEASE_GAS_PROD_VOL'].sum()
  top_100 = df.loc[top_100_names]
  else_df = df.loc[other_names]
  top_vals = []
  else_vals = []
  x_vals = np.unique([i[1] for i in df.index])
  for y in x_vals:
    top_vals.append(top_100['TOTAL_LEASE_FLARE_VOL'].unstack(level=1).fillna(0)[y].sum() / 1000)
    else_vals.append(else_df['TOTAL_LEASE_FLARE_VOL'].unstack(level=1).fillna(0)[y].sum() / 1000)
  y_vals = [top_vals, else_vals]
  pal = sns.color_palette("Set1")
  plt.stackplot(x_vals, y_vals, labels=['Top 25 Operators', 'Everyone Else'], colors=pal, alpha=0.5)
  plt.title('Operator Contribution - Flaring')
  plt.xlabel('YEAR')
  plt.ylabel('Flare Volume (MMcf')
  plt.legend(loc='upper left')
  plt.show()
  display(top_100_names)


# %%
def decay_eda(data):
  '''factors in the months from first report feature to account for
  decay in production'''
  pass

def worst_leases_eda(data):
  '''investigates the leases with the worst flaring ratios'''
  pass

# %%
if __name__ == '__main__':

  s3 = boto3.client('s3')
  df = clean_data.main()
  # df = pd.read_pickle('s3://cbh-capstone1-texasrrc/clean_df.pkl')

  plot_districts(df)

  oil_price_plotting(df)

  district_boxplot(df)

  num_ops = 25
  operator_eda(df, num_ops)

