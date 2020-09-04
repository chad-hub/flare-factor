# %%
#initialize workspace
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import io
import boto3
import os
import s3fs
import numpy as np


# %%
def s3_to_df(bucket_name, filename, chunk_data=True, chunksize=100000):
  '''
  Establish connection to Amazon s3 bucket, retrieve chunks of data (unless working with EC2 that can handle the whole file),
  clean columns and return dataframes.

  Parameters:
  s3 bucket name: where data is stored
  filename_1: Flaring data
  filename_2: production data
  chunk_data: Boolean to determine if data needs to be handled in chunks
  size of chunks (if working with personal computer)

  Returns:
  Dataframe

  '''
  s3 = boto3.client('s3') #establish boto3
  if chunk_data:
    textfile = pd.read_csv('s3://'+bucket_name+'/'+filename,
    delimiter='}', chunksize=chunksize,)
    df = textfile.get_chunk(chunksize)
  else:
    df = pd.read_csv('s3://'+ bucket_name + '/' + filename,
    delimiter='}', chunksize=chunksize, skiprows=30000000)
  return df

# %%
stuff = s3_to_df('cbh-capstone1-texasrrc',
                    'OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv',
                    chunk_data=True, chunksize=100000)
thing = s3_to_df('cbh-capstone1-texasrrc',
                    'OG_LEASE_CYCLE_DATA_TABLE.dsv',
                    chunk_data=True, chunksize=100000)

thing.head()

# %%

haha = merge_dfs(stuff, thing, flare_cols, prod_cols)

haha.head()
# %%
def merge_dfs(df1, df2, cols1, cols2):
  '''Merge production + flare data'''
  df1 = df1[cols1]
  print(df1.shape)
  df2 = df2[cols2]
  print(df2.shape)
  print(df1.merge(df2, how='left').shape)
  df = df1.merge(df2, how='inner')
  print(df.shape)
  return df

# %%
merged_df = merged_df.reindex(columns=['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR_MONTH', 'OPERATOR_NO_x', 'OPERATOR_NO_y', 'OPERATOR_NAME_x', 'OPERATOR_NAME_y', 'LEASE_OIL_PROD_VOL',
       'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL', 'LEASE_CSGD_PROD_VOL', 'GAS_FLARED', 'CASINGHEAD_GAS_FLARED', 'TOTAL_LEASE_FLARE_VOL' ])

test = pd.to_datetime(df['CYCLE_YEAR_MONTH'], yearfirst=True, format='%Y%m') + MonthEnd()

# %%
df_districts = df.groupby(['DISTRICT_NO', 'YEAR'])['LEASE_OIL_PROD_VOL',
                            'LEASE_GAS_PROD_VOL',
                            'LEASE_COND_PROD_VOL',
                            'LEASE_CSGD_PROD_VOL',
                            'TOTAL_LEASE_FLARE_VOL'].sum().reset_index()

# %%

# %%
if __name__ == '__main__':
  flare_df = s3_to_df('cbh-capstone1-texasrrc',
                    'OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv',
                    chunk_data=True, chunksize=100000)

  prod_df = s3_to_df('cbh-capstone1-texasrrc',
                    'OG_LEASE_CYCLE_DATA_TABLE.dsv',
                    chunk_data=True, chunksize=100000)

  flare_cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR' , 'CYCLE_MONTH',
                    'OPERATOR_NO','OPERATOR_NAME',
                    'LEASE_CSGD_DISPCDE04_VOL', 'LEASE_GAS_DISPCD04_VOL']

  prod_cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR' , 'CYCLE_MONTH',
                    'OPERATOR_NO','OPERATOR_NAME', 'LEASE_OIL_PROD_VOL',
                    'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL',
                    'LEASE_CSGD_PROD_VOL']

  df = merge_dfs(flare_df, prod_df, flare_cols, prod_cols)
  df.rename(columns={'LEASE_CSGD_DISPCDE04_VOL': 'CASINGHEAD_GAS_FLARED', 'LEASE_GAS_DISPCD04_VOL':'GAS_FLARED'}, inplace=True)
  df['TOTAL_LEASE_FLARE_VOL'] = df['CASINGHEAD_GAS_FLARED'] + df['GAS_FLARED']

# %%
df.head()
# %%
