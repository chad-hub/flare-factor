# %%
#initialize workspace
import pandas as pd
import io
import boto3
import os
import s3fs
import numpy as np
from tqdm import tqdm

# %%
def s3_to_df(bucket_name, filename, cols,
            chunk_data=True, chunksize=100000, year=2010):
  '''
  Establish connection to Amazon s3 bucket,
  retrieve chunks of data (unless working with EC2 that can handle the whole file),
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
    df_list = []
    # textfile = pd.read_csv('s3://'+bucket_name+'/'+filename,
    # delimiter='}', chunksize=chunksize, usecols=cols)
    for df_chunk in tqdm(pd.read_csv('s3://'+bucket_name+'/'+filename,
                            delimiter='}', chunksize=chunksize, usecols=cols)):
      df_chunk = df_chunk[df_chunk['CYCLE_YEAR'] >= year]
      df_list.append(df_chunk)
    df = pd.concat(df_list)
    del df_list
  else:
    df = pd.read_csv('s3://'+ bucket_name + '/' + filename,
    delimiter='}', chunksize=chunksize, usecols=cols)
    df = df[df['CYCLE_YEAR'] >= year]
  return df


# %%
def merge_dfs(df1, df2):
  '''Merge production + flare data'''
  # df1 = pd.concat(df_list1)
  # df2 = pd.concat(df_list2)
  df = df1.merge(df2, how='left', on=['DISTRICT_NO', 'LEASE_NO',
                                       'CYCLE_YEAR', 'CYCLE_MONTH',
                                       'OPERATOR_NO', 'OPERATOR_NAME'])
  del df1, df2
  return df

# %%
# merged_df = merged_df.reindex(columns=['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR_MONTH', 'OPERATOR_NO_x', 'OPERATOR_NO_y', 'OPERATOR_NAME_x', 'OPERATOR_NAME_y', 'LEASE_OIL_PROD_VOL',
#        'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL', 'LEASE_CSGD_PROD_VOL', 'GAS_FLARED', 'CASINGHEAD_GAS_FLARED', 'TOTAL_LEASE_FLARE_VOL' ])

# test = pd.to_datetime(df['CYCLE_YEAR_MONTH'], yearfirst=True, format='%Y%m') + MonthEnd()

# # %%
# df_districts = df.groupby(['DISTRICT_NO', 'YEAR'])['LEASE_OIL_PROD_VOL',
#                             'LEASE_GAS_PROD_VOL',
#                             'LEASE_COND_PROD_VOL',
#                             'LEASE_CSGD_PROD_VOL',
#                             'TOTAL_LEASE_FLARE_VOL'].sum().reset_index()

# %%

# %%
if __name__ == '__main__':

  flare_cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR' , 'CYCLE_MONTH',
                    'OPERATOR_NO','OPERATOR_NAME',
                    'LEASE_CSGD_DISPCDE04_VOL', 'LEASE_GAS_DISPCD04_VOL']

  prod_cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR' , 'CYCLE_MONTH',
                    'OPERATOR_NO','OPERATOR_NAME', 'LEASE_OIL_PROD_VOL',
                    'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL',
                    'LEASE_CSGD_PROD_VOL']

  flare_df = s3_to_df('cbh-capstone1-texasrrc',
                    'OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv', flare_cols,
                    chunk_data=True, chunksize=15000000, year=2020)

  prod_df = s3_to_df('cbh-capstone1-texasrrc',
                    'OG_LEASE_CYCLE_DATA_TABLE.dsv', prod_cols,
                    chunk_data=True, chunksize=15000000, year=2020)


  df = merge_dfs(flare_df, prod_df)
  df.rename(columns={'LEASE_CSGD_DISPCDE04_VOL': 'CASINGHEAD_GAS_FLARED', 'LEASE_GAS_DISPCD04_VOL':'GAS_FLARED'}, inplace=True)
  df['TOTAL_LEASE_FLARE_VOL'] = df['CASINGHEAD_GAS_FLARED'] + df['GAS_FLARED']

# %%
# import csv
# # fileobj = s3.get_object(Bucket='cbh-capstone1-texasrrc',
#                         # Key='OG_LEASE_CYCLE_DATA_TABLE.dsv')
# # fs = s3fs.S3FileSystem(anon=True,)
# s3 = boto3.client('s3')
# top = pd.read_csv('s3://cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DATA_TABLE.dsv', delimiter="}", nrows=1)
# headers = top.columns.values
# with open('s3://cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DATA_TABLE.dsv', "r") as f, open('cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DATA_TABLE_TEST.dsv', "w") as g:
#   last_num = f.readlines()[-200000].strip.split("}")
#   c = csv.writer(g)
#   c.writerow(headers)
#   c.writerow(last_num)

# bottom = pd.read_csv('cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DATA_TABLE_TEST.dsv', delimiter="}")
# bottom.reset_index(inplace=True, drop=True)
# bottom.head()

# %%
# import dask.dataframe as dd

# # %%
# f_path = 's3://cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv'
# p_path = 's3://cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DATA_TABLE.dsv'
