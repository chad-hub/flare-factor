# %%
#initialize workspace
import pandas as pd
import io
import boto3
import os
import s3fs
import numpy as np
from tqdm import tqdm
import dask.dataframe as dd

# import pyspark
# from pyspark.sql import SparkSession
# from pyspark.sql import Row
# import pyspark.sql.functions as f
# from pyspark.sql.functions import udf, array

# %%
def s3_to_df(bucket_name, filename_1, filename_2, cols_1, cols_2,
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
    df_list_1, df_list_2  = [], []

    for df_chunk_1 in tqdm(pd.read_csv('s3://'+bucket_name+'/'+filename_1,
                            delimiter='}', chunksize=chunksize, usecols=cols_1)):
      df_chunk_1 = df_chunk_1[df_chunk_1['CYCLE_YEAR'] >= year]
      df_list_1.append(df_chunk_1)
    df_1 = pd.concat(df_list_1)
    del df_list_1

    for df_chunk_2 in tqdm(pd.read_csv('s3://'+bucket_name+'/'+filename_2,
                              delimiter='}', chunksize=chunksize, usecols=cols_2)):
      df_chunk_2 = df_chunk_2[df_chunk_2['CYCLE_YEAR'] >= year]
      df_list_2.append(df_chunk_2)
    df_2 = pd.concat(df_list_2)
    del df_list_2
    df = merge_dfs(df_1, df_2)
    del df_1, df_2

  else:
    df_1 = pd.read_csv('s3://'+ bucket_name + '/' + filename_1,
    delimiter='}', chunksize=chunksize, usecols=cols_1)
    df_1 = df_1[df_1['CYCLE_YEAR'] >= year]

    df_2 = pd.read_csv('s3://'+ bucket_name + '/' + filename_2,
    delimiter='}', chunksize=chunksize, usecols=cols_2)
    df_2 = df_2[df_2['CYCLE_YEAR'] >= year]
    df = merge_dfs(df_1, df_2)
    del df_1, df_2
  return df

# %%
def merge_dfs(df1, df2):
  '''Merge production + flare data'''
  df = df1.merge(df2, how='left', on=['DISTRICT_NO', 'LEASE_NO',
                                       'CYCLE_YEAR', 'CYCLE_MONTH',
                                       'OPERATOR_NO', 'OPERATOR_NAME'])
  # df = df.reindex(columns=['DISTRICT_NO', 'LEASE_NO', 'CYCLE_MONTH', 'CYCLE_YEAR'
  #                         'OPERATOR_NO', 'OPERATOR_NAME',
  #                         'LEASE_OIL_PROD_VOL', 'LEASE_GAS_PROD_VOL',
  #                         'LEASE_COND_PROD_VOL', 'LEASE_CSGD_PROD_VOL',
  #                          'GAS_FLARED', 'CASINGHEAD_GAS_FLARED',
  #                          'TOTAL_LEASE_FLARE_VOL' ])
  df = feature_engineer(df)
  return df


# %%
def feature_engineer(chunk):
  '''
  Create features: months from first production / flare report,
  convert produced volumes to energy

  oil_kwh = 1700 / bbl
  gas_kwh = 293 / mcf
  cond_kwh = 1589 / bbl (0.935 bbl oil = 1 bbl condensate)

  Parameters:
  Chunk: pandas df chunk
  year: earliest year to pull reports

  Returns:
  Chunk after year input and additional features
  '''
  oil_kwh = 1700
  gas_kwh = 293
  cond_kwh = 1589
  ## Rename columns for easier ID
  chunk.rename(columns={'LEASE_CSGD_DISPCDE04_VOL': 'CASINGHEAD_GAS_FLARED', 'LEASE_GAS_DISPCD04_VOL':'GAS_FLARED'}, inplace=True)
  chunk['TOTAL_LEASE_FLARE_VOL'] = chunk['CASINGHEAD_GAS_FLARED'] + chunk['GAS_FLARED']
  #
  chunk_min = chunk.groupby(['LEASE_NO'])['CYCLE_YEAR', 'CYCLE_MONTH'].min().reset_index()
  chunk_min['FIRST_REPORT'] = chunk_min['CYCLE_MONTH'].map(str) + '-' + chunk_min['CYCLE_YEAR'].map(str)
  chunk_min['FIRST_REPORT'] = pd.to_datetime(chunk_min['FIRST_REPORT'], yearfirst=False, format='%m-%Y').dt.to_period('M')
  chunk_max = chunk.groupby(['LEASE_NO'])['CYCLE_YEAR', 'CYCLE_MONTH'].max().reset_index()
  chunk_max['LAST_REPORT'] = chunk_max['CYCLE_MONTH'].map(str) + '-' + chunk_max['CYCLE_YEAR'].map(str)
  chunk_max['LAST_REPORT'] = pd.to_datetime(chunk_max['LAST_REPORT'], yearfirst=False, format='%m-%Y').dt.to_period('M')
  chunk = pd.merge_ordered(chunk, chunk_min[['LEASE_NO', 'FIRST_REPORT']], on=['LEASE_NO'], how='left')
  del chunk_min
  chunk = pd.merge_ordered(chunk, chunk_max[['LEASE_NO', 'LAST_REPORT']], on=['LEASE_NO'], how='left')
  del chunk_max
  chunk['REPORT_DATE'] = chunk['CYCLE_MONTH'].map(str) + '-' + chunk['CYCLE_YEAR'].map(str)
  chunk['REPORT_DATE'] = pd.to_datetime(chunk['REPORT_DATE'], yearfirst=False, format='%m-%Y').dt.to_period('M')
  chunk['MONTHS_FROM_FIRST_REPORT'] = chunk['REPORT_DATE'].astype('int') - chunk['FIRST_REPORT'].astype('int')

  chunk['OIL_ENERGY (GWH)'] = (chunk['LEASE_OIL_PROD_VOL'] * oil_kwh) / 1000000
  chunk['GAS_ENERGY (GWH)'] = (chunk['LEASE_GAS_PROD_VOL'] * gas_kwh) / 1000000
  chunk['CSGD_ENERGY (GWH)'] = (chunk['LEASE_CSGD_PROD_VOL'] * gas_kwh) / 1000000
  chunk['COND_ENERGY (GWH)'] = (chunk['LEASE_COND_PROD_VOL'] * cond_kwh) / 1000000
  chunk['FLARE_ENERGY (GWH)'] = (chunk['TOTAL_LEASE_FLARE_VOL'] * gas_kwh) / 1000000
  chunk['TOTAL_ENERGY_PROD (GWH)'] = (chunk['LEASE_COND_PROD_ENERGY (GWH)'] +
                                      chunk['LEASE_CSGD_PROD_ENERGY (GWH)'] +
                                      chunk['LEASE_GAS_PROD_ENERGY (GWH)'] +
                                      chunk['LEASE_OIL_PROD_ENERGY (GWH)'])

  return chunk


# %%

# %%
# merged_df = merged_df.

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

  # spark = SparkSession.builder.getOrCreate()

  flare_cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR' , 'CYCLE_MONTH',
                    'OPERATOR_NO','OPERATOR_NAME',
                    'LEASE_CSGD_DISPCDE04_VOL', 'LEASE_GAS_DISPCD04_VOL']

  prod_cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR' , 'CYCLE_MONTH',
                    'OPERATOR_NO','OPERATOR_NAME', 'LEASE_OIL_PROD_VOL',
                    'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL',
                    'LEASE_CSGD_PROD_VOL']

  # flare_df = s3_to_df('cbh-capstone1-texasrrc',
  #                   'OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv', flare_cols,
  #                   chunk_data=True, chunksize=15000000, year=2020)

  # prod_df = s3_to_df('cbh-capstone1-texasrrc',
  #                   'OG_LEASE_CYCLE_DATA_TABLE.dsv', prod_cols,
  #                   chunk_data=True, chunksize=15000000, year=2020)

  # df = merge_dfs(flare_df, prod_df)

  # del flare_df
  # del prod_df

  # df = feature_engineer(df)

  df = s3_to_df('cbh-capstone1-texasrrc',
                'OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv',
                'OG_LEASE_CYCLE_DATA_TABLE.dsv',
                flare_cols, prod_cols,
                chunk_data=True, chunksize=15000000, year=2020)



# %%

# %%


# # %%
# f_path = 's3://cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv'
# p_path = 's3://cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DATA_TABLE.dsv'

# %%

# %%
lease_df_min = df.groupby(['LEASE_NO'])['CYCLE_YEAR', 'CYCLE_MONTH'].min().reset_index()
lease_df_min['FIRST_REPORT'] = lease_df_min['CYCLE_MONTH'].map(str) + '-' + lease_df_min['CYCLE_YEAR'].map(str)
lease_df_min['FIRST_REPORT'] = pd.to_datetime(lease_df_min['FIRST_REPORT'], yearfirst=False, format='%m-%Y').dt.to_period('M')
lease_df_min.head()
# %%
lease_df_max = df.groupby(['LEASE_NO'])['CYCLE_YEAR', 'CYCLE_MONTH'].max().reset_index()
lease_df_max['LAST_REPORT'] = lease_df_max['CYCLE_MONTH'].map(str) + '-' + lease_df_max['CYCLE_YEAR'].map(str)
lease_df_max['LAST_REPORT'] = pd.to_datetime(lease_df_max['LAST_REPORT'], yearfirst=False, format='%m-%Y').dt.to_period('M')
lease_df_max.head()

# %%
df = pd.merge_ordered(df, lease_df_min[['LEASE_NO', 'FIRST_REPORT']], on=['LEASE_NO'], how='left')
# %%
df = pd.merge_ordered(df, lease_df_max[['LEASE_NO', 'LAST_REPORT']], on=['LEASE_NO'], how='left')

# %%
df['REPORT_DATE'] = df['CYCLE_MONTH'].map(str) + '-' + df['CYCLE_YEAR'].map(str)
df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'], yearfirst=False, format='%m-%Y').dt.to_period('M')
# %%
# %%
df['MONTHS_FROM_FIRST_REPORT'] = df['REPORT_DATE'].astype('int') - df['FIRST_REPORT'].astype('int')
df.info()
# %%
df.columns
# %%
df.drop(['LAST_REPORT_x','LAST_REPORT_y','FIRST_PROD_REPORT'], axis=1, inplace=True)
df.head()
# %%

# %%
