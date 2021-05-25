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
# import oil_price

# import pyspark
# from pyspark.sql import SparkSession
# from pyspark.sql import Row
# import pyspark.sql.functions as f
# from pyspark.sql.functions import udf, array
# %%
def s3_to_df(bucket_name, filename_1, filename_2, filename_3, cols_1, cols_2, cols_3,
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
  if chunk_data: # if memory + processing power is a concern
    df_list_1, df_list_2, df_list_3  = [], [], []

    for df_chunk_1 in tqdm(pd.read_csv('s3://'+bucket_name+'/'+filename_1,
                            delimiter='}', chunksize=chunksize, usecols=cols_1, )):
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

    for df_chunk_3 in tqdm(pd.read_csv('s3://'+bucket_name+'/'+filename_3,
                            delimiter='}', chunksize=chunksize, usecols=cols_3)):
      df_chunk_3 = df_chunk_3[df_chunk_3['CYCLE_YEAR'] >= year]
      df_list_3.append(df_chunk_3)
    df_3 = pd.concat(df_list_3)
    del df_list_3


    df = merge_dfs(df_1, df_2, df_3)
    del df_1, df_2, df_3

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

def merge_dfs(df1, df2, df3):
  '''Merge production + flare data'''
  df_new = df1.merge(df2, how='left', on=['DISTRICT_NO', 'LEASE_NO',
                                       'CYCLE_YEAR', 'CYCLE_MONTH',
                                       'OPERATOR_NO', 'OPERATOR_NAME'])
  print(df_new.columns)
  df = df_new.merge(df3, how='left', on=['DISTRICT_NO', 'LEASE_NO',
                                       'CYCLE_YEAR', 'CYCLE_MONTH',
                                       'OPERATOR_NO', 'OPERATOR_NAME'])
  print(df.columns)
  del df_new
  df = feature_engineer(df)
  df = df.reindex(columns=['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR', 'CYCLE_MONTH',
        'COUNTY_NO', 'COUNTY_NAME', 'OPERATOR_NO',
        'OPERATOR_NAME', 'LEASE_OIL_PROD_VOL', 'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL',
        'LEASE_CSGD_PROD_VOL', 'GAS_FLARED', 'CASINGHEAD_GAS_FLARED',
        'TOTAL_LEASE_FLARE_VOL', 'FIRST_REPORT', 'LAST_REPORT',
        'REPORT_DATE', 'MONTHS_FROM_FIRST_REPORT',
        'OIL_ENERGY (GWH)', 'GAS_ENERGY (GWH)', 'CSGD_ENERGY (GWH)',
        'COND_ENERGY (GWH)', 'FLARE_ENERGY (GWH)', 'TOTAL_ENERGY_PROD (GWH)'])
  return df

def feature_engineer(df):
  '''
  Create features: months from first production / flare report,
  convert produced volumes to energy

  oil_kwh = 1700 / bbl
  gas_kwh = 293 / mcf
  cond_kwh = 1589 / bbl (0.935 bbl oil = 1 bbl condensate)

  Parameters:
  df: pandas df from specified year to last report

  Returns:
  Chunk after year input and additional features
  '''
  oil_kwh = 1700
  gas_kwh = 293
  cond_kwh = 1589
  ## Rename columns for easier ID
  print(df.columns)
  df = df.rename(columns={'LEASE_CSGD_DISPCDE04_VOL': 'CASINGHEAD_GAS_FLARED', 'LEASE_GAS_DISPCD04_VOL':'GAS_FLARED'})
  print(df.columns)
  df['TOTAL_LEASE_FLARE_VOL'] = df['CASINGHEAD_GAS_FLARED'] + df['GAS_FLARED']
  ## Determine first report date for each lease
  df_min = df.groupby(['LEASE_NO'])['CYCLE_YEAR', 'CYCLE_MONTH'].min().reset_index()
  df_min['FIRST_REPORT'] = df_min['CYCLE_MONTH'].map(str) + '-' + df_min['CYCLE_YEAR'].map(str)
  df_min['FIRST_REPORT'] = pd.to_datetime(df_min['FIRST_REPORT'], yearfirst=False, format='%m-%Y').dt.to_period('M')
  ## Determine last report date for each lease
  df_max = df.groupby(['LEASE_NO'])['CYCLE_YEAR', 'CYCLE_MONTH'].max().reset_index()
  df_max['LAST_REPORT'] = df_max['CYCLE_MONTH'].map(str) + '-' + df_max['CYCLE_YEAR'].map(str)
  df_max['LAST_REPORT'] = pd.to_datetime(df_max['LAST_REPORT'], yearfirst=False, format='%m-%Y').dt.to_period('M')
  df = pd.merge_ordered(df, df_min[['LEASE_NO', 'FIRST_REPORT']], on=['LEASE_NO'], how='left')
  del df_min
  df = pd.merge_ordered(df, df_max[['LEASE_NO', 'LAST_REPORT']], on=['LEASE_NO'], how='left')
  del df_max
  ## Establish report date as datetime for each lease
  df['REPORT_DATE'] = df['CYCLE_MONTH'].map(str) + '-' + df['CYCLE_YEAR'].map(str)
  df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'], yearfirst=False, format='%m-%Y').dt.to_period('M')
  ## Create months fron first report feature, capturing the decay component of production
  df['MONTHS_FROM_FIRST_REPORT'] = df['REPORT_DATE'].astype('int') - df['FIRST_REPORT'].astype('int')
  ## Convert volume data to standardized energy units
  df['OIL_ENERGY (GWH)'] = (df['LEASE_OIL_PROD_VOL'] * oil_kwh) / 1000000
  df['GAS_ENERGY (GWH)'] = (df['LEASE_GAS_PROD_VOL'] * gas_kwh) / 1000000
  df['CSGD_ENERGY (GWH)'] = (df['LEASE_CSGD_PROD_VOL'] * gas_kwh) / 1000000
  df['COND_ENERGY (GWH)'] = (df['LEASE_COND_PROD_VOL'] * cond_kwh) / 1000000
  df['FLARE_ENERGY (GWH)'] = (df['TOTAL_LEASE_FLARE_VOL'] * gas_kwh) / 1000000
  df['TOTAL_ENERGY_PROD (GWH)'] = (df['COND_ENERGY (GWH)'] +
                                      df['CSGD_ENERGY (GWH)'] +
                                      df['GAS_ENERGY (GWH)'] +
                                      df['OIL_ENERGY (GWH)'])

  oil_price = pd.read_csv('s3://cbh-capstone1-texasrrc/price_of_oil.csv', index_col=0)
  df = pd.merge_ordered(df, oil_price, on=['CYCLE_MONTH', 'CYCLE_YEAR'], how='left')
  return df

def main():
  flare_cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR' , 'CYCLE_MONTH',
                    'OPERATOR_NO','OPERATOR_NAME',
                    'LEASE_CSGD_DISPCDE04_VOL', 'LEASE_GAS_DISPCD04_VOL']

  prod_cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR' , 'CYCLE_MONTH',
                    'OPERATOR_NO','OPERATOR_NAME', 'LEASE_OIL_PROD_VOL',
                    'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL',
                    'LEASE_CSGD_PROD_VOL']

  county_cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR' ,'CYCLE_MONTH',
                    'OPERATOR_NO','OPERATOR_NAME', 'COUNTY_NO',
                    'COUNTY_NAME']
  df = s3_to_df('cbh-capstone1-texasrrc',
                'OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv',
                'OG_LEASE_CYCLE_DATA_TABLE.dsv',
                'OG_COUNTY_LEASE_CYCLE_DATA_TABLE.dsv',
                flare_cols, prod_cols, county_cols,
                chunk_data=True, chunksize=15000000, year=2010)
  # pd.to_pickle(df, 's3://cbh-capstone1-texasrrc/clean_df.pkl')

# %%
if __name__ == '__main__':
  df = main()
# %%
