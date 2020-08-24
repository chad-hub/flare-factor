# %%
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import io
import boto3
import os
import s3fs
import numpy as np

# %%
s3 = boto3.client('s3')
s3.list_buckets()
# %%
response = s3.list_objects_v2(Bucket='cbh-capstone1-texasrrc')
for obj in response['Contents']:
  print(obj['Key'])
# %%
textfile = pd.read_csv('s3://cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv',
                       delimiter='}', chunksize=1000000)

# %%
ave_bytes = 0
for idx, chunk in enumerate(textfile, start=1):
  ave_bytes += chunk.memory_usage()

print('total number of chunks: ', idx)
print(f'Average bytes per loop: {ave_bytes/idx}')
# %%

def chunks_to_dfs(chunk_file, cols):
  data = []
  for idx, chunk in enumerate(chunk_file, start=1):
    data_array = chunk[cols].to_numpy()
    data.append(data_array)
    print(idx)
  return data


# %%
cols = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR_MONTH', 'OPERATOR_NO',
       'OPERATOR_NAME', 'LEASE_CSGD_DISPCDE04_VOL', 'LEASE_GAS_DISPCD04_VOL']

file = pd.read_csv('s3://cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv',
                       delimiter='}', chunksize=1000000)
flare_by_lease_arr = chunks_to_dfs(file, cols)

# %%

# %%

flare_df = pd.read_csv('s3://cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DISP_DATA_TABLE.dsv',
                       delimiter='}', usecols=cols)
# %%
flare_df.info()
# %%
flare_df.head()
# %%
flare_df.rename(columns={'LEASE_CSGD_DISPCDE04_VOL': 'CASINGHEAD_GAS_FLARED', 'LEASE_GAS_DISPCD04_VOL':'GAS_FLARED'}, inplace=True)
flare_df['TOTAL_LEASE_FLARE_VOL'] = flare_df['CASINGHEAD_GAS_FLARED'] + flare_df['GAS_FLARED']

# %%
flare_df.head()
# %%
flare_df.to_csv('flare_info.csv')
# %%

# %%
resp = s3.list_objects_v2(Bucket='cbh-capstone1-texasrrc')
for obj in resp['Contents']:
  print(obj['Key'])



# %%
test_og = pd.read_csv('s3://cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DATA_TABLE.dsv',
               delimiter='}', chunksize=10000)

# %%
test_df = test_og.get_chunk(100000)
# %%
test_df.head()
# %%
test_df.columns
# %%
cols_og = ['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR_MONTH', 'OPERATOR_NO',
       'OPERATOR_NAME', 'LEASE_OIL_PROD_VOL', 'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL', 'LEASE_CSGD_PROD_VOL']
# %%
og_df = pd.read_csv('s3://cbh-capstone1-texasrrc/OG_LEASE_CYCLE_DATA_TABLE.dsv',
                       delimiter='}', usecols=cols_og)
# %%
og_df.info()
# %%
og_df['CYCLE_YEAR_MONTH'].min()
# %%
og_df.head()
# %%
flare_df = pd.read_csv('flare_info.csv')
# %%
flare_df['CYCLE_YEAR_MONTH'].min()
# %%
og_df.to_csv('og_info.csv')
# %%
flare_df.drop('Unnamed: 0', axis=1, inplace=True)
# %%
flare_df.head()
# %%
flare_df.shape
# %%
og_df.shape
# %%
flare_df['DISTRICT_NO'].value_counts()
# %%
import time
# %%
start_time = time.time()
merged_df = pd.merge(flare_df, og_df, on=['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR_MONTH'], how='left')
print(f'Merging took {time.time() - start_time} to execute')
# %%
merged_df.info()
# %%
merged_df.head()
# %%
merged_df.columns
# %%
merged_df = merged_df.reindex(columns=['DISTRICT_NO', 'LEASE_NO', 'CYCLE_YEAR_MONTH', 'OPERATOR_NO_x', 'OPERATOR_NO_y', 'OPERATOR_NAME_x', 'OPERATOR_NAME_y', 'LEASE_OIL_PROD_VOL',
       'LEASE_GAS_PROD_VOL', 'LEASE_COND_PROD_VOL', 'LEASE_CSGD_PROD_VOL', 'GAS_FLARED', 'CASINGHEAD_GAS_FLARED', 'TOTAL_LEASE_FLARE_VOL' ])
# %%
merged_df.head()
# %%
merged_df['TOTAL_LEASE_FLARE_VOL'].sum()
# %%
merged_df.to_csv('merged_flare_og.csv')
# %%
import codecs
import gzip
# %%
f = gzip.open('grf047.ebc.gz', 'r')
file_content = f.read()
# file_content = file_content.decode('COMP')
file_content
# %%
