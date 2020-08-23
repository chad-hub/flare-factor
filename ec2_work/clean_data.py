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
  ave_bytes += chunk.memory_usage

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
# new_df.rename(columns={'LEASE_CSGD_DISPCDE04_VOL': 'CASINGHEAD_GAS_FLARED', 'LEASE_GAS_DISPCD04_VOL':'GAS_FLARED'})
    # new_df['TOTAL_LEASE_FLARE_VOL'] = df['CASINGHEAD_GAS_FLARED'] + df['GAS_FLARED']
# %%

flare_df = pd.DataFrame(flare_by_lease_arr)
# %%
flare_df.head()