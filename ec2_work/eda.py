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
plt.style.use('ggplot'), plt.
# %%
df_dist_year = pd.read_csv('group_by_district_yr.csv')
# %%
## Load in larger data set
s3 = boto3.client('s3')
df = pd.read_csv('s3://cbh-capstone1-texasrrc/merged_flare_og.csv')
# %%
df.drop('Unnamed: 0', axis=1, inplace=True)
# %%
df.to_csv('s3://cbh-capstone1-texasrrc/merged_flare_og.csv')
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
districts = range(1,15)
fig, ax = plt.subplots(7,2)
for d, subplot in zip(districts, ax.flatten()):
  sns.pairplot(df_dist_year[df_dist_year['DISTRICT_NO'] == d].loc[:,['LEASE_OIL_PROD_VOL',
                            'LEASE_GAS_PROD_VOL',
                            'LEASE_COND_PROD_VOL',
                            'LEASE_CSGD_PROD_VOL',
                            'TOTAL_LEASE_FLARE_VOL']],
                        palette='coolwarm')


# %%
df_2008_plus = df_dist_year[df_dist_year['YEAR'] >= 2008]
df_2008_plus = df_2008_plus[df_dist_year['YEAR'] < 2020]
df_2007_less = df_dist_year[df_dist_year['YEAR'] < 2008]
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
