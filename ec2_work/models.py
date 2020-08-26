# %%
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, precision_score, recall_score
# %%
import boto3
import os
import s3fs
import numpy as np
import pandas as pd
# %%
s3 = boto3.client('s3')
df = pd.read_csv('s3://cbh-capstone1-texasrrc/merged_flare_og.csv')
# %%
df.drop('Unnamed: 0', axis=1, inplace=True)
# %%
df.head()

# %%
dums = pd.get_dummies(df['DISTRICT_NO'], prefix='DIST')
dums.head()
# %%
df_dum = df.drop('DISTRICT_NO', axis=1)

# %%
df_dum = pd.concat((df_dum, dums), axis=1)
# %%
df_dum.columns

# %%
X.head()
# %%
X = df_dum.drop(['Unnamed: 0','LEASE_NO', 'CYCLE_YEAR_MONTH',
                  'OPERATOR_NO_x', 'OPERATOR_NO_y',
                  'OPERATOR_NAME_x', 'OPERATOR_NAME_y',
                  'GAS_FLARED', 'CASINGHEAD_GAS_FLARED'], axis=1)
y = df_dum.pop('TOTAL_LEASE_FLARE_VOL')
# %%
X.drop('TOTAL_LEASE_FLARE_VOL', axis=1, inplace=True)
# %%
X_train, X_test, y_train, y_test = tts(np.array(X), np.array(y), test_size=0.33, random_state=42)
# %%


# %%
rfr = RandomForestRegressor(n_estimators=20, max_features=10,
                            verbose=2, oob_score=True,
                            n_jobs=-1)
rfr.fit(X_train, y_train)
# %%
y_pred_test = rfr.predict(X_test)
y_pred_train = rfr.predict(X_train)
# %%
test_accuracy = accuracy_score(y_test, y_pred_test)
test_accuracy
# %%
from sklearn.dummy import DummyRegressor
# %%
dummy = DummyRegressor()
dummy.fit(X_train, y_train)
# %%
dummy.predict(X_test)
# %%
dummy.score(X_test, dummy.predict(X_test))
# %%
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
# %%
y_pred_reg = reg.predict(X_test)
reg.score(X_test, y_test)
# %%
