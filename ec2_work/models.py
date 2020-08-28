# %%
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import cross_validate
# %%
import boto3
import os
import s3fs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)
sns.set_style(style="whitegrid")
plt.style.use('ggplot')
# %%

df = pd.read_pickle('df_2000.pkl')

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
df_dum.head()
# %%
cols_to_drop = ['LEASE_NO', 'MONTH', 'YEAR', 'OPERATOR_NO_x', 'OPERATOR_NAME_x',
       'LEASE_OIL_PROD_ENERGY (GWH)', 'LEASE_GAS_PROD_ENERGY (GWH)',
       'LEASE_CSGD_PROD_ENERGY (GWH)', 'LEASE_COND_PROD_ENERGY (GWH)', 'GAS_FLARED', 'CASINGHEAD_GAS_FLARED',
       'TOTAL_LEASE_FLARE_VOL', 'FIRST_PROD_REPORT', 'LAST_REPORT',
       'REPORT_DATE', 'LEASE_FLARE_ENERGY (GWH)', 'TOTAL_ENERGY_PROD (GWH)', 'WASTE_RATIO']

# %%
samp_df = df_dum.sample(frac=0.25, random_state=42)
samp_df.columns
# %%
# X = samp_df.drop(cols_to_drop, axis=1)
y = samp_df['TOTAL_LEASE_FLARE_VOL']
X_1 = samp_df[['LEASE_CSGD_PROD_VOL', 'MONTHS_FROM_FIRST_REPORT', 'COMPANY_CAT_(2.0, 14900.0]',
 'DIST_1',
 'COMPANY_CAT_(-0.001, 2.0]',
 'Price of Oil',
 'LEASE_COND_PROD_VOL',
 'DIST_10',
 'LEASE_GAS_PROD_VOL',
 'LEASE_OIL_PROD_VOL' ]]
# %%
X_train, X_test, y_train, y_test = tts(np.array(X_1), np.array(y), test_size=0.33, random_state=42)
# %%


# %%
rfr = RandomForestRegressor(n_estimators=1000, max_features=10,
                            verbose=1, oob_score=True,
                            n_jobs=-1, max_samples=0.25,
                            max_depth=3)
rfr.fit(X_train, y_train)
# %%
y_pred_test_rfr = rfr.predict(X_test)
y_pred_train_rfr = rfr.predict(X_train)
# %%
test_mse = mean_squared_error(y_test, y_pred_test_rfr)
train_mse = mean_squared_error(y_train, y_pred_train_rfr)
print(f'Test RMSE: {test_mse**0.5}')
print(f'Train RMSE: {train_mse**0.5}')

# %%
r2_test_rfr = rfr.score(X_test,y_test)
r2_train_rfr = rfr.score(X_train,y_train)
print(f'Test r2: {r2_test_rfr}')
print(f'Train r2: {r2_train_rfr}')
# %%
plot_residuals(X_test, y_pred_test_rfr, y_test)
# %%
from sklearn.ensemble import GradientBoostingRegressor
# %%
gbr = GradientBoostingRegressor(n_estimators=100,
                                criterion='mse',
                                verbose=1,
                                learning_rate=0.01)
gbr.fit(X_train, y_train)
# %%
def stage_score_plot(model, X_train, y_train, X_test, y_test):
    train_scores = []
    test_scores = []
    x_list= []
    for i,predict in enumerate(model.staged_predict(X_train)):
        train_scores.append(mean_squared_error(y_train, predict))
        x_list.append(i)
    for i,predict in enumerate(model.staged_predict(X_test)):
        test_scores.append(mean_squared_error(y_test, predict))
    #return x_list, test_scores, train_scores

    plt.plot(x_list, train_scores, label = f'{model.__class__.__name__} Train - learning rate {model.learning_rate}', ls = '-')
    plt.plot(x_list, test_scores, label= f'{model.__class__.__name__} Test - learning rate {model.learning_rate}')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title(f'{model.__class__.__name__}')
# %%
stage_score_plot(gbr, X_train, y_train, X_test, y_test)
plt.legend()
plt.show()


# %%
y_pred_test_gbr = gbr.predict(X_test)
y_pred_train_gbr = gbr.predict(X_train)

# %%
test_mse_gbr = mean_squared_error(y_test, y_pred_test_gbr)
train_mse_gbr = mean_squared_error(y_train, y_pred_train_gbr)
print(f'Test RMSE GBR: {test_mse_gbr**0.5}')
print(f'Train RMSE GBR: {train_mse_gbr**0.5}')

# %%
r2_test_gbr = gbr.score(X_test,y_test)
r2_train_gbr = gbr.score(X_train,y_train)
print(f'Test r2: {r2_test_gbr}')
print(f'Train r2: {r2_train_gbr}')

# %%
gbr.feature_importances_



# %%
def plot_residuals(Xvals, y_pred, y_true):
  sns.residplot(Xvals, y_true)
  sns.residplot(Xvals, y_pred)
  plt.show()


# %%
random_forest_grid = {'max_depth': [1,3,5],
                      'max_features': ['sqrt', 'log2', None],
                      'min_samples_split': [2, 4],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False],
                      'n_estimators': [10, 20, 40, 80],
                      'random_state': [1]}

rf_gridsearch = GridSearchCV(RandomForestRegressor(),
                             random_forest_grid,
                             n_jobs=-1,
                             verbose=True,
                             scoring='neg_mean_squared_error')
rf_gridsearch.fit(X_train, y_train)

print("best parameters:", rf_gridsearch.best_params_)

best_rf_model = rf_gridsearch.best_estimator_


# %%
gb_grid =  {'learning_rate': [0.1, 0.4, 0.7, 1.0],
                      'max_depth': [1,3, None],
                      'min_samples_leaf': [1, 2, 4],
                      'max_features': ['sqrt', 'log2', None],
                      'n_estimators': [10, 20, 40, 80],
                      'subsample': [1, 0.5, 0.1]}

gb_gridsearch = GridSearchCV(GradientBoostingRegressor(),
                                gb_grid,
                                n_jobs=-1,
                                verbose=True,
                                scoring='r2')

gb_gridsearch.fit(X_train, y_train)

print("best parameters:", gb_gridsearch.best_params_)

best_gb_model = gb_gridsearch.best_estimator_


# %%
from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
# %%
# model = LinearRegression()
visualizer = ResidualsPlot(gbr)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
# %%
import pickle
gbr2 = pickle.load(open('GradBoost.sav', 'rb'))
# %%
gbr2.fit(X_train, y_train)

# %%
stage_score_plot(gbr2, X_train, y_train, X_test, y_test)
plt.legend()
plt.show()
# %%
y_pred_train_gbr2 = gbr2.predict(X_train)
y_pred_test_gbr2 = gbr2.predict(X_test)

test_mse_gbr2 = mean_squared_error(y_test, y_pred_test_gbr2)
train_mse_gbr2 = mean_squared_error(y_train, y_pred_train_gbr2)
print(f'Test RMSE GBR: {test_mse_gbr2**0.5}')
print(f'Train RMSE GBR: {train_mse_gbr2**0.5}')

# %%
r2_test_gbr2 = gbr2.score(X_test,y_test)
r2_train_gbr2 = gbr2.score(X_train,y_train)
print(f'Test r2: {r2_test_gbr2}')
print(f'Train r2: {r2_train_gbr2}')
# %%
gbr2.feature_importances_
# %%
fig, ax = plt.subplots()

feature_importances = 100*gbr2.feature_importances_ 
names = X.columns
feature_importances, feature_names, feature_idxs = \
    zip(*sorted(zip(feature_importances, names, range(len(names)))))

width = 0.8

idx = np.arange(len(names))
ax.barh(idx, feature_importances, align='center')
ax.set_yticks(idx, feature_names)
ax.set_yticklabels(feature_names)
ax.set_title("Feature Importances in Gradient Booster")
ax.set_xlabel('Relative Importance of Feature', fontsize=14)
ax.set_ylabel('Feature Name', fontsize=14)

# %%
X.columns[0]
# %%
print(feature_importances)
print(feature_names)
# %%
X_2 = X[['LEASE_COND_PROD_VOL', 'MONTHS_FROM_FIRST_REPORT']]
XT, Xt, yT, yt = tts(np.array(X_2), np.array(y),test_size=0.33, random_state=42 )

# %%
gbr3 = GradientBoostingRegressor()
gbr3.fit(np.array(XT),np.array(yT))
# %%

# %%
stage_score_plot(gbr3, XT, yT, Xt, yt)
plt.legend()
plt.show()

# %%
y_pred_train_gbr3 = gbr3.predict(XT)
y_pred_test_gbr3 = gbr3.predict(Xt)

test_mse_gbr3 = mean_squared_error(yt, y_pred_test_gbr3)
train_mse_gbr3 = mean_squared_error(yT, y_pred_train_gbr3)
print(f'Test RMSE GBR: {test_mse_gbr3**0.5}')
print(f'Train RMSE GBR: {train_mse_gbr3**0.5}')

# %%
r2_test_gbr3 = gbr3.score(Xt,yt)
r2_train_gbr3 = gbr3.score(XT,yT)
print(f'Test r2: {r2_test_gbr3}')
print(f'Train r2: {r2_train_gbr3}')
# %%
