
# -*- coding: utf-8 -*-

"""
Created on Mon Jun 19 14:01:26 2023

@author: kosta
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.compose import StackingForecaster, EnsembleForecaster, AutoEnsembleForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.compose import make_reduction
from sklearn.neural_network import MLPRegressor
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter,temporal_train_test_split, ForecastingRandomizedSearchCV
from xgboost import XGBRegressor, XGBRFRegressor
from sktime.performance_metrics.forecasting import MeanSquaredError, MeanAbsoluteError, MeanAbsoluteScaledError, MeanSquaredScaledError
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from lightgbm import LGBMRegressor
from sktime.pipeline import make_pipeline
from scipy.stats import uniform
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from sktime.transformations.series.adapt import PandasTransformAdaptor
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.summarize import WindowSummarizer
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMAResults 
plt.rcParams['figure.dpi'] = 300


df = pd.read_excel('C:\\Users\\kosta\\Documents\\PhD related\\Python\\Feature forecasting\\All metals.xlsx',
                  parse_dates = ['DATE'],index_col='DATE')


df = np.log(df/df.shift(1))
#df = df.pct_change().dropna()
#df = df.diff().dropna()

df.columns = [j.title() for j in df.columns]

name = df.columns[3]
metal = df[name].copy()

metal = metal.loc['1990-01-01':]
metal.plot()
plt.grid()
plt.title(metal.name)
plt.show()


# corr = []
# dates = []
# for i in range(10,len(metal)):
#     series = metal.iloc[:i]
#     dates.append(series.index[-1])
#     sht = series.shift().dropna()
#     corr.append(np.corrcoef(series.iloc[1:],sht)[1,0])

# pd.DataFrame(corr,index=dates).plot()
# plt.show()
# #metal = metal.iloc[:-40]

plot_pacf(metal)
plt.show()


metal.index = pd.PeriodIndex(metal.index,freq='M')

test_size = 80
y_train,y_test = temporal_train_test_split(metal,test_size=test_size)

n_iter = 600
strategy = 'recursive'

HO_size = 100
splitter_HO = ExpandingWindowSplitter(fh=[1,2,3,4,5,6],initial_window=len(y_train)-HO_size ,step_length=1)

def hyperparam_plot(results,model):
    ax = y_train.iloc[-HO_size:].plot(color='black',marker='o')
    for i in range(len(results)):
        results['y_pred'].iloc[i].plot(ax=ax,color='red',alpha=.7)
        #ax.get_legend().remove()
        #plt.title('Naive CV 70 months')
    plt.title(f"{model} error {results['test_MeanSquaredError'].mean()}")
    plt.ylabel('Log Returns')
    plt.legend(['Real','Predicted'])
    plt.grid()
    plt.show()


 
def test_plot(results,model):
    ax = metal.iloc[-test_size:].plot(color='black',marker='o')
    for i in range(len(results)):
        plt.ylabel('Log Returns')
        results['y_pred'].iloc[i].plot(ax=ax,color='red',alpha=.7)
        #ax.get_legend().remove()
        #plt.title('Naive CV 70 months')
    plt.title(f"{name} Ensemble {np.round(results['test_MeanSquaredError'].mean(),6)},ARIMA {np.round(arima_error,6)},Naive {np.round(naive_error,6)},LGBM {np.round(error_lgbm,6)}")
    plt.legend(['Real','Predicted'])
    plt.grid()
    plt.show()
    
def test_plot2(results,model):
    ax = metal.iloc[-test_size:].plot(color='black',marker='o')
    for i in range(len(results)):
        plt.ylabel('Log Returns')
        results['y_pred'].iloc[i].plot(ax=ax,color='red',alpha=.7)
        #ax.get_legend().remove()
        #plt.title('Naive CV 70 months')
    plt.title(f"{name} {model}")
    plt.legend(['Real','Predicted'])
    plt.grid()
    
    plt.savefig('C:\\Users\kosta\\Documents\\PhD related\\Short-term forecasting Ensemble\\'+name[:4]+model+'png')
    plt.show()
############################################# LGBM ####################################################


params_lgbm = {
    'estimator__n_estimators': [20,50, 100, 150, 200, 250, 300],
    'estimator__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
    'estimator__max_depth': [3, 5, 7, 9, 12, 15, 20, 25, 30, 35, 40, 45, 50],
    'estimator__num_leaves': [2, 4, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50,80],
    'estimator__min_child_samples': [2,5, 10, 15, 20, 25, 30, 35],
    # 'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    # 'estimator__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'estimator__reg_alpha': [0, 0.001, 0.01, 0.1, 1.0, 10.0],
    'estimator__reg_lambda': [0, 0.001, 0.01, 0.1, 1.0, 10.0],
    'estimator__importance_type': ['split', 'gain'],
    'window_length': np.arange(1, 7)
}


# params_lgbm = {
#     'estimator__n_estimators': [100, 200, 300],
#     'estimator__learning_rate': [0.01, 0.1, 0.5, 1.0],
#     'estimator__max_depth': [3, 5, 7, 9],
#     'estimator__num_leaves': [10, 20, 30, 40, 50],
#     'estimator__reg_alpha': [0.0, 0.1, 0.5],
#     'estimator__reg_lambda': [0.0, 0.1, 0.5],
#     'window_length': [ 1,2,3,4]
# }


lgbm = make_reduction(LGBMRegressor(random_state=0),strategy=strategy,window_length=None)

#forecaster = WindowSummarizer(**kwargs,n_jobs=-1) * lgbm


grid_lgbm= ForecastingRandomizedSearchCV(forecaster = lgbm,cv = splitter_HO,
                                          param_distributions=params_lgbm,scoring = MeanSquaredError(),n_jobs = -1,n_iter=n_iter,
                                          random_state=0,error_score='raise',return_n_best_forecasters=10)
grid_lgbm.fit(y_train,fh=[1,2,3,4,5,6])

results_lgbm_HO = evaluate(y=y_train,forecaster=grid_lgbm.best_forecaster_,cv=splitter_HO,
                            return_data=True,scoring=MeanSquaredError(square_root=True))

print(grid_lgbm.best_params_)
print(grid_lgbm.best_score_)
hyperparam_plot(results_lgbm_HO, 'LGBM')

cv = ExpandingWindowSplitter(fh = np.arange(1,7),initial_window=len(metal)-test_size,step_length=1)
results_lgbm = evaluate(y=metal,forecaster=grid_lgbm.best_forecaster_,cv=cv,return_data=True,
                        scoring=[MeanSquaredError(square_root=True),MeanAbsoluteError(),
                                 MeanAbsoluteScaledError(),MeanSquaredScaledError(square_root=True)])

error_lgbm = results_lgbm['test_MeanSquaredError'].mean()



############################# XGBoost ########################################################



# params_xgb = {
#     'estimator__n_estimators': [30,50,80,100,150,300,400],
#     'estimator__eta': [0.1,0.3,0.8,0.9,1],
#       'estimator__max_depth': np.arange(1,10,2),
#     'estimator__learning_rate': [0.001, 0.01, 0.1, 0.3,0.8,1],
#     # 'estimator__subsample': [0.3,0.5,0.8,1],
#     # 'estimator__colsample_bytree': [0.8, 0.9,0.95,0.99,1.0],
#       'estimator__reg_lambda': [0.0001,0.001,0.01],
#       'estimator__reg_alpha': [0.0,0.001,0.01,0.1,],
#     # 'estimator__min_child_weight': [0.01, 0.1, 1,5,3],
#       'estimator__gamma': [0.0,0.001,0.005,0.01, 0.1,],
#     # 'estimator__objective': ['reg:squarederror'],
#     'window_length': np.arange(1, 2)
# }

# xgb = make_reduction(XGBRegressor(random_state=0),strategy=strategy,)
# grid_xgb = ForecastingRandomizedSearchCV(forecaster = xgb,cv = splitter_HO,
#                                           param_distributions=params_xgb,scoring = MeanSquaredError(),
#                                           n_jobs = -1,n_iter=n_iter,random_state=0,
#                                         return_n_best_forecasters=10)
# grid_xgb.fit(y_train,fh=[1,2,3,4,5,6])


# results_xgb_HO = evaluate(y=y_train,forecaster=grid_xgb.best_forecaster_,cv=splitter_HO,
#                           return_data=True,scoring=MeanSquaredError())

# print(grid_xgb.best_params_)


# hyperparam_plot(results_xgb_HO, 'XGB')



# ############################# NN ########################################################


# params_nn = {
#     'estimator__hidden_layer_sizes': [(10,),(5,),
#                                       (15,),(8,8),
#                                       (20,),(8,)],   # Number of trees in the forest
#     'estimator__alpha': [0,0.0001, 0.001, 0.01],     # Learning rate
#     'estimator__shuffle': [True,False] ,               # Maximum depth of a tree
#     'estimator__learning_rate': ['constant', 'invscaling', 'adaptive'],             # Subsample ratio of the training instances
#     'estimator__early_stopping': [True,False],      # Subsample ratio of columns when constructing each tree
#     'estimator__max_iter': [500,700,1000,2000],                   # Minimum loss reduction required to make a further partition on a leaf node
#     'window_length' :np.arange(1,4),
#     'estimator__solver' :['adam','lbfgs']
# }


# nn = make_reduction(MLPRegressor(random_state=0),strategy=strategy)

# grid_nn = ForecastingRandomizedSearchCV(forecaster = nn,cv = splitter_HO,
#                                          param_distributions=params_nn,scoring = MeanSquaredError(),
#                                          n_jobs = -1,n_iter=n_iter,random_state=0,
#                                        return_n_best_forecasters=10,error_score='raise')
# grid_nn.fit(y_train,fh=[1,2,3,4,5,6])

# results_nn_HO = evaluate(y=y_train,forecaster=grid_nn.best_forecaster_,cv=splitter_HO,return_data=True,
#                          scoring=MeanSquaredError())

# print(grid_nn.best_params_)

# hyperparam_plot(results_nn_HO, 'NN')
############################# EXP Smoothing ########################################################
cv = ExpandingWindowSplitter(fh = np.arange(1,7),initial_window=len(metal)-test_size,step_length=1)
results_exp = evaluate(y=metal,forecaster=ExponentialSmoothing(),cv=cv,return_data=True,scoring=[MeanSquaredError(square_root=True),MeanAbsoluteError(),
                                                                                 MeanAbsoluteScaledError(),MeanSquaredScaledError(square_root=True)])

exp_error = results_exp['test_MeanSquaredError'].mean()
test_plot(results_exp, 'EXP')     
############################# ARIMA ########################################################

orders = AutoARIMA().fit(y_train)._get_fitted_params()['order']
summary = ARIMA(order=orders,with_intercept=False).fit(y_train).summary()
print(summary)
print(ARIMA(order=orders,with_intercept=False).fit(y_train).predict_residuals().mean())
ljungbox = sm.stats.acorr_ljungbox(ARIMA(order=orders,with_intercept=False).fit(y_train).predict_residuals(), 
                                   lags=[10], model_df=sum(orders),return_df=True)
ljungbox

results_val_arima = evaluate(y=y_train,forecaster=ARIMA(order=orders,with_intercept=False),cv=splitter_HO,return_data=True,scoring=[MeanSquaredError(square_root=True),
                                                                                 MeanAbsoluteError(),
                                                                                 MeanAbsoluteScaledError(),MeanSquaredScaledError(square_root=True)])

cv = ExpandingWindowSplitter(fh = np.arange(1,7),initial_window=len(metal)-test_size,step_length=1)
results_arima = evaluate(y=metal,forecaster=ARIMA(order=orders,with_intercept=False),cv=cv,return_data=True,scoring=[MeanSquaredError(square_root=True),
                                                                                 MeanAbsoluteError(),
                                                                                 MeanAbsoluteScaledError(),MeanSquaredScaledError(square_root=True)])
arima_error = results_arima['test_MeanSquaredError'].mean()

hyperparam_plot(results_val_arima, 'ARIMA')

############################# NAIVE ########################################################
results_val_naive = evaluate(y=y_train,forecaster=NaiveForecaster('mean'),cv=splitter_HO,return_data=True,scoring=[MeanSquaredError(square_root=True),
                                                                                 MeanAbsoluteError(),
                                                                                 MeanAbsoluteScaledError(),MeanSquaredScaledError(square_root=True)])

cv = ExpandingWindowSplitter(fh = np.arange(1,7),initial_window=len(metal)-test_size,step_length=1)
results_naive = evaluate(y=metal,forecaster=NaiveForecaster('mean'),cv=cv,return_data=True,scoring=[MeanSquaredError(square_root=True),
                                                                                 MeanAbsoluteError(),
                                                                                 MeanAbsoluteScaledError(),MeanSquaredScaledError(square_root=True)])
naive_error = results_naive['test_MeanSquaredError'].mean()

hyperparam_plot(results_val_naive, 'NAIVE')
############################### ENSEMBLE MODELLING ##############################################

forecasters = [#('Naive', NaiveForecaster('mean')),
              ('AR1', ARIMA(order=orders)),
              #('Theta', ThetaForecaster(deseasonalize=False)),
              #('NN',grid_nn.best_forecaster_),
             #('XGB', grid_xgb.best_forecaster_),
             ('LGBM', grid_lgbm.best_forecaster_)
             ]

# forecasters = []

# for i in range(len(grid_lgbm.n_best_forecasters_[:5])):
#     forecasters.append((str(i),grid_lgbm.n_best_forecasters_[i][1]))


forecaster = EnsembleForecaster(forecasters=forecasters,n_jobs=-1)
forecaster.fit(y=y_train,fh=[1,2,3,4,5,6])

cv = ExpandingWindowSplitter(fh = [1,2,3,4,5,6],initial_window=len(metal)-test_size,step_length=1)
results = evaluate(y=metal,forecaster=forecaster,cv=cv,return_data=True,scoring=[MeanSquaredError(square_root=True),
                                                                                 MeanAbsoluteError(),
                                                                                 MeanAbsoluteScaledError(),
                                                                                 MeanSquaredScaledError(square_root=True)])


test_plot(results,'Ensemble')

test_metrics = pd.DataFrame(results_lgbm[['test_MeanSquaredError','test_MeanAbsoluteError',
                                          'test_MeanAbsoluteScaledError','test_MeanSquaredScaledError']])

test_metrics.columns = ['RMSE_lgbm','MAE_lgbm','MASE_lgbm','RMSSE_lgbm']

test_metrics['RMSE_arima'] = results_arima['test_MeanSquaredError']
test_metrics['MAE_arima'] = results_arima['test_MeanAbsoluteError']
test_metrics['MASE_arima'] = results_arima['test_MeanAbsoluteScaledError']
test_metrics['RMSSE_arima'] = results_arima['test_MeanSquaredScaledError']


test_metrics['RMSE_exp'] = results_exp['test_MeanSquaredError']
test_metrics['MAE_exp'] = results_exp['test_MeanAbsoluteError']
test_metrics['MASE_exp'] = results_exp['test_MeanAbsoluteScaledError']
test_metrics['RMSSE_exp'] = results_exp['test_MeanSquaredScaledError']

test_metrics['RMSE_naive'] = results_naive['test_MeanSquaredError']
test_metrics['MAE_naive'] = results_naive['test_MeanAbsoluteError']
test_metrics['MASE_naive'] = results_naive['test_MeanAbsoluteScaledError']
test_metrics['RMSSE_naive'] = results_naive['test_MeanSquaredScaledError']

test_metrics['RMSE_ensemble'] = results['test_MeanSquaredError']
test_metrics['MAE_ensemble'] = results['test_MeanAbsoluteError']
test_metrics['MASE_ensemble'] = results['test_MeanAbsoluteScaledError']
test_metrics['RMSSE_ensemble'] = results['test_MeanSquaredScaledError']

test_metrics_avg = test_metrics.mean().to_frame().T

test_metrics_avg.index = [name]
test_metrics_avg.to_excel(excel_writer='C:\\Users\kosta\\Documents\\PhD related\\Short-term forecasting Ensemble\\'+str(name)+'_test_metrics.xlsx')

# test_metrics.filter(like='RMSE').mean()
# test_metrics.filter(like='MAE').mean()
# test_metrics.filter(like='MASE').mean()
# test_metrics.filter(like='RMSSE').mean()

test_plot2(results_lgbm,'LightGBM')

test_plot2(results,'Ensemble')

