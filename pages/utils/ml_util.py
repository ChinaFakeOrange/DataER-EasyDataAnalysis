# -*- ecoding: utf-8 -*-
# @Author: NUO

import pandas as pd
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,mean_absolute_error,r2_score,roc_auc_score,roc_curve,classification_report
from sklearn.preprocessing import PolynomialFeatures
import optuna
from fancyimpute import KNN
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import os
import streamlit as st

def calcDrop(res):
    # All variables with correlation > cutoff
    all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))

    # All unique variables in drop column
    poss_drop = list(set(res['drop'].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))

    # Drop any variables in same row as a keep variable
    p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
    q = list(set(p['v1'].tolist() + p['v2'].tolist()))
    drop = (list(set(q).difference(set(keep))))

    # Remove drop variables from possible drop
    poss_drop = list(set(poss_drop).difference(set(drop)))

    # subset res dataframe to include possible drop pairs
    m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2' ,'drop']]

    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
    for item in more_drop:
        drop.append(item)

    return drop
def corrX_new(df, cut = 0.9) :

    # Get correlation matrix and upper triagle
    corr_mtx = df.corr().abs()
    avg_corr = corr_mtx.mean(axis = 1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype('bool'))

    dropcols = list()

    res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target',
                                 'v2.target' ,'corr', 'drop' ]))

    for row in range(len(up ) -1):
        col_idx = row + 1
        for col in range (col_idx, len(up)):
            if(corr_mtx.iloc[row, col] > cut):
                if(avg_corr.iloc[row] > avg_corr.iloc[col]):
                    dropcols.append(row)
                    drop = corr_mtx.columns[row]
                else:
                    dropcols.append(col)
                    drop = corr_mtx.columns[col]

                s = pd.Series([ corr_mtx.index[row],
                                up.columns[col],
                                avg_corr[row],
                                avg_corr[col],
                                up.iloc[row ,col],
                                drop],
                              index = res.columns)

                res = pd.concat([res ,pd.DataFrame(s).T] ,ignore_index=True)

    # dropcols_names = calcDrop(res)

    return res
def dataFrameFill(train):
    valDict = {}
    train.replace('',np.nan,inplace=True)
    for cname in train.columns:
        if train[cname].dtype=='object' and train[[cname]].isnull().sum().values[0]>0:
            valDict[cname]=train[cname].mode().values[0]
        elif train[cname].dtype!='object' and train[[cname]].isnull().sum().values[0]!=len(train[cname]):
            X_filled_knn = KNN(k=5,verbose=False).fit_transform(train[[cname]])
            train[[cname]] = X_filled_knn
    train.fillna(value = valDict, inplace = True)
    return train
def cal_miss(df):
    percent_missing = ((df.isnull().sum()/df.shape[0])*100)
    num_values = len(df) - df.isnull().sum()
    missing_df = pd.DataFrame({'Number of Values':num_values,
                           'Missing Percentage (%)':percent_missing}).reset_index().sort_values(by='Missing Percentage (%)')
    # missing_df = missing_df.loc[(missing_df!=0).all(axis=1)]
    return missing_df

def one_hot(df):
    res = []
    cols = df.columns
    for col in cols:
        if df[col].dtype =='object':
            uv = df[col].unique()
            uv.sort()
            oh = dict(zip(uv,[i for i in range(len(df[col].unique()))]))
            res.append([col,oh])
    return res
def naive_preprocess(pdf,target_name,train_=True,res=None,outlier=False):
    if outlier:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    # missing_df = cal_miss(pdf)
    pdf = dataFrameFill(pdf)
    if train_:
        missing_df = cal_miss(pdf)
        pdf = pdf.drop(missing_df[missing_df['Missing Percentage (%)'] >= 30]['index'].tolist(), axis='columns')
        res = one_hot(pdf)
        for oh in res:
            pdf[oh[0]] = pdf[oh[0]].map(oh[1])
        target = pdf.pop(target_name)
        pdf = pd.DataFrame(scaler.fit_transform(pdf), columns=pdf.columns)
        return pdf,target,res
    else:
        if res:
            for oh in res:
                if oh[0] not in pdf.columns:
                    continue
                else:
                    pdf[oh[0]] = pdf[oh[0]].map(oh[1])
        pdf = dataFrameFill(pdf)
        missing_df = cal_miss(pdf)
        pdf[missing_df[missing_df['Missing Percentage (%)'] > 0]['index'].tolist()]=0
        # print(pdf['Cabin'].isnull().sum())
        # print(missing_df)
        return pd.DataFrame(scaler.fit_transform(pdf), columns=pdf.columns)
def hcor_remover(pdf):
    # pdf.drop(corrX_new(pdf,0.6),axis=1,inplace = True)
    res = corrX_new(pdf,0.7)
    l = []
    for i in range(len(res)):
        l.append(res['v1'].iloc[i])
        l.append(res['v2'].iloc[i])
    l = np.unique(l)
    cor_df = pdf[l]
    if len(l)>0:
        pdf.drop(l, axis=1, inplace=True)
        n_components = len(cor_df.columns)
        # if n_components < 1:
        #     n_components = 1
        pca = PCA(n_components=n_components)
        cor_df = pca.fit_transform(cor_df)
        cols = ['pca' + str(i) for i in range(n_components)]
        cor_df = pd.DataFrame(cor_df, columns=cols)
        pdf = pd.concat((pdf, cor_df), axis=1)
        return pdf,l
    else:
        return pdf,None


def blend_c(models,x,alpha=0.5):
    results = models[0].predict_proba(x)*alpha
    results+=models[1].predict_proba(x)*(1-alpha)/3
    results+=models[2].predict_proba(x)*(1-alpha)/3
    results+=models[3].predict_proba(x)*(1-alpha)/3
    return np.argmax(results,axis=1)

def blend_proba(models,x,alpha=0.5):
    results = models[0].predict_proba(x)*alpha
    results+=models[1].predict_proba(x)*(1-alpha)/3
    results+=models[2].predict_proba(x)*(1-alpha)/3
    results+=models[3].predict_proba(x)*(1-alpha)/3
    # results = models[4].predict_proba(x)
    return results
def blend_r(models,x,alpha=0.5):
    results = models[0].reg_predict(x)*alpha
    results+=models[1].predict(x)*(1-alpha)/3
    results+=models[2].predict(x)*(1-alpha)/3
    results+=models[3].predict(x)*(1-alpha)/3
    return results


class Classifier:
    def __init__(self, X, y, task,n_trials=5,loss_f='F1'):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        self.X = X
        self.y = y
        self.kf = KFold(n_splits=3,shuffle=True)
        self.n_trials = n_trials
        self.task = task
        self.params = {}
        self.best_model = None
        self.model = None
        self.auc = False
        self.n_class = len(np.unique(self.y))

        if self.task == 'Classification' or self.task == '分类任务':
            if loss_f == 'AUC':
                self.loss_f = self.AUC
                self.auc =True
            elif loss_f == 'ACC':
                self.loss_f = accuracy_score
            else:
                self.loss_f = self.F1
        else:
            if loss_f == 'R2':
                self.loss_f = self.R2
            elif loss_f == 'MSE':
                self.loss_f = mean_squared_error
            elif loss_f == 'MAE':
                self.loss_f = mean_absolute_error
            else:
                self.loss_f = self.RMSE

        self.tuning1(),self.tuning3(),self.tuning4()
        if self.task == 'Classification' or self.task == '分类任务':
            self.tuning6()
        else:
            self.tuning9()
        self.tuning10()

    def AUC(self,y,yp):
        return roc_auc_score(y,yp,average='micro') if self.n_class<3 else roc_auc_score(y,yp,average='micro',multi_class = 'ovr')
    def RMSE(self,y,yp):
        return np.sqrt(mean_squared_error(y,yp))
    def F1(self,y,yp):
        return f1_score(y,yp,average='micro')
    def R2(self,y,yp):
        return 1-r2_score(y,yp)

    def callback(self,study, trial):
        if study.best_trial == trial:
            self.best_model=self.model
    def my_scorer(self,estimator, x, y):
        if self.auc and self.n_class>2:
            yPred = estimator.predict_proba(x)
        else:
            yPred = estimator.predict(x)
        return self.loss_f(y,yPred)

    def objective1(self, trial):
        param = {'learning_rate': trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                 'n_estimators': trial.suggest_int("n_estimators", 100, 2500),
                 'max_depth': trial.suggest_int("max_depth", 3, 11, step=1),
                 'min_child_weight': trial.suggest_int("min_child_weight", 1, 7),
                 'gamma': trial.suggest_float("gamma", 0, 1),
                 'subsample': trial.suggest_float("subsample", 0.5, 1, log=True),
                 'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1, log=True),
                 # 'nthread': -1,
                 'verbosity': 0,
                 # 'tree_method' :'gpu_hist'
                 }
        if self.task == 'Classification' or self.task == '分类任务':
            self.model = XGBClassifier(**param)
        else:
            self.model = XGBRegressor(**param)

        s = 0
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test, y_train, y_test = self.X.iloc[train_index], self.X.iloc[test_index], self.y.iloc[train_index], self.y.iloc[test_index]
            self.model.fit(X_train, y_train,verbose = False)
            s+=self.my_scorer(self.model,X_test,y_test)
        return s/3

    def tuning1(self):
        study = optuna.create_study(direction="maximize" if self.task == 'Classification' or self.task == '分类任务' else 'minimize')

        func = lambda trial: self.objective1(trial)

        study.optimize(func, n_trials=self.n_trials,callbacks=[self.callback])
       # param = study.best_params
        # param.update({'nthread': -1, 'random_state': 42, 'verbosity': 0,'tree_method' :'gpu_hist'})
        # self.params['xgb'] = param
        self.params['xgb'] = self.best_model


    def objective3(self, trial):
        param = {'num_leaves': trial.suggest_int("num_leaves", 3, 12),
                 'learning_rate': trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                 'n_estimators': trial.suggest_int("n_estimators", 100, 2500),
                 'verbosity': -1,
                 'min_child_samples':trial.suggest_int("min_child_samples", 10, 100,),
                 'min_child_weight':trial.suggest_float("min_child_weight", 1e-4, 1e2, log=True),
                  'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1),
                'reg_alpha': trial.suggest_float("reg_alpha", 1e-10, 10,),
                'reg_lambda': trial.suggest_float("reg_lambda", 1e-10, 10,),
                 'bagging_fraction':trial.suggest_float("bagging_fraction", 0.5, 1,),
                 # 'device': 'gpu',
                # 'feature_fraction': trial.suggest_float("feature_fraction", 1e-10, 1, log=True),
                 }

        if self.task == 'Classification' or self.task == '分类任务':
            self.model = LGBMClassifier(**param)
        else:
            self.model = LGBMRegressor(**param)
        s = 0
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test, y_train, y_test = self.X.iloc[train_index], self.X.iloc[test_index], self.y.iloc[train_index], self.y.iloc[test_index]
            self.model.fit(X_train, y_train)
            s+=self.my_scorer(self.model,X_test,y_test)
        return s/3

    def tuning3(self):
        study = optuna.create_study(direction="maximize" if self.task == 'Classification' or self.task == '分类任务' else 'minimize')

        func = lambda trial: self.objective3(trial)

        study.optimize(func, n_trials=self.n_trials,callbacks=[self.callback])
        self.params['lgb'] =self.best_model
    def objective4(self, trial):
        param = {
            'l2_regularization': trial.suggest_float("l2_regularization", 1e-4, 100, log=True),
            'early_stopping': True,
            'learning_rate': trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            'max_iter': trial.suggest_int("max_iter", 100, 5000),
            'max_depth': trial.suggest_int("max_depth", 3, 11),
            'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 15),
            'max_leaf_nodes': trial.suggest_int("max_leaf_nodes", 3, 27),
        }

        if self.task == 'Classification' or self.task == '分类任务':
            param.update({'class_weight': 'balanced'})
            self.model = HistGradientBoostingClassifier(**param)
        else:
            self.model = HistGradientBoostingRegressor(**param)
        s = 0
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test, y_train, y_test = self.X.iloc[train_index], self.X.iloc[test_index], self.y.iloc[train_index], self.y.iloc[test_index]
            self.model.fit(X_train, y_train)
            s+=self.my_scorer(self.model,X_test,y_test)
        return s/3
    def tuning4(self):
        study = optuna.create_study(direction="maximize" if self.task == 'Classification' or self.task == '分类任务' else 'minimize')

        func = lambda trial: self.objective4(trial)

        study.optimize(func, n_trials=5,callbacks=[self.callback])
        self.params['hist'] = self.best_model


    def objective6(self, trial):
        param = {'n_neighbors': trial.suggest_int("n_neighbors", 5, 100),
                 'weights': trial.suggest_categorical("weights", ['uniform', 'distance']),
                 'metric': trial.suggest_categorical("metric", ['minkowski', 'euclidean', 'manhattan'])}

        self.model = KNeighborsClassifier(**param)

        s = 0
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test, y_train, y_test = self.X.iloc[train_index], self.X.iloc[test_index], self.y.iloc[train_index], self.y.iloc[test_index]
            self.model.fit(X_train, y_train)
            s+=self.my_scorer(self.model,X_test,y_test)
        return s/3

    def tuning6(self):
        study = optuna.create_study(direction="maximize")

        func = lambda trial: self.objective6(trial)

        study.optimize(func, n_trials=self.n_trials*4,callbacks=[self.callback])
        self.params['en'] = self.best_model

    def objective7(self, trial):
        param = {
                  'alpha':trial.suggest_float("alpha", 1e-10, 1e4, log=True),
                  }
        self.model = Lasso(**param)
        s = 0
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test, y_train, y_test = self.X.iloc[train_index], self.X.iloc[test_index], self.y.iloc[train_index], self.y.iloc[test_index]
            self.model.fit(X_train, y_train)
            s+=self.my_scorer(self.model,X_test,y_test)
        return s/3


    def tuning7(self):
        study = optuna.create_study(direction="maximize" if self.task == 'Classification' or self.task == '分类任务' else 'minimize')

        func = lambda trial: self.objective7(trial)

        study.optimize(func, n_trials=self.n_trials,callbacks=[self.callback])
        self.params['lasso'] = self.best_model
    def objective8(self, trial):
        param = {
                'alpha':trial.suggest_float("alpha", 1e-10, 1e4, log=True),
                  }
        self.model = Ridge(**param)
        s = 0
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test, y_train, y_test = self.X.iloc[train_index], self.X.iloc[test_index], self.y.iloc[train_index], self.y.iloc[test_index]
            self.model.fit(X_train, y_train)
            s+=self.my_scorer(self.model,X_test,y_test)
        return s/3

    def tuning8(self):
        study = optuna.create_study(direction="maximize" if self.task == 'Classification' or self.task == '分类任务' else 'minimize')
        func = lambda trial: self.objective8(trial)
        study.optimize(func, n_trials=self.n_trials,callbacks=[self.callback])
        self.params['rg'] = self.best_model

    def objective9(self, trial):
        param = {
            'alpha': trial.suggest_float('alpha', 1e-8, 1, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 1e-8, 1, log=True),
            'tol': trial.suggest_float('tol', 1e-8, 1.0, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 10000, log=True),
                  }
        self.model = ElasticNet(**param)
        s = 0
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test, y_train, y_test = self.X.iloc[train_index], self.X.iloc[test_index], self.y.iloc[train_index], self.y.iloc[test_index]
            self.model.fit(X_train, y_train)
            s+=self.my_scorer(self.model,X_test,y_test)
        return s/3

    def tuning9(self):
        study = optuna.create_study(direction="maximize" if self.task == 'Classification' or self.task == '分类任务' else 'minimize')

        func = lambda trial: self.objective9(trial)

        study.optimize(func, n_trials=self.n_trials*4,callbacks=[self.callback])
        self.params['en'] = self.best_model

    def objective10(self, trial):
        param = {'learning_rate': trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                 'n_estimators': trial.suggest_int("n_estimators", 100, 2000),
                 'max_depth': trial.suggest_int("max_depth", 3, 11, step=1),
                 'min_child_weight': trial.suggest_int("min_child_weight", 1, 7),
                 'gamma': trial.suggest_float("gamma", 0, 1),
                 'subsample': trial.suggest_float("subsample", 0.5, 1, log=True),
                 'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1, log=True),
                 'nthread': -1,
                 'verbosity': 0,
                 # 'tree_method' :'gpu_hist'
                 }
        if self.task == 'Classification' or self.task == '分类任务':
            self.model = StackingModel(self.params,XGBClassifier(**param))
            self.model.clf_fit(self.X_train, self.y_train)
            if self.auc and self.n_class>2:
                s = self.loss_f(self.y_test, self.model.predict_proba(self.X_test))
            else:
                s = self.loss_f(self.y_test, self.model.clf_predict(self.X_test))
        else:
            self.model = StackingModel(self.params,XGBRegressor(**param))
            self.model.reg_fit(self.X_train, self.y_train)
            s = self.loss_f(self.y_test, self.model.reg_predict(self.X_test))
        return s

    def tuning10(self):
        study = optuna.create_study(direction="maximize" if self.task == 'Classification' or self.task == '分类任务' else 'minimize')

        func = lambda trial: self.objective10(trial)

        study.optimize(func, n_trials=self.n_trials,callbacks=[self.callback])
        self.params['stack'] = self.best_model

class StackingModel:
    def __init__(self, base_models,meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.f_key = list(self.base_models.keys())
        self.poly = PolynomialFeatures(2)

    def clf_fit(self,X,y):
        for k in self.f_key:
            self.base_models[k].fit(X,y)

        preds = self.base_models[self.f_key[0]].predict_proba(X)
        for k in range(1,len(self.f_key)):
            preds = np.concatenate((preds,self.base_models[self.f_key[k]].predict_proba(X)),axis=1)

        self.meta_model.fit(preds,y)
        return self

    def reg_fit(self,X,y):
        for k in self.f_key:
            self.base_models[k].fit(X,y)

        preds = np.expand_dims(self.base_models[self.f_key[0]].predict(X),1)
        for k in range(1, len(self.f_key)):
            preds = np.concatenate((preds, np.expand_dims(self.base_models[self.f_key[k]].predict(X),1)),axis=1)
        # preds = pd.DataFrame(self.poly.fit_transform(preds), columns=['poly' + str(i) for i in range(15)])
        # preds.pop('poly0')
        self.meta_model.fit(preds, y)
        return self

    def clf_predict(self,X):
        preds = self.base_models[self.f_key[0]].predict_proba(X)
        for k in range(1, len(self.f_key)):
            preds = np.concatenate((preds, self.base_models[self.f_key[k]].predict_proba(X)),axis=1)
        res = self.meta_model.predict(preds)
        return res

    def reg_predict(self,X):
        preds = np.expand_dims(self.base_models[self.f_key[0]].predict(X),1)
        for k in range(1, len(self.f_key)):
            preds = np.concatenate((preds, np.expand_dims(self.base_models[self.f_key[k]].predict(X),1)),axis=1)
        # preds = pd.DataFrame(self.poly.fit_transform(preds), columns=['poly' + str(i) for i in range(15)])
        # preds.pop('poly0')
        res = self.meta_model.predict(preds)
        return res
    def predict_proba(self,X):
        preds = self.base_models[self.f_key[0]].predict_proba(X)
        for k in range(1, len(self.f_key)):
            preds = np.concatenate((preds, self.base_models[self.f_key[k]].predict_proba(X)),axis=1)
        proba = self.meta_model.predict_proba(preds)
        return proba

    def score(self, X,y,task='Classification'):
        if task == 'Classification' or task == '分类任务':
            preds = self.base_models[self.f_key[0]].predict_proba(X)
            for k in range(1, len(self.f_key)):
                preds = np.concatenate((preds, self.base_models[self.f_key[k]].predict_proba(X)), axis=1)
            res = self.meta_model.predict(preds)
            s = f1_score(y,res,average = 'micro')
        else:
            preds = np.expand_dims(self.base_models[self.f_key[0]].predict(X), 1)
            for k in range(1, len(self.f_key)):
                preds = np.concatenate((preds, np.expand_dims(self.base_models[self.f_key[k]].predict(X), 1)),axis=1)
            # preds = pd.DataFrame(self.poly.fit_transform(preds), columns=['poly' + str(i) for i in range(15)])
            # preds.pop('poly0')
            res = self.meta_model.predict(preds)
            s=r2_score(y,res)
        return s


def ml_pip(df,dp_uv,dp_tg,cg_ops,dp_tk,
           dp_mt,sd_th,sd_mw,test_ratio,dp_ld,ml_file):
    pg_txt = "训练中，请稍等"
    pg = st.progress(0,pg_txt)
    pg_time = 0

    df.replace('',np.nan,inplace=True)
    outlier_p,hc_p,at_fea,fea_ex,r_fit=False,False,False,False,False
    pdf = df
    if len(dp_uv)>0:
        dp_uv = [i for i in dp_uv]
        pdf.drop(dp_uv,axis=1,inplace=True)

    outlier_p = True if 'Data Balance' in cg_ops or '数据平衡' in cg_ops else False
    hc_p = True if 'Collinearity Solver' in cg_ops or '共线性处理' in cg_ops else False
    at_fea = True if 'Auto-Feature'  in cg_ops or "自选变量" in cg_ops else False
    fea_ex = True if 'Feature Extension' in cg_ops or '变量扩展' in cg_ops else False
    r_fit = True if 'Refit'  in cg_ops or '载入模型' in cg_ops else False

    if r_fit:
        models = joblib.load(ml_file[dp_ld])[1]
        (stack_model, xgb, lgb, hist_md, en) = models

    pdf, target, res = naive_preprocess(pdf=pdf, target_name=dp_tg,
                                        outlier=outlier_p)
    ini_param = pdf.columns

    pg_time+=10
    pg.progress(pg_time,pg_txt)

    #solve collinearity
    if hc_p:
        pdf, fuse_names = hcor_remover(pdf)
    else:
        fuse_names = None
    if dp_tk=='Classification' or dp_tk=='分类任务':
        fsc = XGBClassifier()
    else:
        fsc = XGBRegressor()

# feature importance map
    fsc.fit(pdf, target)
    fea = pd.DataFrame(fsc.feature_importances_, index=pdf.columns).sort_values(0, ascending=False)

    pg_time+=10
    pg.progress(pg_time,pg_txt)


#polynomial transformation
    if fea_ex:
        poly_fea = fea.index
        if len(poly_fea) > 4 and len(poly_fea) < 11:
            poly = PolynomialFeatures(2)
            poly_df = pd.DataFrame(poly.fit_transform(pdf[poly_fea[:5]]),
                                   columns=['poly' + str(i) for i in range(21)])
            poly_df.pop('poly0')
            pdf.drop(poly_fea[:5], inplace=True, axis=1)
            pdf = pd.concat((pdf, poly_df), axis=1)
            fea_s = poly_fea[:5]

        elif len(poly_fea) > 3:
            poly = PolynomialFeatures(2)
            poly_df = pd.DataFrame(poly.fit_transform(pdf[poly_fea[:4]]),
                                   columns=['poly' + str(i) for i in range(15)])
            poly_df.pop('poly0')
            pdf.drop(poly_fea[:4], inplace=True, axis=1)
            pdf = pd.concat((pdf, poly_df), axis=1)
            fea_s = poly_fea[:4]

        elif len(poly_fea) > 2:
            poly = PolynomialFeatures(2)
            poly_df = pd.DataFrame(poly.fit_transform(pdf[poly_fea[:3]]),
                                   columns=['poly' + str(i) for i in range(10)])
            poly_df.pop('poly0')
            pdf.drop(poly_fea[:3], inplace=True, axis=1)
            pdf = pd.concat((pdf, poly_df), axis=1)
            fea_s = poly_fea[:3]
    else:
        fea_s = None


    pg_time+=10
    pg.progress(pg_time,pg_txt)
#feature auto selection
    if at_fea:
        fsc.fit(pdf, target)
        fea_ = pd.DataFrame(fsc.feature_importances_, index=pdf.columns).sort_values(0, ascending=False)
        fea_ = fea_[fea_ >= (1 / (10 * len(fea_)))].dropna().index
        pdf = pd.DataFrame(pdf[fea_])
        # print("Auto Selected Features: ", fea_)
    kept_param = pdf.columns

    X_trainValid, X_test, y_trainValid,y_test = train_test_split(pdf, target,
                                                                    test_size=test_ratio)

    pg_time+=10
    pg.progress(pg_time,pg_txt)


    if r_fit==False:
        params = Classifier(pdf, target, dp_tk,
                           sd_th, dp_mt).params
        (xgb, lgb, hist_md,
         en, stack_model) = (params['xgb'],params['lgb'],
                            params['hist'],params['en'],params['stack'])

    pg_time+=50
    pg.progress(pg_time,pg_txt)

    if dp_tk=='Classification' or dp_tk=='分类任务':
        blender = blend_c
        score_f = accuracy_score
        stack_model.clf_fit(X_trainValid,y_trainValid)

    else:
        blender = blend_r
        score_f = r2_score
        stack_model.reg_fit(X_trainValid, y_trainValid)


    xgb.fit(X_trainValid, y_trainValid, verbose=False)
    lgb.fit(X_trainValid, y_trainValid)
    hist_md.fit(X_trainValid, y_trainValid)
    en.fit(X_trainValid,y_trainValid)

    models = [stack_model, xgb, lgb, hist_md, en]
    train_score = score_f(y_trainValid, blender(models, X_trainValid, sd_mw))
    test_score = score_f(y_test, blender(models,X_test, sd_mw))
    meta_score = stack_model.score(X_test,y_test, dp_tk)

    if dp_tk == 'Classification' or dp_tk=='分类任务':
        las = df[dp_tg].unique()
        las.sort()
        cr = classification_report(y_test,
                                   blender(models, X_test, sd_mw),
                                   labels=[i for i in range(len(las))],
                                   target_names=[str(i) for i in las],
                                   output_dict=True)

        cr = pd.DataFrame(cr).iloc[:-1, :]
        cr = cr.round(3)
        fig1 = ff.create_annotated_heatmap(cr.values, x=cr.columns.to_list(), y=cr.index.to_list(),
                                          colorscale='sunsetdark')
        # add title
        fig1.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                          # width=900, height=600,
                          yaxis=dict(scaleanchor="x", scaleratio=1),
                          xaxis=dict(constrain='domain'), )

        # add colorbar
        fig1['data'][0]['showscale'] = True



        y_scores = blend_proba(models, X_test, sd_mw)
        y_onehot = pd.get_dummies(df[dp_tg][y_test.index])

        fig2 = go.Figure()
        fig2.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        for i in range(len(y_onehot.columns)):
            y_true = y_onehot.iloc[:, i]
            y_score = y_scores[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig2.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        fig2.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
        )

    else:
        las = None
        x_range = X_test
        y_range = blender(models, x_range, sd_mw)
        y_range = pd.concat(
            (pd.DataFrame(y_test).reset_index(drop=True), pd.DataFrame(y_range).reset_index(drop=True)),
            axis=1).sort_values(by=0).reset_index(drop=True)
        fig1 = go.Figure([
            go.Scatter(x=y_range.index, y=y_range.iloc[:, 0].values,
                       name='test', mode='markers'),
            go.Scatter(x=y_range.index, y=y_range.iloc[:, 1].values,
                       name='pred', mode='markers')
        ])

        fig1.update_layout(
            title="Prediction Overview",
            xaxis_title='Series',
            yaxis_title='Values'
        )
        # Save the plot as an HTML file

        X_Full = pdf
        y_pf = blender(models, X_Full, sd_mw)
        tt = pd.DataFrame()
        tt['target'] = target
        tt['pred'] = pd.DataFrame(y_pf)
        tt['split'] = 'train'
        tt.loc[y_test.index, 'split'] = 'test'

        fig2 = px.scatter(tt, x='pred', y='target',
                         marginal_x='histogram', marginal_y='histogram',
                         color='split', trendline='ols'
                         )
        fig2.update_traces(histnorm='probability', selector={'type': 'histogram'})


    res_txt = "混合模型分数: "+str(round(meta_score,3))
    res_txt += ("\n平均训练分数: "+str(round(train_score,3)))
    res_txt += ("\n平均测试分数:"+str(round(test_score,3)))

    if dp_tk=='Classification' or dp_tk =='分类任务':
        ml_model_path = os.path.join('ml_saved\\model', 'clf' +
                                     str(len(os.listdir('ml_saved\\model'))) + 'x' + '.pkl')
    else:
        ml_model_path = os.path.join('ml_saved\\model', 'reg' +
                                     str(len(os.listdir('ml_saved\\model'))) + 'x' + '.pkl')

    saved_param = (kept_param,models,outlier_p,hc_p,
                 at_fea,fea_ex,ini_param,
                 dp_uv,res,fuse_names,fea_s,
                 dp_tk,las)
    joblib.dump(saved_param,ml_model_path)

    new_ck = os.path.split(ml_model_path)[-1]
    ml_file[new_ck] = ml_model_path

    pg_time+=10
    pg.progress(pg_time,pg_txt)
    pg.empty()

    return res_txt, fig1, fig2,ml_file,[k for k,v in ml_file.items()]

def file_create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def ml_pred(dp_ld,ml_file,sd_mw,df):
    (kept_param, models, outlier_p, hc_p,
     at_fea, fea_ex, ini_param,
     dp_uv,res,fuse_names,fea_s,dp_tk,las) = joblib.load(ml_file[dp_ld])
    df.replace('',np.nan,inplace=True)
    if dp_uv != []:
        pdf = df.drop(dp_uv, axis=1)
    else:
        pdf = df
    pdf = naive_preprocess(pdf, kept_param, False, res=res,
                           outlier=outlier_p)
    pdf = pdf[ini_param]
    if hc_p and fuse_names is not None:
        print("Performing PCA preprocessing...")
        cor_df = pdf[fuse_names]
        pdf.drop(fuse_names, axis=1, inplace=True)
        n_components = len(fuse_names)
        pca = PCA(n_components=n_components)
        cor_df = pca.fit_transform(cor_df)
        cols = ['pca' + str(i) for i in range(n_components)]
        cor_df = pd.DataFrame(cor_df, columns=cols)
        pdf = pd.concat((pdf, cor_df), axis=1)
    if fea_s is not None:
        if len(fea_s) == 5:
            ep = 21
        elif len(fea_s) == 4:
            ep = 15
        else:
            ep = 10
        poly = PolynomialFeatures(2)
        poly_df = pd.DataFrame(poly.fit_transform(pdf[fea_s]),
                               columns=['poly' + str(i) for i in range(ep)])
        poly_df.pop('poly0')
        pdf.drop(fea_s, inplace=True, axis=1)
        pdf = pd.concat((pdf, poly_df), axis=1)

    pdf = pdf[kept_param]
    if dp_tk == 'Classification' or dp_tk=='分类任务':
        pred_res = pd.DataFrame(blend_c(models, pdf,sd_mw), columns=['Results'])
        pred_proba = pd.DataFrame(blend_proba(models, pdf, sd_mw), columns=[str(i) for i in las])
        pred_res = pd.concat((pred_res, pred_proba), axis=1)
        p_path = os.path.join("ml_saved\\res",'pred'+str(len(os.listdir('ml_saved\\res'))) + '.csv')
        pred_res = pd.concat((df.reset_index(drop=True), pred_res), axis=1)
        pred_res['Results'] = pred_res['Results'].map(dict(zip([i for i in range(len(las))], las)))
        # pred_res.to_csv(
        #     p_path, index=False,encoding='utf-8'
        # )
    else:

        pred_res = pd.DataFrame(blend_r(models, pdf, sd_mw), columns=['Results'])
        p_path = os.path.join("ml_saved\\res",'pred'+str(len(os.listdir('ml_saved\\res')))+'.csv')
        pred_res = pd.concat((df.reset_index(drop=True), pred_res), axis=1)
        # pred_res.to_csv(
        #     p_path, index=False,encoding='utf-8'
        # )

    return pred_res
