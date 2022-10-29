#!/usr/bin/env python
# coding: utf-8

# ### Important imports

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, MinMaxScaler  

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, cross_val_score

from sklearn.feature_selection import chi2, SelectKBest, RFE, SelectFromModel, SelectPercentile

from sklearn.utils import resample

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score,                             recall_score, roc_auc_score, precision_score, recall_score, classification_report

from sklearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline, Pipeline

import xgboost as xgb


# ## Functions to print basic info about dataset

# In[8]:


def unique_col(data):
    """Printing number of unique value in all columns in given dataset."""
    print("How many unique value have each columns:")
    for col in data.columns:
        print(f"- {col.upper()}:  {data[col].nunique()}. This is {round(data[col].nunique() / data.shape[0] ,3)}% of data")

def null_value(data):
    """
    Printing list with columns which contain null value, but also return response (yes or no) if there is \
    any column with null value.
    """
    col_with_null = []
    count = 0
    for col in data.columns:
        null = data[col].isnull().sum()
        if null > 0:
            col_with_null.append(col)
        if data[col].notnull().sum() != data.shape[0]:
            count += 1
    if count != 0:
        response = 'Yes'
    else:
        response = 'No'
    return col_with_null, response

def column_to_delete(data):
    """Print list of columns which have only unique value and can be delete from dataset."""
    col_to_delete = []
    for col in data.columns:
        if (data[col].nunique() / data.shape[0]) == 1:
            col_to_delete.append(col)
    return col_to_delete

def return_categorical(data):
    """Return list with categorical variables"""
    cat_col = [col_name for col_name in data if data[col_name].dtype == 'O']
    return cat_col


def return_numeric(data):
    """Return list with numeric variables"""
    num_col = [col_name for col_name in data if data[col_name].dtype != 'O']
    return num_col
    
def print_basic_info(data, target, threshold):
    """
    This function print basic info about dataset which has only 2 classes to predict.
    Parameters:
    > data: our dataset,
    > target: which column we want to predict,
    > threshold: how many % we consider our data to be inbalance.
    """
    data_y = data[target]
    y = round((data_y.sum() / data_y.shape[0])* 100 ,2)
    col_with_null, response = null_value(data)
    num_col = return_numeric(data)
    cat_col = return_categorical(data)
    col_to_delete = column_to_delete(data)
    
    if y < threshold:
        data_unbalanced = 'Yes'
    else: 
        data_unbalanced = 'No'
    
    print("BASIC INFORMATION ABOUT THE DATASET \n")
    print("--------------------------------------------------------------------------")
    print(f"Number of row: {data.shape[0]}, number of columns {data.shape[1]} \n")
    print("--------------------------------------------------------------------------")
    print(f"Is there null values: {response} \n")
    print(f"Columns with null values: {col_with_null} \n")   
    print("--------------------------------------------------------------------------")
    print(f"Categorical columns: {cat_col} \n") 
    print(f"Numeric columns: {num_col} \n") 
    print(f"{unique_col(data)} \n")
    print("--------------------------------------------------------------------------")
    print(f"Is dataset unbalaced? {data_unbalanced}. \nPercentage of clients who made churn is: {y}% \n")
    print(f"Columns to delete because they contains only unique value: {col_to_delete}")


# In[ ]:


def col_to_drop_after_high_corr(data, threshold):
    """
    Parameters:
    > data: our dataset
    > threshold: % above which we want to remove our columns
    This function delete from our dataset columns with high correlation which we can manually choose.
    """
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop


# ## Functions for selecting features

# In[10]:


def print_feature_summary(feature_name, cor_support, chi_support, rfe_support, l2_support, tree_support):
    """
    Return DataFrame with columns names and different methods to select best feature \
    and count how many times each column was selected and and ranks them from the most common.
    """
    # put all selection together
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 
                                         'Pearson':cor_support, 
                                         'Chi-2':chi_support, 
                                         'RFE':rfe_support, 
                                         'Logistics':l2_support,
                                         'Random Forest':tree_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    print("Name of this dataframe: 'feature_selection_df'")
    return feature_selection_df


# ## Functions for transforming data

# In[11]:


def preprocessing(X, preprocessor=None):
    """
    This function preprocess data by function like StandardScaler, MinMaxScaler, \
    Normalizer depending on which method we want to use.
    Parameters:
    < X - our X data
    < preprocessor - method for preprocess X.
    """
    if preprocessor is not None:
        preprocessor.fit(X.values)
        X_preprocessed = X.copy()
        X_preprocessed[:] = preprocessor.transform(X.values)
        X = X_preprocessed
    return X


# ## Functions to build pipeline for many models

# In[14]:


def looking_for_parameters(X_train, X_test, y_train, y_test, models, params, models_name):
    """
    This functions takes:
    < X,
    < y, 
    < models - models we want to check
    < params - which parameters we want to check in certain model
    < models_name - list with model names (it shoul be in the same order as given models)
    And returns DataFrame with metrics for each model and best hyperparameters for them.
    """
    metrics_value = []

    for model, param, name in zip(models, params, models_name):
        print(f"Looking for best parameters for {name}")
        optimizer = find_best_option(model=model, X_train=X_train, y_train=y_train, best_models='yes', params=param) 
        best_param = optimizer.best_params_
        target = y_test
        prediction = optimizer.predict(X_test)
        accuracy, precision, recall, f1, mislabeled, total =  calculate_metrics(target, prediction, average='macro')
        metrics_value.append({
            'model_name': name,
            'best_params': best_param,
            'accuracy': round(accuracy,2),
            'precision': round(precision,2),
            'recall': round(recall,2),
            'f1': round(f1,2),
            'mislabeled': mislabeled,
            'total': total
        })
        print("Done!")
    df_metrics_value = pd.DataFrame(metrics_value, columns=['model_name', 'best_params', 'accuracy','precision',                                                           'recall', 'f1', 'mislabeled', 'total'])
    df_metrics_value = df_metrics_value.sort_values(by ='f1', ascending=False)
    df_metrics_value = df_metrics_value.style.set_properties(subset=['best_params'], **{'width': '320px'})
    
    return df_metrics_value, metrics_value


# In[ ]:


def find_best_option(model, X_train, y_train, params=None, balance=None, best_models=None):
    """
    This functions takes:
    < model
    < X_train
    < y_train 
    < params (default: None) - which parameters for certain model we want to use
    < balance (default: None) - option to balance unbalanced data, for example using SMOTE method
    < best_models (default: None)
    And returns fit model on which we can make further transformation. 
    """
    if best_models == None:
        if balance == None:
            pipeline = Pipeline([('clf', model)])
        else:
            pipeline = Pipeline([('sm', balance), 
                                 ('clf', model)])

        gs = GridSearchCV(pipeline, params, cv=5, n_jobs=-1, scoring='f1', return_train_score=True)
        gs.fit(X_train, y_train)
        return gs
    else:
        optimizer = GridSearchCV(model, 
                                 param_grid=params, 
                                 cv=5, 
                                 scoring="f1",
                                 n_jobs=-1)
        optimizer.fit(X_train, y_train)
        return optimizer


# ## Functions for visualizations

# In[15]:


def plot_countplot(df, feature, title='', size=2):
    """
    This functions takes:
    < df - our data
    < feature - columns to count their value.
    < title - to name out plot
    < size
    And returns count plot with labels in % on how many each of value there is in given feature.
    """
    f, ax = plt.subplots(1,1, figsize=(3*size,2*size))
    total = float(len(df))
    sns.countplot(x=df[feature], order = df[feature].value_counts().index, palette='Set3')
    plt.title(title, fontsize=16)
    if(size > 5):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()


# In[16]:


def plot_hist_box(columns, data):
    """ Plotting histogram and boxplot for selected columns."""
    for col in columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))    
        ax1.hist(data[col], bins=25)
        ax1.set_xlim(0, max(data[col]))
        ax2.boxplot(data[col])
        fig.suptitle(col, fontsize=16);


# In[17]:


def compare_with_target(columns, target, data):
    """
    Print plots where we can see how the target is distributed in the selected columns. Parameters:
    < columns - columns to check
    < target - variable to see distribution.
    """
    for col in columns:
        plt.figure(figsize=(15,6))
        sns.countplot(x=col, data=data, hue=target)
        plt.title(col, fontsize=20);
    
        print(col.upper())
        print((pd.DataFrame(data.groupby([col])[target].value_counts(normalize=True))).T)
        print('-------------------------------------')
        print()


# In[18]:


def corr_plot(data, target):
    """ Print corelattion plot with targat variable. Parameters:
    < data - data for which we want to check corelattion
    < target - variable to check correlation with.
    """
    corr = data.corr()[target]
    corr.plot.bar(figsize=(14,5));
    for i in range(len(corr)):
        plt.text(i, corr[i], round(corr[i],3), ha='center', va='bottom')
    plt.title(f"Correletion with {target}", fontsize = 16);


# In[19]:


def target_in_outliers(columns, target, data):
    """ 
    This function takes:
    < columns: dict which contains names of the model as key and model as value,
    < target - the column we want to compare other columns to,
    < data - our data,
    And print information about distribution target columns in outliers in other columns and return DataFrame with treshold \
    from we can names value outliers and also count how many rows there is to consider them as outliers.
    """
    print("What % of distribution the target in outliers has in given columns: \n")
    outliers_value = []
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1 
        fence_low = q1 - (1.5 * iqr)
        fence_high = q3 + (1.5 * iqr)
        cnt_low = len(data[data[col] <= fence_low])
        cnt_high = len(data[data[col] >= fence_high])
        outliers_value.append({'columns': col,
                               'fence_low': fence_low,
                               'cnt_low': cnt_low,
                               'fence_high': fence_high,
                               'cnt_high': cnt_high})
        print(col.upper())
        print(data[(data[col] > fence_high) | (data[col] < fence_low)][target].value_counts(normalize=True))
        print('-------------------------------------')
    df_outliers_value = pd.DataFrame(outliers_value, columns=['columns', 'fence_low', 'cnt_low', 'fence_high', 'cnt_high'])
    return df_outliers_value


# ## Functions for cross val score methods

# In[20]:


def cvs_scores_to_df(clf, X, y, scoring, threshold, folds, standarizer=StandardScaler()):
    """ 
    This function takes:
    < clf: dict which contains names of the model as key and model as value,
    < X,
    < y,
    < scoring: which indicates methods for scoring the models through process of cross validation,
    < threshold: which indicate max different within models to accept them as not offerfitting,
    < folds: have many folds of data we want,
    < standarizer: takes one of the function which standarize data. Default: StandardScaler.
    And return dataframe which conteins name of models, mean, min, max value from cross validation each of the given models.
    """
    scores = []
    
    for name, model in clf.items(): 
        pipe = make_pipeline(standarizer, model)
        cvs_score = cross_val_score(pipe, X, y, cv=folds, scoring=scoring)
        if max(cvs_score) - min(cvs_score) > threshold:
            overfitting = 'Yes'
        else:
            overfitting = 'No' 
        scores.append({
            'model': name,
            'mean': cvs_score.mean(),
            'min': (min(cvs_score)),
            'max': (max(cvs_score)),
            'overfitting': overfitting,
            'diff': round((max(cvs_score) - min(cvs_score)),2)
        })
    
    df_cv_scores = pd.DataFrame(scores, columns=['model', 'mean', 'min', 'max', 'overfitting', 'diff'])
    print(f"This function made dataframe with scores from different models through cross validation using {scoring} to score models.")
    return df_cv_scores


# ## Functions to create metrics for models

# In[21]:


def create_measures(y,y_pred): 
    """ 
    This function takes:
    < y 
    < y_pred
    And return DataFrame with 'AUC' and gini metrics for train and test dataset. 
    """
    score_test = roc_auc_score(y, y_pred)
    Gini_index = 2*score_test - 1
    
    d = {'AUC': [round(score_test,4)], 'GINI': [round(Gini_index,4)]}
    d = pd.DataFrame.from_dict(d)
    
    return d

def calculating_metrics(X_train, X_test, y_train, y_test, model):
    """ 
    This function takes:
    < X_train,
    < X_test,
    < y_train,
    < y_test,
    < model
    And using predict_proba function to predict the class probabilities for train and test data \
    return their measures in Dataframe. 
    """
    train = create_measures(y_train,model.predict_proba(X_train)[:, 1])
    test = create_measures(y_test,model.predict_proba(X_test)[:, 1])
    
    measures =  pd.concat([train,test]).set_index([pd.Index(['TRAIN', 'TEST'])]) 
    
    return measures


# In[22]:


def calculate_metrics(target, prediction, average='macro'):
    """ 
    This function takes:
    < target 
    < prediction
    < average - an option that allows the given metrics to be measured based on it
    And return metrics like: accuracy, precision, recall, f1, mislabeled, total 
    """
    accuracy = accuracy_score(target, prediction)
    precision = precision_score(target, prediction, average=average)
    recall = recall_score(target, prediction, average=average)
    f1 = f1_score(target, prediction, average=average)
    mislabeled = (target != prediction).sum()
    total = len(target)
    return accuracy, precision, recall, f1, mislabeled, total 


# In[23]:


def print_report_for_classification(model, X_train, X_test, y_train, y_test, X, y, scoring='f1', target_names: list=None):
    """ 
    This function takes:
    < model - it can be also pipeline
    < X, y
    < scoring - to calculate metrics for model (default: 'f1')
    < target_names - how to rename output class for prediction variable
    And print all metrics for given model based on define scoring.
    """
    
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    crossval = cross_val_score(model, X, y, scoring=scoring)
    accuracy, precision, recall, f1, mislabeled, total =  calculate_metrics(y_test, predict, average='macro')
    confusion_m = confusion_matrix(y_test, predict)
    measures = calculating_metrics(X_train, X_test, y_train, y_test, model)
    
    print(measures)
    print("\n ------------------------------------------------------\n")
    print('Training set score: ' + str(model.score(X_train,y_train)))
    print('Test set score: ' + str(model.score(X_test,y_test)))
    print("\n ------------------------------------------------------")
    print("Classification_report: \n")
    print(classification_report(y_test, predict, target_names = target_names))
    print("\nXG Boosting cross-validation scores: {}".format(crossval))
    print("XG Boosting cross-validation mean score: {:.2f}".format(crossval.mean()))
    print("\n ------------------------------------------------------\n")
    print("Metrics:")
    print(f"Accuracy: {round(accuracy,2)}")
    print(f"Precision: {round(precision,2)}")
    print(f"Recall: {round(recall,2)}")
    print(f"f1: {round(f1,2)}")
    print(f"Mislabeled: {mislabeled}. It's: {round((mislabeled/total)*100, 2)}% of test data")
    print(f"Total: {total}")
    print("\n ------------------------------------------------------\n")
    print("Confusion matrix:\n{}".format(confusion_m))

