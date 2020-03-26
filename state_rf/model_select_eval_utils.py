#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contains utility functions for the processes of model selection and 
model evaluation.

@author: joejohnson
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import time
import csv

def load_data(source_directory, source_file, input_features, labels, 
              desc_features = None):
    # loads data from file.
    # 
    # parameters:
    #
    # source_directory:       name of the source file directory
    # source_file:            name of source file
    # input_features:         list of names of input features in file
    # labels:                 list of labels/targets in file
    # desc_features:          list of descriptive fields for the records - 
    #                         these might include key field(s) or description
    #                         fields
    #
    # returns: 
    # dictionary containing the following:
    # X:                      data frame of input features from file
    # y:                      1d array of labels from file
    # desc:                   data frame of description fields
    
    # os.chdir(source_directory)
    df = pd.read_csv(source_directory + '/' + source_file)
    
    data = {}

    X = df[input_features]
    y = df[labels].copy()
    y = np.ravel(y, order = "F") # flatten y so GridSearchCV does not complain

    data['X'] = X
    data['y'] = y
    
    if desc_features != None:
        desc = df[desc_features]
        data['desc'] = desc
    
    return data


def select_model_one_fold(source_directory, source_train_fold_file, estimator, 
                          input_features, labels, param_grid):
    # performs hyperparameter tuning to determine the optimal model for 
    # one cv fold.  That is, this function finds 
    # the optimal set of hyperparameters for an estimator (provided as input) 
    # for a cv train/test split (where the train
    # test splits are stored in separate csv files.)
    #
    # parameters:
    #
    # source_directory:         name of the source file directory for dataset
    # source_train_fold_file:   name of source train file for the fold
    # estimator:                estimator object (RandomForestRegressor, 
    #                           KNeighborsRegressor, etc.)
    # input_features            list of names of input features in the dataset 
    #                           file for the model 
    # labels                    list of target labels in the dataset file
    #                           for the model (usually this list includes only 
    #                           one field)
    # param_grid                a dictionary whose keys/values are model 
    #                           parameter names, and whose values are 
    #                           lists of potential assignments for those
    #                           parameter names 
    #
    # returns:
    # 
    # a dictionary containing the following key/value pairs:
    # best_params:            dictionary containing
    #                         the parms for highest performing model
    # best_mse:               mse for the best_params model 
    # run_time:               elapsed time for the grid search
    
    data = load_data(source_directory, source_train_fold_file, 
                                 input_features, labels)
    X_train = data['X']
    y_train = data['y']
    
    start_time = time.process_time()
    grid_search = GridSearchCV(estimator,param_grid, cv=3,
                           scoring='neg_mean_squared_error')
    grid_search.fit(X_train,y_train)
    
    end_time = time.process_time()
    run_time = end_time - start_time
    
    best_model = {}
    best_model['run_time'] = run_time
    best_model['best_mse'] = grid_search.best_score_
    best_model['best_params'] = grid_search.best_params_
    
    return best_model


def select_model_all_folds(source_directory, estimator, input_features, 
                           labels, param_grid):
    # performs hyperparameter tuning to determine the optimal model for 
    # across all 5-fold cv splits.  This is done by calling the 
    # select_model_one_fold() function for each of the 5 fold splits and 
    # returning the results for each fold.  This function does ** not **
    # select the best params from among the folds - the final selection 
    # of optimal parameters should be done manually after viewing these
    # results for the five folds.    
    #
    # parameters:
    #
    # source_directory:       name of the source file directory for dataset
    # estimator:              estimator object (RandomForestRegressor, 
    #                         KNeighborsRegressor, etc.)
    # input_features          list of names of input features in the dataset 
    #                         file for the model 
    # labels                  list of target labels in the dataset file
    #                         for the model (usually this list includes only 
    #                         one field)
    # param_grid              a dictionary whose keys are model parameter names
    #                         and whose values are lists of potential 
    #                         assignments for those parameter names 
    #
    # returns:
    # 
    # a dictionary containing the best model params for each of the 5 folds:
    # best_model_1:           a dictionary containing the results for fold 1
    # best_model_2:           a dictionary containing the results for fold 2
    # best_model_3:           a dictionary containing the results for fold 3
    # best_model_4:           a dictionary containing the results for fold 4
    # best_model_5:           a dictionary containing the results for fold 5

    best_models = {}
    
    for i in range(5):
        best_models['best_model_' + str(i+1)] = \
            select_model_one_fold(source_directory, 
                                  'strat_fold_' + str(i + 1) + '_train.csv', 
                                  estimator, 
                                  input_features, 
                                  labels, 
                                  param_grid)
    return best_models
        
    
def evaluate_model(model, source_directory, input_features, labels, 
                   desc_features):
    # evaluates a model - runs the model on each of 
    # 5-fold CV train/test splits stored in .csv files and then
    # returns the average test mse.
    #
    # parameters:
    #
    # source_directory:         location of the train/test split csv files
    # input_features:           list of input features for the model
    # labels:                   list of output targets for the model
    # desc_features:            list of descriptive fields to retrieve from
    #                           file
    #
    # returns:
    # 
    # dictionary containing the following:
    # mean_mse_train:           mean mse on the training set
    # mean_mse_test:            mean mse on the test set
    # fold_mses_train:          list containing the training set mses for the 
    #                           5 folds
    # fold_mses_test:           list containing the mses test set mses for the
    #                           5 folds
    
    model_results = {}
    fold_mses_train = []
    fold_mses_test = []
    
    for i in range(5):
        train_file = 'strat_fold_' + str(i+1) + '_train.csv'
        test_file = 'strat_fold_' + str(i+1) + '_test.csv'
        
        data_train = load_data(source_directory, train_file, input_features, 
                              labels, desc_features)
        X_train = data_train['X']
        y_train = data_train['y']
        desc_train = data_train['desc']
        
        data_test = load_data(source_directory, test_file, input_features, 
                             labels, desc_features)
        X_test = data_test['X']
        y_test = data_test['y']
        desc_test = data_test['desc']

        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        record_detail(desc_train, X_train, y_train, y_pred_train, 
                      model_results, 'train', i + 1)
                     
        record_detail(desc_test, X_test, y_test, y_pred_test, 
                      model_results, 'test', i + 1)
        
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        
        fold_mses_train.append(mse_train)
        fold_mses_test.append(mse_test)
        
    mean_mse_train = np.mean(fold_mses_train)    
    mean_mse_test = np.mean(fold_mses_test)    
    
    model_results['mean_mse_train'] = mean_mse_train
    model_results['mean_mse_test'] = mean_mse_test
    model_results['fold_mses_train'] = fold_mses_train
    model_results['fold_mses_test'] = fold_mses_test

    return model_results

def evaluate_bagging_model(model, source_directory, input_features, labels, bags, seed):
    model_results = {}
    fold_mses_train = []
    fold_mses_test = []

    for i in range(5):
        print('Processing fold ' + str(i+1) + '...')
        train_file = 'strat_fold_' + str(i+1) + '_train.csv'
        test_file = 'strat_fold_' + str(i+1) + '_test.csv'

        data_train = load_data(source_directory, train_file, input_features, labels)
        X_train = data_train['X']
        y_train = data_train['y']

        data_test = load_data(source_directory, test_file, input_features, labels)
        X_test = data_test['X']
        y_test = data_test['y']

        y_pred_train = np.zeros(X_train.shape[0])
        y_pred_test = np.zeros(X_test.shape[0])

        for n in range(bags):
            model.set_params(random_state = seed + n)
            model.fit(X_train, y_train)
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            y_pred_train += train_preds
            y_pred_test += test_preds

        y_pred_train /= bags
        y_pred_test /= bags

        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        fold_mses_train.append(mse_train)
        fold_mses_test.append(mse_test)
        
        print('Fold ' + str(i+1) + ' completed.')

    mean_mse_train = np.mean(fold_mses_train)    
    mean_mse_test = np.mean(fold_mses_test)    

    model_results['mean_mse_train'] = mean_mse_train
    model_results['mean_mse_test'] = mean_mse_test
    model_results['fold_mses_train'] = fold_mses_train
    model_results['fold_mses_test'] = fold_mses_test
    
    return model_results



def record_detail(desc, X, y, y_pred, model_results, split, fold):
    # helper function for evaluate_model() function - not to be called from
    # outside of this module.
    #
    # combines input features X with y and y_pred, calculates se at the 
    # record level, and adds to the model_results dictionary.
    
    fold_detail_train = pd.concat([desc, X], axis = 1)
    fold_detail_train['y'] = y
    fold_detail_train['y_pred'] = y_pred
    diff = np.absolute(y - y_pred)
    se = np.square(diff)
    fold_detail_train['diff'] = diff
    fold_detail_train['se'] = se
    model_results['fold_' + str(fold) + '_detail_' + split] = fold_detail_train
    

def write_model_results_to_file(model_id, model_description, model_results, 
                                output_file, append):
    # formats data in a model results dictionary into as a list and writes 
    # the list to a file.
    #
    # parameters:
    # model_id:            a unique name to assign to the model/model results
    #                      as it should appear in a report, such as 'KNN_1_1'
    #                      or 'RF_2_1'
    # model_description:   descriptive comments for the nature of the model,
    #                      'knn model, k = 3, no high occlusives or elastomers'
    # model_results:       dictionary containin results from evalute_model()
    #                      function.
    # output_file:         name of the file to which to write the model 
    #                      results record
    # append:              boolean field set to True if model results record
    #                      is to be appended to an existing file or else false
    #                      if new file to be created
    
    # construct results list to be written to file.
    results_row = [model_id, model_description, model_results['mean_mse_test'], 
                   model_results['mean_mse_train']]

    for fold_mse_test in model_results['fold_mses_test']:
        results_row.append(fold_mse_test)

    for fold_mse_train in model_results['fold_mses_train']:
        results_row.append(fold_mse_train)
        
    # write the list to a file
    if append:
        file_mode = 'a'
    else:
        file_mode = 'w'
    
    with open(output_file, file_mode) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(results_row)
