import pandas as pd

import math

"""

Instructions:
    
    1.  Call get_train_test(path). Takes the path to where the county files as an argument. 
        County files need to be names as listed below in the setup function.

        get_train_test() returns the following:

            1.  train:  a list of five strata consisting on the county data sets from 2009-2017 as dataframes
            2.  test:   the 2018 county dataframe  

    thats really it! 

"""

def setup_data(path=''):
    
    import sys
    sys.path.insert(1, path)
    
    county2009 = pd.read_csv('county2009_final.csv')
    county2010 = pd.read_csv('county2010_final.csv')
    county2011 = pd.read_csv('county2011_final.csv')
    county2012 = pd.read_csv('county2012_final.csv')
    county2013 = pd.read_csv('county2013_final.csv')
    county2014 = pd.read_csv('county2014_final.csv')
    county2015 = pd.read_csv('county2015_final.csv')
    county2016 = pd.read_csv('county2016_final.csv')
    county2017 = pd.read_csv('county2017_final.csv')
    county2018 = pd.read_csv('county2018_final.csv')
    
    counties = [county2009, county2010, county2011, county2012, county2013, 
                county2014, county2015, county2016, county2017, county2018]
    
    return counties

def build_county(dataframes):
    sample = dataframes[0]
    train = pd.DataFrame(columns=sample.columns).drop(['Unnamed: 0'] ,axis=1)
    for df in dataframes:
        train = train.append(df, ignore_index=True, sort=False)
    return train

def float_check(dataframes):
    for df in dataframes:
        for col in df.columns:
            for i in range(len(df)):
                df.loc[i, col] = float(df.loc[i, col])

def stratify(dataframe):
    
    counties = dataframe.copy()
    total = len(dataframe)
    partition = math.floor(total/5)
    
    strata1 = counties.loc[0           : 1*partition, :].reset_index().drop(['index'], axis=1)
    strata2 = counties.loc[1*partition : 2*partition, :].reset_index().drop(['index'], axis=1)
    strata3 = counties.loc[2*partition : 3*partition, :].reset_index().drop(['index'], axis=1)
    strata4 = counties.loc[3*partition : 4*partition, :].reset_index().drop(['index'], axis=1)
    strata5 = counties.loc[4*partition:, :].reset_index().drop(['index'], axis=1)

    strata1['Strata'] = [1] * len(strata1)
    strata2['Strata'] = [2] * len(strata2)
    strata3['Strata'] = [3] * len(strata3)
    strata4['Strata'] = [4] * len(strata4)
    strata5['Strata'] = [5] * len(strata5)


    strata = [strata1, strata2, strata3, strata4, strata5]
    
    for i in range(len(strata)):
        if 'Unnamed: 0' in strata[i].columns:
            strata[i] = strata[i].drop(['Unnamed: 0'], axis=1)
        if 'Unnamed: 0.1' in strata[i].columns:
            strata[i] = strata[i].drop(['Unnamed: 0.1'], axis=1)
            
    float_check(strata)
    
    return strata

def get_train_test(path):
    counties = setup_data(path)
    
    # Separate 2009-2017 and 2018
    train = counties[:-1]
    test = counties[-1].drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    
    # Process training
    train = build_county(train)
    train = stratify(train)
    
    return train, test