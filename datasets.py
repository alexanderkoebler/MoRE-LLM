"""
Copyright 2024 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import numpy as np
from alibi.utils.data import Bunch

def fetch_adult(features_drop = None, return_X_y = False):
    if features_drop is None:
        features_drop = ["fnlwgt", "Education-Num"]

    raw_features = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital Status',
                    'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
                    'Hours per week', 'Country', 'Target']

    adult = fetch_openml("adult", version=2)
    raw_data = adult['data']
    raw_data['target'] = adult['target'] 
    raw_data.columns = raw_features

    labels = (raw_data['Target'] == '>50K').astype(int).values
    features_drop += ['Target']
    data = raw_data.drop(features_drop, axis=1)
    features = list(data.columns)

    education_map = {
        '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
        'Some-college': 'High School grad', 'Masters': 'Masters',
        'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
        'Assoc-voc': 'Associates'
    }
    occupation_map = {
        "Adm-clerical": "Admin", "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar"
    }
    country_map = {
        'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
        'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
        'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
        'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
        'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
        'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
        'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
        'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
        'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
        'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
        'United-States': 'United-States', 'Vietnam': 'SE-Asia'
    }
    married_map = {
        'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
        'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
    }
    mapping = {'Education': education_map, 'Occupation': occupation_map, 'Country': country_map,
               'Marital Status': married_map}
    
    for f in features:
        if data[f].dtype == 'category':
            data[f] = data[f].astype('O')
            data[f] = data[f].fillna('?')

    data_copy = data.copy()
    for f, f_map in mapping.items():
        data_tmp = data_copy[f].values
        for key, value in f_map.items():
            data_tmp[data_tmp == key] = value
        data[f] = data_tmp

    categorical_features = [f for f in features if data[f].dtype == 'O']
    category_map = {}
    for f in categorical_features:
        le = LabelEncoder()
        data_tmp = le.fit_transform(data[f].values)
        data[f] = data_tmp
        category_map[features.index(f)] = list(le.classes_)

    data = data.values.astype(np.int64)
    target_names = ['<=50K', '>50K']
    dataset_size = data.shape[0]
    idxs = np.random.choice(np.arange(0, dataset_size), 1000, replace=False)
    data = data[idxs]
    labels = labels[idxs]

    if return_X_y:
        return data, labels

    return Bunch(data=data, target=labels, feature_names=features, target_names=target_names, category_map=category_map)

def fetch_diabetes(features_drop = None, return_X_y = False):
    raw_features = ['Times Pregnant', 'Plasma Glucose Concentration', 'Diastolic Blood Pressure (mm Hg)', 'Triceps Skin Fold Thickness (mm)', '2-Hour Serum Insulin (mu U/ml)', 'Body Mass Index',
                    'Diabetes Pedigree Function', 'Age', 'Target']
    if features_drop is None:
        features_drop = []

    credit = fetch_openml("diabetes", version=1)
    raw_data = credit['data']
    raw_data['target'] = credit['target'] 
    raw_data.columns = raw_features

    labels = (raw_data['Target'] == 'tested_positive').astype(int).values
    features_drop += ['Target']
    data = raw_data.drop(features_drop, axis=1)
    features = list(data.columns)

    for f in features:
        if data[f].dtype == 'category':
            data[f] = data[f].astype('O')
            data[f] = data[f].fillna('?')

    categorical_features = [f for f in features if data[f].dtype == 'O']
    category_map = {}
    for f in categorical_features:
        le = LabelEncoder()
        data_tmp = le.fit_transform(data[f].values)
        data[f] = data_tmp
        category_map[features.index(f)] = list(le.classes_)

    data = data.values.astype(np.float32)
    target_names = ['tested_negative', 'tested_positive']

    if return_X_y:
        return data, labels

    return Bunch(data=data, target=labels, feature_names=features, target_names=target_names, category_map=category_map)

def fetch_german(features_drop = None, return_X_y = False):
    if features_drop is None:
        features_drop = []

    raw_features = ['Status of existing checking account', 'Duration in months', 'Credit history', 'Purpose of the credit', 'Credit amount', 'Status of savings account', \
    'Years of resent employment', 'Installment rate in percentage of disposable income', 'Personal status', 'Other debtors', 'Present residence since X years', 'Property', \
    'Age in years', 'Other installment plans', 'Housing', 'Number of existing credits', 'Job', 'Number of people being liable to provide maintenance for', 'Telephone', 'Foreign worker', 'Target']

    credit = fetch_openml("credit-g", version=1)
    raw_data = credit['data']
    raw_data['target'] = credit['target'] 
    raw_data.columns = raw_features

    labels = (raw_data['Target'] == 'bad').astype(int).values
    features_drop += ['Target']
    data = raw_data.drop(features_drop, axis=1)
    features = list(data.columns)

    for f in features:
        if data[f].dtype == 'category':
            data[f] = data[f].astype('O')
            data[f] = data[f].fillna('?')

    checking_map = {
        "0<=X<200": "0-200", "<0": "less then 0",
        ">=200": "more then 200", "no checking": "no checking"
    }
    savings_map = {
        '100<=X<500': '100-500', '500<=X<1000': '500-1000', '<100':
            'less then 100', '>=1000': 'more then 100', 'no known savings': 'no known savings'}
    employment = {
        '1<=X<4': '1-4', '4<=X<7': '4-7',
        '<1': 'less then 1', '>=7': 'more then 7'
    }
    mapping = {'Status of existing checking account': checking_map, 'Status of savings account': savings_map, 'Years of resent employment': employment}
    data_copy = data.copy()
    for f, f_map in mapping.items():
        data_tmp = data_copy[f].values
        for key, value in f_map.items():
            data_tmp[data_tmp == key] = value
        data[f] = data_tmp

    categorical_features = [f for f in features if data[f].dtype == 'O']
    category_map = {}
    for f in categorical_features:
        le = LabelEncoder()
        data_tmp = le.fit_transform(data[f].values)
        data[f] = data_tmp
        category_map[features.index(f)] = list(le.classes_)

    data = data.values.astype(np.int64)
    target_names = ['good', 'bad']

    if return_X_y:
        return data, labels

    return Bunch(data=data, target=labels, feature_names=features, target_names=target_names, category_map=category_map)
