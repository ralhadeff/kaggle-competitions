"""
Feature Tester
this is a work in progress

TODO:
    printout feature to column number (for X) list
    add transformed features (log, 1/x, x**2 etc)
    improve the scaling
    add restore feature
    fix the problem with missing dummies between data and the prediction input
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

class FeatureTester():
        
    def __init__(self, data, test_size=0.2, random_seed='random', cv=5, precision=4):
        """
        Initialize the feature tester, provide data as a pandas dataframe, including the y column
        """
        self.data = data
        # features to use, and type of feature
        self.features = []
        # features not to use
        self.dont_use = []
        # list of estimators and their names
        self.estimators = []
        self.names = []
        # random seem to train_test_split
        if (random_seed=='random'):
            self.random_seed = np.random.randint(0,1000)
        else:
            self.random_seed = random_seed
        # test size (in train_test_split)
        self.test_size = test_size
        # number of CV iterations
        self.cv = cv
        # number of decimal points to show
        self.precision = precision
        # previous scores for comparison
        self.previous = None
        # feature scaler
        self.scaler = StandardScaler()
        
    def set_y(self,column):
        """Mark the y column (by name)"""
        self.y = column
    
    def add_feature(self,feature,dtype='numerical',arg='auto'):
        '''
        Add feature to the regression
            default is numerical
            
            ordinal can be specified in dtype, if so, a conversion dictionary might be required
            
            categorical can be specified in dtype, if so, feature will be converted to dummy columns
                by default omitting NaN or the least frequest label
                (otherwise, dummy to omit can be specified by name)
        '''
        if (dtype=='numerical'):
            # check that there are no missing values
            if (self.data[feature].isnull().sum()>0):
                if (arg is None):
                    raise ValueError('This feature has missing values, add missing values manually\
                                     or specify a default value')
                else:
                    # arg is the default value
                    self.features.append((feature,dtype,arg))
            else:
                # add as is
                self.features.append((feature,dtype))
        elif(dtype=='ordinal'):
            # add converted with optional dictionary
            self.features.append((feature,dtype,arg))
        elif(dtype=='categorical'):
            # add converted to dummies
            self.features.append((feature,dtype,arg))  
    
    def print_features(self,type_filter=None,return_tuple=False):
        """Return list of features in a concise representation"""
        if (type_filter) == None:
            return [feature[0] for feature in self.features]
        elif (return_tuple):
            return [feature for feature in self.features if feature[1]==type_filter]
        else:
            return [feature[0] for feature in self.features if feature[1]==type_filter]
    
    def find_feature_index(self,feature):
        """Find the index of a feature by name"""
        i = 0
        for f in self.features:
            if (f[0] == feature):
                break
            i+=1
        return i

    def modify_feature(self,feature,dtype,arg=None):
        """Modify a feature and score it"""
        i = self.find_feature_index(feature)
        if (arg is None):
            if (dtype=='numerical'):
                if (arg is None):
                    if (self.data[feature].isnull().sum()>0):
                        raise ValueError('This feature has missing values, add missing values manually\
                                     or specify a default value')
                    self.features[i] = (feature,dtype)
                else:
                    self.features[i] = (feature,dtype,arg)
            else:
                self.features[i] = (feature,dtype,'auto')
        else:
            self.features[i] = (feature,dtype,arg)
        return self.score_feature(feature)
    
    def remove_feature(self,feature):
        """Move the last added feature(s) to the No pile
        """
        i = self.find_feature_index(feature)
        removed = self.features.pop(i)
        self.dont_use.append(removed[0])
        # run fit again to correct self.previous
        self.fit()
    
    def add_estimator(self,estimator,name):
        """Add a named estimator to perform regression with"""
        self.estimators.append(estimator)
        self.names.append(name)
    
    def fit(self,compare=True,skip=None):
        """
        Fit the selected features and return scores
            compare will also print the different in the score from the previous fit
            skip allows user to skip a single feature (for recomparing etc)
        """
        X, y = self.build_data(skip)
        # fit and score
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=self.test_size, random_state=self.random_seed)
        scores = []
        for estimator in self.estimators:
            scores.append( cross_val_score(estimator,X_train,y_train,cv=5).mean() )
        if (compare and self.previous is not None):
            diff = np.asarray(scores) - np.asarray(self.previous)
            self.previous = scores
            scores_diff = np.round(np.concatenate((scores,diff)),self.precision)
            return np.concatenate((self.names,scores_diff)).reshape(3,len(scores)).T
        self.previous = scores
        return np.concatenate((self.names,np.round(scores,self.precision))).reshape(2,len(scores)).T
    
    def score_test_set(self,skip=None):
        X, y = self.build_data(skip)
        # fit and score
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=self.test_size, random_state=self.random_seed)
        scores = []
        for estimator in self.estimators:
            estimator.fit(X_train,y_train)
            scores.append( estimator.score(X_test,y_test) )
        return np.concatenate((self.names,np.round(scores,self.precision))).reshape(2,len(scores)).T

    def predict(self,df):
        X = self.build_data(skip=None,data=df)
        # fit and score
        predictions = []
        for estimator in self.estimators:
            predictions.append( estimator.predict(X) )
        return predictions

    def score_feature(self,feature):
        """Check the difference is scoring with and without a feature"""
        self.fit(skip=feature)
        return self.fit(True)
    
    def score_all_features(self):
        """Score all features, and return a report"""
        # here I will use reverse of the score, to save time
        scores = self.fit()
        names = scores[:,0]
        scores = scores[:,1].astype(float)
        for feature in self.features:
            print(feature[0])
            current = self.fit(False,skip=feature[0])[:,1].astype(float)
            diff = np.round(scores-current,self.precision)
            print(np.concatenate((names,diff))
                           .reshape(2,len(self.estimators)).T)
    
    def sanity_check(self):
        """Make sure there is nothing that makes not sense or that raises red flags"""
        features = self.print_features()
        duplicates = set([f for f in features if features.count(f) > 1])
        if (len(duplicates)>0):
            print('Warning, you have duplicates')
            print(duplicates)
        yes_and_no = set([f for f in features if f in self.dont_use])
        if (len(yes_and_no)>0):
            print('Warning, you have features both in use and in dont_use')
            print(yes_and_no)
        missing_features = []
        for f in self.data.columns:
            if (f not in features and f not in self.dont_use and f!=self.y):
                missing_features.append(f)
        if (len(missing_features)>0):
            print('Note, there are features in the dataset that were not tested')
            print(missing_features)
        
    def build_data(self,skip=None,data=None):
        # keep track of column names
        col_names = []
        # set data
        if (data is None):
            data = self.data
        # add numerical features
        numerical = self.print_features('numerical',True)
        X = np.array([])
        # for correct reshape
        skip_num = False
        for feature in numerical:
            if (feature[0]!=skip):
                col_names.append(feature[0])
                if (len(feature)==2):
                    x = data[feature[0]]
                else:
                    if (feature[2]=='mean'):
                        fill = data[feature[0]].mean()
                    else:
                        fill = feature[2]
                    x = data[feature[0]].fillna(fill)
                X = np.concatenate((X,x))
            else:
                skip_num = True
        if (skip_num):
            X = X.reshape(len(numerical)-1,len(data)).T
        else:
            X = X.reshape(len(numerical),len(data)).T
        # add ordinal features
        ordinal = self.print_features('ordinal',True)
        for feature in ordinal:
            if (feature[0]!=skip):
                col_names.append(feature[0])
                if (feature[2] == 'auto'):
                    X = np.hstack((X,data[feature[0]][:,None]))
                else:
                    column = data.fillna('nan')[feature[0]].map(feature[2])
                    X = np.hstack((X,column[:,None]))
        # scale before dummy features
        if (data is self.data):
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        # add categorical features
        categorical = self.print_features('categorical',True)
        for feature in categorical:
            if (feature[0]!=skip):
                if (feature[2]=='auto'):
                    dummies = make_dummy(data,feature[0])
                else:
                    if (type(feature[2]) is str):
                        dummies = make_dummy(data,feature[0],omit=feature[2])
                    else:
                        dummies = make_dummy(data,feature[0],binning=feature[2])
                col_names.append(dummies.columns)
                X = np.hstack((X,dummies))
        self.col_names = col_names
        # return
        if (data is self.data):
            # y
            y = data[self.y]
            return X, y 
        else:
            return X

def get_dict(df,column):
    """
    Return an empty dictionary for user to fill
        this should make ordinal and categorical binning easier
    """
    return dict([(f,0) for f in df[column].unique()])

def get_list(df,column):
    return df[column].unique()

def observe_feature(df,column,y):
    n = df[column].nunique()
    if (n>=20):
        sns.scatterplot(x=column,y=y,data=df)
    if (n<20):
        print(df[column].value_counts())
        sns.boxplot(x=column,y=y,data=df.fillna('nan'))
        plt.show()
            
def make_dummy(df, column,omit=None,binning=None):
    """
    Take a column name and make a dummy DataFrame from it
        If there are Null values, do not omit one (null will be the 0's)
        otherwise, remove the one will the least counts,
        return the dataFrame
            optional - omit the specified label
        Another option is binning - provided by a list of lists, where each inner list is one bin
    """
    nulls = df[column].isnull().sum()
    rarest = df[column].value_counts().index[-1]
    dummies = pd.get_dummies(df[column])
    if (binning is not None):
        for ls in binning:
            name = ''
            for col in ls:
                name += str(col) + '_'
            dummies[name] = dummies[ls].sum(axis=1)
            dummies.drop(ls,axis=1,inplace=True)
    if (nulls>0):
        if (omit is not None):
            raise ValueError('NaN values are present, omit cannot be specified')
        return dummies
    elif (omit is None):
        return dummies.drop(rarest,axis=1)
    else:
        return dummmies.drop(omit,axis=1)
