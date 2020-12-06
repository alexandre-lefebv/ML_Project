
# -*- coding: utf-8 -*-

"""tools.py: This file contain the functions used for the developpement project
in machine learning."""

__author__ = "Guillaume Ghienne, Aleandre Lefebvre"
__date__ = "December 2020"
__license__ = "Libre"
__version__ = "1.0"
__maintainer__ = "Not maintened"
__email__ = ["guillaume.ghienne@imt-atlantique.net",
            "alexandre.lefebvre@imt-atlantique.net"]
__status__ = "Complet"



import numpy  as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics


# -------------------------- Import the dataset -------------------------- #


# Guillaume
def read_data(fname):
    """ Read a dataset for a ML project.

    The dataset must be in a text format with ',' characters between the
    columns. Data type of columns must be int, float or string. Columns name
    must be provided.
    Usually using ' ' and '\t' in the file is not recommanded as it uselessly
    increase the size of the file and this function will ignore extra ' ' and
    '\t'.
    The missing value must be represented by '' (no characters between two ',')
    and not by '?', 'Nan' or any words of this kind since they would be read as
    string (and thus interpreted as non-missing data). If the file doesn't
    satisfy this rule, it must be modified before beeing read by this function.

    Parameters:
        fname (str): Absolute or relative path to the file containing the data.

    Returns:
        df (pandas.core.frame.DataFrame) : The raw dataframe.
    """

    df = pd.read_csv(fname,sep='\s*\t*[,]\s*\t*',engine='python')
    return df


# -------------------------- Clean the dataset -------------------------- #


# Alexandre
def split_df_XY(df,class_index=None):
    """ Split the data frame into raws of features and their class.
    
    Parameters:
        df (pandas.core.frame.DataFrame) : The data set to be split.
        class_index (str or None) : The column to be used as the class index.
            The default column used for the class index is the last one.
    
    Returns:
        X_res (ndarray[nb_sample,nb_features]) : Raws of features representing
            the samples.
        Y_res (ndarray[nb_samples]) : The class index for each samples.
    """
    if class_index == None:
        class_index = df.columns[-1]
    X = df.drop(columns=class_index).to_numpy()
    Y = df[class_index].to_numpy()

    return X,Y


# Alexandre
def rm_raw_without_class_id(X,Y):
    """ Remove the samples with missing class index.

    A missing value is expected to be 

    Parameters:
        X (ndarray[nb_sample,nb_features]) : The samples set to be cleaned.
        Y (ndarray[nb_samples]) : The class index for each samples.

    Returns:
        X_res (ndarray[nb_sample,nb_features]) : The cleaned samples set.
        Y_res (ndarray[nb_samples]) : The cleaned class index for each samples.
    """
    Y_nan = pd.Series(Y).notna()
    X_res = X[Y_nan]
    Y_res = Y[Y_nan]

    return X_res,Y_res


# Alexandre
def replace_string_by_int(X):
    """ Replace binary string features by binary int 0 or 1

    Parameters:
        X (ndarray[nb_sample,nb_features]) : The samples set to be processed.

    Returns:
        X_res (ndarray[nb_sample,nb_features]) : The processed samples set.
    """
    
    def try_float_convertion(x):
        try:
            return np.array(x,dtype=float)
        except:
            return x
    
    if len(np.shape(X)) == 1:
        reshaped=True
        X=X[:,np.newaxis]
    else:
        reshaped=False

    n_samp,n_feat = np.shape(X)
    X_res = np.zeros((n_samp,n_feat),dtype=float)
    
    for col in range(n_feat):
        X_feat = try_float_convertion(X[:,col])
        
        # Data type of the feature is float
        if X_feat.dtype==float:
            X_res[:,col] = np.array(X_feat)
            continue

        # Data type is not float so str is expected
        X_feat_unique = pd.Series(X_feat).dropna().unique()
        for k in range(len(X_feat_unique)):
            feat_value = X_feat_unique[k]
            X_feat[X_feat==feat_value] = k
        X_res[:,col] = np.array(X_feat)
    
    if reshaped:
        X_res = X_res[:,0]  
    return X_res


# Guillaume
def fill_missing_values(X,Y,fill_missing_with='median'):
    """Replace NaN (missing values) by a numerical value in the set of samples.

    For each features, apply method over non-NaN values for each of the two
    class and replace Nan values by the result. In case of class with only Nan,
    replace it by the method over all the feature.

    Parameters:
        X (numpy.ndarray) : The samples set.
        Y (numpy.ndarray) : The class index for each samples.
        fill_missing_with (str) : The method to use to fill the missing datas
            - 'avg' to use the mean of samples of the same class
            - 'med' (default) to use the median.

    Returns:
        X (numpy.ndarray) : The filled samples set.

    """
    m=np.shape(X)[1]

    for j in range(m):

        if fill_missing_with=='median':
            # replace NaN of each class by the median of non-NaN of this class
            X[Y==0,j]=np.nan_to_num(X[Y==0,j],nan=np.nanmedian(X[Y==0,j]))
            X[Y==1,j]=np.nan_to_num(X[Y==1,j],nan=np.nanmedian(X[Y==1,j]))

            # replace NaN of NaN-only class by the median of the feature
            X[:,j]=np.nan_to_num(X[:,j],nan=np.nanmedian(X[:,j]))
        elif fill_missing_with=='average':

            # replace NaN of each class by the mean of non-NaN of this class
            X[Y==0,j]=np.nan_to_num(X[Y==0,j],nan=np.nanmean(X[Y==0,j]))
            X[Y==1,j]=np.nan_to_num(X[Y==1,j],nan=np.nanmean(X[Y==1,j]))

            # replace NaN of NaN-only class by the mean of the feature
            X[:,j]=np.nan_to_num(X[:,j],nan=np.nanmean(X[:,j]))

    return X


# Guillaume
def normalize_data(X):
    """Center and normalize the set of samples.

    Parameters:
        X (numpy.ndarray) : The samples set.

    Returns:
        X (numpy.ndarray) : The normalalized samples set.

    """
    X=X-np.mean(X,axis=0)
    norms=np.linalg.norm(X, ord=2, axis=0)
    norms[norms==0]=1 # to avoid division by 0

    return X/norms # normalized


# Alexandre
def clean_data(df,class_index=None,fill_missing_with='median'):
    """ Wrapper for applying all the preprocessing funtions in one call.

    Parameters:
        df (pandas.core.frame.DataFrame) : The dataset to be preprocessed.
        class_index (str or None) : The column to be used as the class index.
            The default column used for the class index is the last one.
        fill_missing_with (str) : The method to use to fill the missing (nan)
            datas. Allowed values are 'average' and 'median'.

    Returns:
        X (ndarray[nb_sample,nb_features]) : Raws of features
            representing the samples.
        Y (ndarray[nb_samples]) : The class index for each samples.

    """
    X,Y = split_df_XY(df,class_index=class_index)
    X,Y = rm_raw_without_class_id(X,Y)
    X = replace_string_by_int(X)
    Y = replace_string_by_int(Y)
    print()
    X = fill_missing_values(X,Y,fill_missing_with=fill_missing_with)

    return X,Y



# --------------------------  Split the dataset -------------------------- #


# Alexandre
def split_dataset(X, Y,test_size=0.25,cv_test_size=0.1):
    """ Split data between training and test set, define the cross-validation.

    Parameters:
        X (ndarray[nb_sample,nb_features]) : Raws of features
            representing the samples.
        Y (ndarray[nb_samples]) : The class index for each samples.
        test_size (float or int, default=0.25): If float, should be between 0.0
            and 1.0 and represent the proportion of the dataset to include in
            the test split. If int, represents the absolute number of test
            samples.
        cv_test_size (float or int, default=0.1): If float, should be between
            0.0 and 1.0 and represent the proportion of the training dataset to
            include in the test split for the cross-validation procedure. If
            int, represents the absolute number of test samples.
        n_splits (int, default=10): Number of re-shuffling and splitting
            iterations for the cross-validation procedure.

    Returns:
        x_train (ndarray[nb_train_sample,nb_features]): Raws of features
            representing the train samples.
        x_test  (ndarray[nb_test_sample ,nb_features]): Raws of features
            representing the test samples.
        y_train (ndarray[nb_train_sample]): The class index of train samples.
        y_test  (ndarray[nb_test_sample ]): The class index of test samples.
        cvp (sklearn.model_selection._split.ShuffleSplit): The cross-validation
            procedure.

    """

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    cvp = ShuffleSplit(test_size=test_size,n_splits=1000)

    return x_train, x_test, y_train, y_test, cvp

# --------------------------  Train the model -------------------------- #


# Guillaume
def select_features(X,select_features):
    """Select some features to keep and eventually adjust the set of samples.

    Parameters:
        X (numpy.ndarray) : The samples set.
        select_features (int or list of int): Either a list of index of features
            to keep or an int as number of features to keep using ACP.

    Returns:
        X (numpy.ndarray) : Keeped (and adjusted) features.

    """
    if type(select_features)==list:
        X=X[:,select_features]
        return X
    elif type(select_features)==int:
        acp=PCA(select_features)
        X=acp.fit_transform(X)
        return X,

#--------------------------Model--------------------------------------------------#

#Guillaume
def classifier_decision_tree(x_train,y_train,cvp, depth=None,max_depth=10,verbose=False):
    """Return a decision_tree classifier.

    Parameters:
        x_train(ndarray) : The samples of the training set.
        y_train(ndarray) : The class indexes of training samples.
        cvp(sklearn.model_selection._split.ShuffleSplit): The cross-validation
            procedure.
        depth(int): the desired depth of the tree (optimal if None)
        max_depth(int): the maximal depth to search the optimal depth
        verbose(bool): Boolen to disp some informations

    Returns:
        classifier (function): The trained classifier.
            |  Parameters:
            |    sample (ndarray): A samples set.
            |
            |  Returns:
            |    predicted_class (ndarray):

    """
    if depth==None:
        depths = np.linspace(1, max_depth, max_depth)
        tab_RMSE_tree = np.zeros(max_depth)
        for i in range(n_depths):
            class_tree = DecisionTreeClassifier(max_depth=depths[i])
            tab_RMSE_tree[i] = np.median(np.sqrt(-cross_val_score(class_tree,
                x_train, y_train,scoring='neg_mean_squared_error', cv=cvp)))
        depth=depths[np.argmin(tab_RMSE_tree)]
    tree = DecisionTreeClassifier(max_depth=depth)
    tree.fit(x_train, y_train)
    if verbose==True:
        plt.plot(depths,tab_RMSE_tree)
        plt.show()
    def classifier(x_test):
        return tree.predict(x_test)
    return tree_classifier

#Guillaume
def classifier_random_forest(x_train, y_train,cvp, n_trees=100,depth=None, max_depth=10,verbose=False):
    """Return a random forest classifier.

    Parameters:
        x_train(ndarray): The samples of the training set.
        y_train(ndarray) : The class indexes of training samples.
        cvp(sklearn.model_selection._split.ShuffleSplit): The cross-validation
            procedure.
        depth(int): the desired depth of the trees (optimal if None)
        max_depth(int): the maximal depth to search the optimal depth
        verbose(bool): Boolean to disp some informations

    Returns:
        classifier (function): The trained classifier.
            |  Parameters:
            |    sample (ndarray): A samples set.
            |
            |  Returns:
            |    predicted_class (ndarray):

    """
    if depth==None:
        depths = np.linspace(1, max_depth, max_depth)
        tab_RMSE_tree = np.zeros(max_depth)
        for i in range(n_depths):
            class_tree = DecisionTreeClassifier(max_depth=depths[i])
            tab_RMSE_tree[i] = np.median(np.sqrt(-cross_val_score(class_tree, x_train, y_train,scoring='neg_mean_squared_error', cv=cvp)))
        depth=depths[np.argmin(tab_RMSE_tree)]
    forest = RandomForestClassifier(n_estimators=n_trees,max_depth=depth)
    forest.fit(x_train, y_train)
    if verbose==True:
        plt.plot(depths,tab_RMSE_tree)
        plt.show()
    def classifier(x_test):
        return forest.predict(x_test)
    return forest_classifier

#Guillaume
def classifier_ada_boost(x_train, y_train):
    """Return an ada boost classifier.

    Parameters:
        x_train(ndarray): The samples of the training set.
        y_train(ndarray) : The class indexes of training samples.

    Returns:
        classifier (function): The trained classifier.
            |  Parameters:
            |    sample (ndarray): A samples set.
            |
            |  Returns:
            |    predicted_class (ndarray):

    """
    class_boost = AdaBoostClassifier()
    class_boost.fit(x_train, y_train)
    def ada_boost_classifier(x_test):
        return class_boost.predict(x_test)
    return ada_boost_classifier
