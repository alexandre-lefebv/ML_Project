
import numpy  as np
import pandas as pd
from sklearn.decomposition import PCA

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
        return X


