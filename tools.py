
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

# Alexandre
def spit_df_XY(df,class_index=None):
    """ Split the data frame into raws of features and their class.
    
    Parameters:
        df (pandas.core.frame.DataFrame) : The data set to be split.
        class_index (string or None) : The column to be used as the class index.
            The default column used for the class index is the last one.
    
    Returns:
        X_res (ndarray[nb_sample,nb_features]) : Raws of features representing the
            samples.
        Y_res (ndarray[nb_samples]) : The class index for each samples.
    """
    if class_index == None:
        class_index = df.columns[-1]
    X = df.drop(columns=class_index).to_numpy()
    Y = df[class_index].to_numpy()

    return X,Y

# Guillaume
def fill_missing_values(X,Y,method='med'):
    """Replace NaN (missing values) by a numerical value in the set of samples.

    For each features, apply method over non-NaN values for each of the two
    class and replace Nan values by the result. In case of class with only Nan,
    replace it by the method over all the feature.

    Parameters:
        X (numpy.ndarray) : The samples set.
        Y (numpy.ndarray) : The class index for each samples.
        method (string)   : The method to use to fill the missing datas
                            - 'avg' to use the mean of samples of the same class
                            - 'med' (default) to use the median.

    Returns:
        X (numpy.ndarray) : The filled samples set.

    """
    m=np.shape(X)[1]

    for j in range(m):

        if method=='med':

            # replace NaN of each class by the median of non-NaN of this class
            X[Y==0,j]=np.nan_to_num(X[Y==0,j],nan=np.nanmedian(X[Y==0,j]))
            X[Y==1,j]=np.nan_to_num(X[Y==1,j],nan=np.nanmedian(X[Y==1,j]))

            # replace NaN of NaN-only class by the median of the feature
            X[:,j]=np.nan_to_num(X[:,j],nan=np.nanmedian(X[:,j]))

        elif method=='avg':

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
