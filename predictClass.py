import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,roc_curve,auc


#Load data into dataframes
probeA = pd.read_csv("../probeA.csv",header=0)
probeB = pd.read_csv("../probeB.csv",header=0)
classA = pd.read_csv("../classA.csv",header=0)

#Function that returns a dataframe consisting of all polynomial combinations of the features with degree less than or equal to the specified degree
def polynomial_feature_ord(X,n):
    poly = preprocessing.PolynomialFeatures(n)
    out = poly.fit_transform(X)
    feature_names = poly.get_feature_names(X.columns)

    X_new = pd.DataFrame(out,columns =feature_names)
    return X_new

#Function that switches columns for each row, from smallest to largest.
def swapColumns(df):
    df2=df.copy()
    for column in ["c","m","n","p"]:
        c1 = df2[[column+"1",column+"2",column+"3"]]
        c2 = c1.values
        c2.sort(axis=1)
        c2_df = pd.DataFrame(c2,columns=c1.columns)
        df2[c1.columns] = c2_df
    return df2

def KNN_with_kfold(X, y, neighbors, folds=10):
    kf = KFold(n_splits=folds, shuffle=False)
    kf.get_n_splits(X)
    total_AUC = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier = KNeighborsClassifier(n_neighbors = neighbors, weights='distance', p=2, metric='euclidean')
        classifier.fit(X_train, np.ravel(y_train))

        y_pred = classifier.predict_proba(X_test)[:,1]
        y_pred = pd.Series(y_pred)
        y_test = y_test.reset_index(drop=True)


        fpr,tpr,threshold = roc_curve(y_test,y_pred,pos_label=1)
        total_AUC += auc(fpr, tpr)

    #print("AVERAGE ACCURACY:", total_accuracy/folds)

    return total_AUC/folds

def KNN_plot(X, Y, max_k):
    # creating a list of K for KNN
    neighbors = list(range(1,max_k))

    # subsetting just the odd ones
    #neighbors = list(filter(lambda x: x % 2 != 0, myList))

    # empty list that will hold AUC CV scores
    AUC_cv_scores = []

    if len(X.shape) > 1:
        for k in neighbors:
            AUC_cv_scores.append(KNN_with_kfold(X, Y, k))

    else:
        for k in neighbors:
            AUC_cv_scores.append(KNN_with_kfold(X.reshape(-1,1), Y, k))

    # changing to misclassification error
    #MSE = [1 - x for x in AUC_cv_scores]

    # determining best k
    optimal_k = neighbors[AUC_cv_scores.index(max(AUC_cv_scores))]
    print("The optimal number of neighbors is " + str(optimal_k) + " with an AUC score of " + str(max(AUC_cv_scores)))

    # plot misclassification error vs k
    plt.plot(neighbors, AUC_cv_scores)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('AUC')
    plt.show()

    classifier = KNeighborsClassifier(n_neighbors = optimal_k, weights='distance', p=2, metric='euclidean')
    classifier.fit(X, np.ravel(Y))

    return classifier

probeA = swapColumns(probeA)
probeB = swapColumns(probeB)

probeA = probeA.drop('tna',1)

probeA_std = StandardScaler().fit_transform(probeA)
probeB_std = StandardScaler().fit_transform(probeB)

probeA_std = pd.DataFrame(probeA_std, columns=probeA.columns)
probeB_std = pd.DataFrame(probeB_std, columns=probeB.columns)

features = ['m2','n2','p1','c3^2','c3 m2','c3 n2','m1 p3','n2^2','n3^2','n3 p1','p2^2']

X_train = data_ord_2_A[features]
X_test = data_ord_2_B[features]

classifier = KNeighborsClassifier(n_neighbors = 24, weights='distance', p=2, metric='euclidean')
classifier.fit(X_train.values, np.ravel(classA))

predictions = classifier.predict_proba(X_test[features])[:,1]

m = KNN_plot(X_train.values, Y, 150)

df = pd.DataFrame(predictions)
df.to_csv("classB.csv", index=False,header = False)
