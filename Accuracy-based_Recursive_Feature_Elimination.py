import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
"""
This example uses KNN, but this method can be used with any ML model that has some sort of accuracy to it.
For example, regression models can use Mean Squared Error (find the lowest MSE)
"""

"""
To implement this model with any other ML model, follow these steps:
1. Import the library of the desired model
2. Replace the hyperparameters (any input after the third in ARFE) with ALL of the hyperparameters of your desired model, setting the default value as required
3. Replace calls to the model with your desired model, including the model's hyperparameters
4. Make sure the hyperparameters inputted into the recursive call are replaced with the hyperparameters of your model
"""

#after the third input, the rest are knn hyperparameters
#default values are set so that if no hyperparameters are given, a default model runs
def ARFE(x,y,avgScoreOriginal=0, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
    #if starting model score >= to normal, function ends
    #test original model
    """
    If the avgScoreOriginal is passed before function call, this is skipped.
    This means that if a feature is dropped, avgScoreOriginal can be altered then and passed on without redundantly recalculating it when the function is recursively called
    However, you can call the function outside of recursion with just ARFE(x,y), since calculating avgScoreOriginal then saves no run time
    """
    if(avgScoreOriginal==0):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)
        model.fit(x,y)
        cvScore = cross_val_score(model, x, y, cv=4)
        avgScoreOriginal = np.average(cvScore)
        avgScoreOriginal = avgScoreOriginal*100
    
    #test model with a removed column each time
    #first value represents score, second represents index
    avgScoreRemoved = [0, -1]
    for k in range(x.shape[1]):
        x2 = x.drop(x.columns[k],axis=1)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)
        model.fit(x2,y)
        cvScore = cross_val_score(model, x2, y, cv=4)
        avgScore = np.average(cvScore)
        avgScore = avgScore*100
        if(avgScore>avgScoreRemoved[0]):
            avgScoreRemoved = [avgScore, k]

    #now compare starting and test
    if(avgScoreRemoved[0]>avgScoreOriginal):
        avgScoreOriginal = avgScoreRemoved[0]
        x = x.drop(x.columns[avgScoreRemoved[1]], axis=1)
        return ARFE(x,y,avgScoreOriginal, n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs)
    else:
        return x
