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
def ARFE(x,y,avgScoreOriginal=0):
    #if starting model score >= to normal, function ends
    #test original model
    """
    NOTE: I will not mess with hyperparameters since I am only interested in feature elimination
    """
    """
    If the avgScoreOriginal is passed before function call, this is skipped.
    This means that if a feature is dropped, avgScoreOriginal can be altered then and passed on without redundantly recalculating it when the function is recursively called
    However, you can call the function outside of recursion with just ARFE(x,y), since calculating avgScoreOriginal then saves no run time
    """
    if(avgScoreOriginal==0):
        knn = KNeighborsClassifier()
        knn.fit(x,y)
        cvScore = cross_val_score(knn, x, y, cv=4)
        avgScoreOriginal = np.average(cvScore)
        avgScoreOriginal = avgScoreOriginal*100
    
    #test model with a removed column each time
    #first value represents score, second represents index
    avgScoreRemoved = [0, -1]
    for k in range(x.shape[1]):
        x2 = x.drop(x.columns[k],axis=1)
        knn = KNeighborsClassifier()
        knn.fit(x2,y)
        cvScore = cross_val_score(knn, x2, y, cv=4)
        avgScore = np.average(cvScore)
        avgScore = avgScore*100
        if(avgScore>avgScoreRemoved[0]):
            avgScoreRemoved = [avgScore, k]

    #now compare starting and test
    if(avgScoreRemoved[0]>avgScoreOriginal):
        avgScoreOriginal = avgScoreRemoved[0]
        x = x.drop(x.columns[avgScoreRemoved[1]], axis=1)
        return ARFE(x,y,avgScoreOriginal)
    else:
        return x


#test: remove in github
#"""
#import csv
csvFile = pd.read_csv('Brawl_Stats_Dataset.csv')
#define x and y for train_test_split
#we need to remove the target column from x
y = []
for k in range(len(csvFile)):
    i = csvFile.loc[k, 'Type']
    y.append(i)

for k in range(len(y)):
    if y[k] == 'Artillery':
        y[k] = 1
    if y[k] == 'Assassin':
        y[k] = 2
    if y[k] == 'Controller':
        y[k] = 3
    if y[k] == 'Damage Dealer':
        y[k] = 4
    if y[k] == 'Marksman':
        y[k] = 5
    if y[k] == 'Support':
        y[k] = 6
    if y[k] == 'Tank':
        y[k] = 7

        
csvFile2 = pd.read_csv('Brawl_Stats_Dataset.csv')
csvFile2 = csvFile2.drop('Type', axis=1)
csvFile2 = csvFile2[['Gem Grab Win Rate', 'Heist Win Rate', 'Knockout Win Rate', 'Brawl Ball Win Rate', 'Hot Zone Win Rate', 'Wipeout Win Rate', 'Basket Brawl Win Rate', 'Bounty Win Rate']]
csvFile2.dropna(subset=['Gem Grab Win Rate'], inplace=True)
csvFile2.dropna(subset=['Heist Win Rate'], inplace=True)
csvFile2.dropna(subset=['Knockout Win Rate'], inplace=True)
csvFile2.dropna(subset=['Brawl Ball Win Rate'], inplace=True)
csvFile2.dropna(subset=['Hot Zone Win Rate'], inplace=True)
csvFile2.dropna(subset=['Wipeout Win Rate'], inplace=True)
csvFile2.dropna(subset=['Basket Brawl Win Rate'], inplace=True)
csvFile2.dropna(subset=['Bounty Win Rate'], inplace=True)

x = csvFile2

print("BEFORE")
for k in range(x.shape[1]):
    print(x.columns[k])

x = ARFE(x,y)

print()
print("AFTER")
for k in range(x.shape[1]):
    print(x.columns[k])

#"""

