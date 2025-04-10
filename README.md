# Accuracy-Based-Recursion-Feature-Elimination
Code for an original algorithm for optimal feature elimination

This algorithm is an original creation of my own, but does share some similarities to Recursive Feature Elimination, hence the similar name.
The main difference between these two algorithms is that RFE calculates feature importance to remove the least important feature until desired feature count is reached. ARFE has a different goal, which is to find the optimal ML model accuracy with Feature Elimination. An unspecified amount of features will be removed to achieve maximum accuracy, rather than removing a certain amount of the least important features. 
This algorithm can be used for feature reduction for any ML model, but is best with models that do not vary in accuracy or error within runs. The code given in this repository runs on a default KNN model, but can easily be modified for any scikit-learn model and hyperparameter selection.

The importance of this algorithm is not necessarily efficiency; in fact, the algorithm is not the most efficient available algorithm for feature elimination. It's value lies in being a simple algorithm to both understand and use, correctly eliminating features for the optimal dataset for model performance and accuracy. It's useful mostly for beginning Machine Learning students like myself to better understand the process and necessity of feature selection.

The algorithm works in only a few basic steps:
1. Define x and y in x2=ARFE(x,y) as the x and y splits of the original dataset. y is only necessary for accuracy measurements, and the features in x will be altered to improve the accuracy. The optimal dataset will be returned
2. At the start of each function, define the starting x dataset as x, and evaluate the model's cross validation score (calculated from accuracy)
3. Create a loop to remove one feature from x and evaluating the cross validation score of the model. If the score is the greatest among all datasets with one removed feature, that score is saved as the highest "removed feature dataset" score
4. If the dataset with a removed feature yields a more accurate model than the starting dataset, the removed feature dataset becomes the starting dataset, and the function is called again as ARFE(removedFeaturex, y).
5. If the starting dataset yields more accurate than all one-feature-removed datasets, that dataset is the optimal dataset for the model. This dataset is then returned to replace x.

