# Step 1 - Problem formulation :
# Supervised learning problem : given a dataset with 6 features,
# predict if a certain crop can grow in that condition or not

# Importing the necessary packages
# For data manipulation
import pandas as pd
import numpy as np
from sklearn import metrics
# For splitting data into train and test datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Reload plt to avoid str' object is not callable error when using plt.title()
from importlib import reload
plt=reload(plt)
# For calculating the metrics
import sklearn.metrics as skl
# For encoding categorical data
from sklearn.preprocessing import LabelEncoder

# For feature scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# For finding the optimal hyperparameters
from sklearn.model_selection import RandomizedSearchCV
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
# For plotting the tree
from xgboost import plot_tree



import xgboost as xgb
import pickle

# Step 2 - Data collection, assessment and management
# Reading the dataset from csv into pandas dataframe
features_csv_file = "/Users/aelangovan/CS688_Smart_Farming_AI_model/final_crop_recommendation.csv"
_dataset = pd.read_csv(features_csv_file)
print(_dataset.shape)

# Step 3 - Data preprocessing: encoding categorical data
# Converting string to int so the classifier understands: each crop would become a number. Eg., rice=1, wheat=2
label_encoder = LabelEncoder()
dataset_columns = [column for column in _dataset.columns if _dataset[column].dtype == 'object']

print('classes are ', np.unique(_dataset['crop']))

for column in dataset_columns:
    _dataset[column] = label_encoder.fit_transform(_dataset[column])

# Step 4 - Feature Engineering
# Extracting Input / Attributes / Features - include all columns except 'crop' as it is the output
X = np.array(_dataset.iloc[:, 0:-1])
# Extracting Output / Target / Class / Labels
Y = np.array(_dataset.iloc[:, -1])
classes = np.unique(Y)
print('classes are \n', classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.3, shuffle=True, stratify=Y)


def build_prediction_model():
    # Step 5 - Build the XGBoost Classifier model with the hyperparameters
    estimator = xgb.XGBClassifier(
        min_child_weight=1,
        subsample=0.7,
        colsample_bytree=0.7,
        objective='multi:softmax',
        nthread=3,
        num_class=22,
        seed=27
    )

    parameters = {
        'max_depth': range(2, 10, 1),
        'n_estimators': range(20, 150, 10),
        'learning_rate': [0.1, 0.01, 0.05]
    }

    # Perform hyperparameter tuning and 5 fold cross-validation
    xgb_clf = RandomizedSearchCV(
        estimator,
        parameters,
        n_jobs=3,
        cv=5,
        verbose=True
    )

    # Fit the model on the training dataset
    xgb_clf.fit(X, Y)

    # Generate the best model
    xgb_clf = xgb_clf.best_estimator_

    # Predict the accuracy score for training & validation set
    print("\nAccuracy score for training & validation dataset:", xgb_clf.score(X_train, Y_train))

    # Step 6- Test the model on test data and generate classification report
    Y_pred = xgb_clf.predict(X_test)
    print('\nClassification report is - \n', skl.classification_report(Y_test, Y_pred))

    # Plot one tree from XGBoost as an example
    plot_tree(xgb_clf, num_trees=4)
    plt.show()

    # Plot feature importance for the model
    importance = xgb_clf.feature_importances_
    # summarize feature importance
    for i, imp in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (_dataset.iloc[:, 0:-1].columns[i], imp))
    # plot feature
    fig = pyplot.bar([x for x in _dataset.iloc[:, 0:-1].columns], importance)
    pyplot.show()

    # Step 7 - Save the model in pickle format to export to the flask server
    pickle.dump(xgb_clf, open('/Users/aelangovan/CS688_Smart_Farming_AI_model/model.pkl', 'wb'))


build_prediction_model()
