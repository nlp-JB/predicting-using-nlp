import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import explore


def get_splits(df, vectorized_df):
    X = vectorized_df
    y = df.gen_language
    #Split dataframe into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state = 123)
    train_predictions = pd.DataFrame(dict(actual=y_train))
    test_predictions = pd.DataFrame(dict(actual=y_test))

    return X_train, X_test, y_train, y_test, train_predictions, test_predictions


def logistic_regression(X_train, X_test, y_train, y_test):
    '''
    Takes in X_train, X_test, y_train, y_test
    Returns the train and test predictions
    '''
    
    # Create and fit the model on the train and test data

    lm = LogisticRegression().fit(X_train, y_train)
    # Create predictions

    train_logistic_regression_predictions = lm.predict(X_train)
    test_logistic_regression_predictions = lm.predict(X_test)

    
    return train_logistic_regression_predictions, test_logistic_regression_predictions


def random_forest_classifier(X_train, X_test, y_train, y_test):
    '''
    Takes in X_train, X_test, y_train, y_test
    Returns the train and test predictions

    '''

    # Create and fit the model on the train and test data
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=3,
                            n_estimators=100,
                            max_depth=3, 
                            random_state=123)
    rf.fit(X_train, y_train)
    # Create predictions
    train_random_forest_predictions = rf.predict(X_train)
    test_random_forest_predictions = rf.predict(X_test)

    return train_random_forest_predictions, test_random_forest_predictions


def knn_classifier(X_train, X_test, y_train, y_test):
    '''
    Takes in X_train, X_test, y_train, y_test
    Returns the train and test predictions

    '''

    # Create and fit the model on the train and test data
    knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    knn.fit(X_train, y_train)
    # Create predictions
    train_knn_predictions = knn.predict(X_train)
    test_knn_predictions = knn.predict(X_test)

    return train_knn_predictions, test_knn_predictions


def make_predictions_df(df, vectorized_df):

    X_train, X_test, y_train, y_test, train_predictions, test_predictions = get_splits(df, vectorized_df)


    # Add baseline to predictions dataframe
    train_predictions['baseline'] = 'Python'
    test_predictions['baseline'] = 'Python'

    # Create logistic regression predictions

    train_logistic_regression_predictions, test_logistic_regression_predictions = logistic_regression(X_train, X_test, y_train, y_test)
        
    # Add logistic regression predictions to predictions dfs
    train_predictions['lr_predictions'] = train_logistic_regression_predictions
    test_predictions['lr_predictions'] = test_logistic_regression_predictions

    #Create random forest predictions
    train_random_forest_predictions, test_random_forest_predictions = random_forest_classifier(X_train, X_test, y_train, y_test)

    # Add random forest predictions to dfs
    train_predictions['rf_predictions'] = train_random_forest_predictions
    test_predictions['rf_predictions'] = test_random_forest_predictions


    # Create knn predictions
    train_knn_predictions, test_knn_predictions = knn_classifier(X_train, X_test, y_train, y_test)

    # Add random forest predictions to dfs
    train_predictions['knn_predictions'] = train_knn_predictions
    test_predictions['knn_predictions'] = test_knn_predictions




    return train_predictions, test_predictions


def train_evaluation(train_predictions):
    # Logistic regression accuracy score, confustion matrix, classification report for train data
    print('Evaluation Metrics for Logistic Regression Model')
    print()
    print()
    print('Accuracy: {:.2%}'.format(accuracy_score(train_predictions.actual, train_predictions.lr_predictions)))
    print('----------------------------------------------------------------------------------------------')
    print('Confusion Matrix')
    print(pd.crosstab(train_predictions.lr_predictions, train_predictions.actual))
    print('----------------------------------------------------------------------------------------------')
    print(classification_report(train_predictions.actual, train_predictions.lr_predictions))

    print()
    print()
    print()
    print()
    print()

    # Random Forest accuracy score, confustion matrix, classification report for train data
    print('Evaluation Metrics for Random Forest Model')
    print()
    print()
    print('Accuracy: {:.2%}'.format(accuracy_score(train_predictions.actual, train_predictions.rf_predictions)))
    print('----------------------------------------------------------------------------------------------')
    print('Confusion Matrix')
    print(pd.crosstab(train_predictions.rf_predictions, train_predictions.actual))
    print('----------------------------------------------------------------------------------------------')
    print(classification_report(train_predictions.actual, train_predictions.rf_predictions))

    print()
    print()
    print()
    print()
    print()
    

    # K Nearest Neighbors accuracy score, confustion matrix, classification report for train data
    print('Evaluation Metrics for K Nearest Nerighbors Model')
    print()
    print()
    print('Accuracy: {:.2%}'.format(accuracy_score(train_predictions.actual, train_predictions.knn_predictions)))
    print('----------------------------------------------------------------------------------------------')
    print('Confusion Matrix')
    print(pd.crosstab(train_predictions.knn_predictions, train_predictions.actual))
    print('----------------------------------------------------------------------------------------------')
    print(classification_report(train_predictions.actual, train_predictions.knn_predictions))

