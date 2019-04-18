__author__ = 'somnath'

import argparse
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from dataset import Dataset
from sklearn.model_selection import train_test_split, cross_val_score

class Regression:

    def __init__(self):
        pass

    def linerRegression(self,train_X, train_y, test_X):
        """
        Prediction using multiple linear regression
        :param train_X: <2d np array>
        :param train_y: <2d np array>
        :param test_X: <2d np array>
        :return: <list> : list of predicted values
        """
        # Train the model with data
        lm = linear_model.LinearRegression()
        reg = lm.fit(train_X,train_y)

        predictions = lm.predict(test_X)
        return predictions


    def elasticNet(self,train_X, train_y, test_X):
        reg = ElasticNet(random_state=0)
        reg.fit(train_X, train_y)
        predictions = reg.predict(test_X)

        return predictions

    def ridge(self,train_X, train_y, test_X):
        reg = Ridge(alpha=1.0, normalize=True ,tol=0.001)
        reg.fit(train_X, train_y)
        predictions = reg.predict(test_X)

        return predictions

    def MLPRegressor(self,train_X, train_y, test_X):
        """
        Using Multi Layer Perceptron  Regressor
        :param train_X: <np 2D array>
        :param train_y: <np 2D array>
        :param test_X : <np 2D array>
        :return: <np 1d array> : Prediction data
        """
        reg = MLPRegressor(hidden_layer_sizes=(200, ),
                           activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant',
                           learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True,
                           tol=0.0001,  momentum=0.9, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
                           )
        # Train model with data (X,y)
        reg.fit(train_X, train_y)
        # Test data with trained model.
        predictions = reg.predict(test_X)

        return predictions

    def evaluation(self, X, y):
        """
        Evaluation: Validate the model using Cross validation
        :param X: Train X
        :param Y: Train y
        :return: None
        """
        print("INFO: Evaluation- Validate model with cross validation ... ")
        """
        lm = linear_model.LinearRegression()
        scores = cross_val_score(lm, X, y, cv=5)
        print ("INFO: Model Score using linear regression :{}".format(scores))

        reg = ElasticNet(random_state=0)
        scores = cross_val_score(reg, X, y, cv=5)
        print("INFO: Model Score using ElasticNet {}".format(scores))

        reg = Ridge(alpha=1.0, normalize=True ,tol=0.001)
        scores = cross_val_score(reg, X, y, cv=5)
        print("INFO: Model Score using Ridge {}".format(scores))
        """

        reg = MLPRegressor(hidden_layer_sizes=(200, ),
                           activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant',
                           learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True, tol=0.0001, momentum=0.9,
                            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
                           )
        scores = cross_val_score(reg, X, y, cv=5)
        print("INFO: Model Score using MLP Regressor {}".format(scores))
        print("      Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





def commandLineArgumentParser():
    parser = argparse.ArgumentParser(description='Help message')
    parser.add_argument('-trip_train', type=str, required=True, help='Provide train trip csv file ')
    parser.add_argument('-order_item',type=str, required=True, help='Specify order item csv file')
    parser.add_argument('-trip_test',type=str, required=True, help='Specify test trip csv file')


    options = parser.parse_args()

    return vars(options)



if __name__ == "__main__":

    params = commandLineArgumentParser()

    train_trip = params['trip_train']
    test_trip  = params['trip_test']
    order_item = params['order_item']

    dataset = Dataset()
    print("INFO: Loading training data ...")
    dataset.loadData(train_trip,order_item)

    print("INFO: len X - {}".format(len(dataset.X)))
    print("INFO: len Y - {}".format(len(dataset.Y)))
    print("INFO: Features ... \n\t{}".format(dataset.features))


    # Create model
    df = pd.DataFrame(dataset.X, columns=dataset.features)
    target = pd.DataFrame(dataset.Y, columns=["time"])

    train_X = df
    train_y = target["time"]

    # Test dataset
    test_dataset = Dataset()
    print("INFO: Loading test data ... ")
    test_dataset.loadData(test_trip, order_item, 'test')
    print("INFO: len test X - {}".format(len(test_dataset.X)))

    # Create model and test
    reg = Regression()


    # Model Evaluation
    reg.evaluation(train_X, train_y)


    #predictions = reg.linerRegression(train_X, train_y, test_dataset.X)
    #predictions = reg.elasticNet(train_X, train_y, test_dataset.X)
    #predictions = reg.ridge(train_X, train_y, test_dataset.X)

    predictions = reg.MLPRegressor(train_X, train_y, test_dataset.X)

    # Result store in a csv file in the form of trip_id and shopping time
    print("INFO: Writing test results into Result.csv ...")
    print("INFO:\t Total Prediction:{}, Total TestTrips-{}".format(len(predictions), len(test_dataset.test_trip_ids)))

    index = 0
    with open("Result.csv", 'w') as fh:
        fh.write("trip_id,shopping_time\n")
        while index < len(predictions):
            fh.write(test_dataset.test_trip_ids[index] + ',' + str(int(predictions[index])) + '\n')
            index +=1














