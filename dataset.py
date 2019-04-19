__author__ = 'somnath'

import os,sys
import numpy as np


class Dataset:
    """
    Prepare dataset from train trips and order items. The independent features are chosen such a way that
    each of them involved in shopping time
    Independent features are..
    shopper_id,
    fulfilment_model => converted to numeric value from "model_1" to "1"
    store_id,
    department_name => combined total department visited in a trip
    quantity => Added all items quantities for a trip

    """

    def __init__(self):

        self.features = []
        self.train_trips = {}
        self.order_items = {}

        # Data type
        self.data_type = ['train', 'test' ]

        # Train
        self.X = []
        self.Y = []

        # Test trip id
        self.test_trip_ids = []


    def setFeatures(self,features=[]):
        if not self.features:
            self.features = features
        else:
            self.features.extend(features)


    def getFeatureNames(self):
        '''
        :return:<list> - list of feature names
        '''
        return  self.features


    def loadData(self, trips_input='', order_input='', type='train'):
        """
        - Create the 2D dataset for all features
        Features of independent variables: ['']
        :trip_input
        :order_input
        :type: <string>: training or test
        :return:<list of list> :
        """

        # Check data type
        if type not in self.data_type:
            print("ERROR: Wrong data type {}, possible values {}".format(type, self.data_type))
            return


        # Get trips data
        trips_data = self.processTrips(trips_input, type)
        orders_data = self.processOrderItems(order_input)

        data = []
        index = 0
        valid_Y = []
        for trip_id in trips_data:
            d = trips_data[trip_id]
            # Drop data if trip_id not present into orders_items
            if orders_data.get(trip_id, False):
                order_items = orders_data.get(trip_id, [])
                data.append(d + order_items)
                if type == 'train':
                    valid_Y.append(self.Y[index])

                # Add trip ids for test trips
                if type == 'test':
                    self.test_trip_ids.append(trip_id)
            else:
                print("WARNING: trip {} data dropped as corresponding order items not found".format(trip_id))

            index +=1

        # Set train data to independent variable X
        self.X =  np.array(data, dtype=float)

        if type == 'train':
            print("DEBUG: Setting Y predicted values as np array")
            self.Y = np.array(valid_Y, dtype=float)

    def processOrderItems(self, order_input=""):
        """
        - Read inputs from Order_items file, preprocess and store into self.order_items
        - Considered independent variables
          - Number of departments visited in a trip
          - Number of product purchased for a trip
        <trip_id>:[<number of department visited>,<products purchased>]
        :return: <dict> : key is the trip id
        """
        print("INFO: Processing order items ...")
        features = []
        order_items = {}
        with open (order_input , 'r') as fh:
            features = (fh.readline().strip()).split(',')
            for line in fh:
                line = line.strip()
                fields = line.split(',')

                # Convert department name to a numeric values
                #if fields[2] not in department_name:
                #    department_name[fields[2]] = count
                #    count +=1
                # Replace department name with a numeric value
                #fields[2] = department_name[fields[2]]

                if fields[0] not in order_items:
                    # Initialize with [1,<product numbers>], 1 indicate first department visited in a store,
                    # second value is number of product purchased from that department
                    order_items[fields[0]] = [1,float(fields[-1])]
                else:
                    order_items[fields[0]][0] +=1  # Increment department count
                    order_items[fields[0]][0] += float(fields[-1]) # Add product counts

        # Add features
        self.setFeatures(features[2:])

        return order_items



    def processTrips(self, trips_input="", type="train"):
        """
        Load the data from file into dictionary
        Training data features:
        trip_id,shopper_id,fulfillment_model,store_id,shopping time
        :return: <dict> : trip data where trip id is the key.
        """
        print("INFO: Processing {} data".format(type))
        trips_data = {}
        with open (trips_input, 'r') as fh:
            features = ((fh.readline().strip()).split(','))[:-2]
            for line in fh:
                line = line.strip()
                fields = line.split(',')


                if type == 'train':
                    time = self.getTimeDifference(fields[-2], fields[-1])
                    self.Y.append(time)

                ## add shopper id, fulfillment model, store id
                # Update fulfilment model as numeric values 1,2 instead of model_1/model_2
                fields[2] = (fields[2].strip('_'))[-1]
                if fields[0] not in trips_data:
                    trips_data[fields[0]]= fields[1:4]
                else:
                    print("DEBUG: extra entry for trip id {}".format(fields[0]))
        # Add features, excluding trip_id
        self.setFeatures(features[1:])

        return trips_data




    def getTimeDifference(self, start, end):
        """
        - get time difference in seconds
        :param start: <string> : 2015-09-01 07:41:06
        :param end: <string> : 2015-09-01 08:46:06
        :return: <int> : time difference in seconds
        """

        (start_date, start_time) = start.split()
        (end_date, end_time) = end.split()
        time = 0.0
        days = 0
        if start_date != end_date:
            #print("DEBUG: Different date {} | {}".format(start,end))
            s_days = int(start_date.split('-')[-1])
            e_days = int(end_date.split('-')[-1])
            days = e_days - s_days

        s_times = start_time.split(':')
        e_times = end_time.split(':')
        s_time_sec = int(s_times[0]) * 3600 + int(s_times[1]) * 60 + int(s_times[2])
        e_time_sec = (int(e_times[0])+ days*24) * 3600 + int(e_times[1]) * 60 + int(e_times[2])
        time = e_time_sec - s_time_sec
        return time


