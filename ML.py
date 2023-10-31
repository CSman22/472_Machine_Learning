"""
Author      : Jackey Weng
Student ID  : 40130001
Description : 
"""
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.pyplot as plt
from Model.Penguin import *
from Model.Abalone import *


class Machine_Learning_Algorithm:
    penguin_path: str = "Data/penguins.csv"
    abalone_path: str = "Data/abalone.csv"

    def __init__(self, data_option=0):
        self.data_option = data_option

    # Return a list of the data set as an object
    def get_data_set(self) -> list:
        current_directory = os.getcwd()
        data_path = ""
        data_list = []

        # Retrieve penguin data set
        if self.data_option == 1:
            # Read the csv file
            data_path = os.path.join(current_directory, self.penguin_path)
            df = pd.read_csv(data_path)

            # Converting the sex and island feature to one hot encoding
            df = pd.get_dummies(df, columns=["sex", "island"])
            # Iterate through the data rows and create object
            for index, row in df.iterrows():
                # dynamically set the arguments to the object class
                constructor_args = {}
                for column in df.columns:
                    constructor_args[column] = row[column]
                penguin = Penguin(**constructor_args)
                data_list.append(penguin)

        # Retrieve abalone data set
        elif self.data_option == 2:
            # Read the csv file
            data_path = os.path.join(current_directory, self.abalone_path)
            df = pd.read_csv(data_path)

            # Iterate through the data rows and create object
            for index, row in df.iterrows():
                constructor_args = {}
                for column in df.columns:
                    constructor_args[column] = row[column]
                abalone = Abalone(**constructor_args)
                data_list.append(abalone)
        return data_list

    # Split of the dataset into training and testing
    # Return both set
    def train_test_set(
        self,
        dataset: list,
        test_size: float = None,
        train_size: float = None,
        random_state: int = None,
    ):
        dataset = [obj.to_dictionary() for obj in dataset]
        object_df = pd.DataFrame(dataset)
        x = None
        y = None
        if self.data_option == 1:
            target = "species"
            # Get all the feature columns
            x = object_df.drop(columns=[target])
            # Get the target column
            y = object_df[target]
        elif self.data_option == 2:
            target = "Type"
            # Get all the feature columns
            x = object_df.drop(columns=[target])
            # Get the target column
            y = object_df[target]

        # Split the data set into training and testing
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, train_size=train_size, random_state=random_state
        )
        return x_train, x_test, y_train, y_test


# Return the data set option chosen by the user
def choose_data_set() -> int:
    while True:
        print("Choose a data set (Default 1) \n 1. Penguin \n 2. Abalone")
        user_input = input("Enter a number: ")
        if user_input.strip() == "":
            return 1
        try:
            user_input = int(user_input)
            if user_input == 1 or user_input == 2:
                return user_input
            else:
                print("Invalid Input: Please choose a option\n")
        except ValueError:
            print("Invalid input: Please enter a number\n")


def print_separator():
    print("--------------------------------")


# Print the data set
def print_dataset(dataset: [], limit: int = 1):
    count = 0
    for e in dataset:
        print(e.to_dictionary())
        print(e)
        count += 1
        if count == limit:
            break


def main():
    print_separator()
    data_option = choose_data_set()
    print_separator()
    MLP = Machine_Learning_Algorithm(data_option=data_option)
    # Step 1 load data set
    dataset = MLP.get_data_set()
    # print_dataset(dataset)

    # Step 2 Plot the percentage of the instances in each output class

    # Step 3 split data set into training and testing (x= feature, y = target)
    x_train, x_test, y_train, y_test = MLP.train_test_set(dataset)
    print(x_train, y_train)
    # Step 4 train model


if __name__ == "__main__":
    main()
