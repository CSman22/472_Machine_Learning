"""
Author      : Jackey Weng
Student ID  : 40130001
Description : Assignment 1
"""
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
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

    # Plot the percentage of the instances in each output class
    def plot_output_class(self, dataset: list):
        dataset = [obj.to_dictionary() for obj in dataset]
        object_df = pd.DataFrame(dataset)
        if self.data_option == 1:
            species_count = object_df["species"].value_counts()
            fig, ax = plt.subplots()
            # species_count.plot(kind="bar", color=["red", "green", "blue"])
            ax.pie(species_count, labels=species_count.index, autopct="%1.1f%%")
            ax.set_title("Percentage of Penguin Species")
            ax.axis("equal")
            plt.savefig("penguin-classes.gif", format="png", dpi=100)
            # plt.show()
        elif self.data_option == 2:
            type_count = object_df["Type"].value_counts()
            # species_count.plot(kind="bar", color=["red", "green", "blue"])
            fig, ax = plt.subplots()
            ax.pie(type_count, labels=type_count.index, autopct="%1.1f%%")
            ax.set_title("Percentage of Abalone Type")
            ax.axis("equal")
            plt.savefig("abalone-classes.gif", format="png", dpi=100)
            # plt.show()

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

    # Base Decision Tree Classifier
    def base_dt(self, x_train, y_train, x_test, y_test):
        # train decision tree
        dtc = tree.DecisionTreeClassifier()
        dtc.fit(x_train, y_train)
        predictions = dtc.predict(x_test)
        # print(predictions)

        # Performance of the decision tree
        conf_matrix = confusion_matrix(y_test, predictions)
        precision = precision_score(y_test, predictions, average=None) * 100
        recall = recall_score(y_test, predictions, average=None) * 100
        f1 = f1_score(y_test, predictions, average=None) * 100
        macro_f1 = f1_score(y_test, predictions, average="macro") * 100
        weighted_f1 = f1_score(y_test, predictions, average="weighted") * 100
        accuracy_percentage = dtc.score(x_test, y_test) * 100

        # String representation of the performance
        result = "***(5.A) Base-DT***\n"
        result += f"(B) The Confusion Matrix:\n{conf_matrix}\n"
        result += "(C) The precision, recall, and F1-measure for each class:\n"
        # Iterate through precision scores
        for i in range(len(precision)):
            result += f"Class {i}: Precision {precision[i]:.2f} | Recall {recall[i]:.2f} | F1 {f1[i]:.2f}\n"
        result += f"(D) The accuracy, macro-average F1 and weighted-average F1 of the model:\n"
        result += f"Accuracy: {accuracy_percentage:.2f}% | Macro-average F1: {macro_f1:.2f}% | weighted-average F1: {weighted_f1:.2f}%\n "
        print(result)

        # Plot the decision tree
        fig = plt.figure(figsize=(25, 20))
        tree.plot_tree(dtc)
        fig.savefig("base_DT_graph.png")
        return result

    # Write into file
    def write_to_file(self, dt_result):
        output = "Jackey Weng 40130001\n"
        output += ""
        output += dt_result
        # record information to file
        if self.data_option == 1:
            with open("penguin-performance.txt", "w") as file:
                file.write(output)
        elif self.data_option == 2:
            with open("abalone-performance.txt", "w") as file:
                file.write(output)


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
    MLP.plot_output_class(dataset)

    # Step 3 split data set into training and testing (x= feature, y = target)
    x_train, x_test, y_train, y_test = MLP.train_test_set(dataset)
    # print(y_test.head(10))

    # Step 4 train model
    # 4.a Base Decision Tree Classifier
    dt_result = MLP.base_dt(x_train, y_train, x_test, y_test)

    # Step 5 Write to file: performance of the model
    MLP.write_to_file(dt_result)


if __name__ == "__main__":
    main()
