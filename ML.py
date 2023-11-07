"""
Author      : Jackey Weng
Student ID  : 40130001
Description : Assignment 1
"""
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import statistics
import os
import pandas as pd
import matplotlib.pyplot as plt
from Model.Penguin import *
from Model.Abalone import *


class Machine_Learning_Algorithm:
    penguin_path: str = "Data/penguins.csv"
    abalone_path: str = "Data/abalone.csv"
    graph_path: str = "Graph"

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
            file_name = "penguin-classes.gif"
            graph_path = os.path.join(os.getcwd(), self.graph_path, file_name)
            plt.savefig(graph_path, format="png")
            # plt.show()
        elif self.data_option == 2:
            type_count = object_df["Type"].value_counts()
            # species_count.plot(kind="bar", color=["red", "green", "blue"])
            fig, ax = plt.subplots()
            ax.pie(type_count, labels=type_count.index, autopct="%1.1f%%")
            ax.set_title("Percentage of Abalone Type")
            ax.axis("equal")
            file_name = "abalone-classes.gif"
            graph_path = os.path.join(os.getcwd(), self.graph_path, file_name)
            plt.savefig(graph_path, format="png")
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

    # Calculate the performance of the model
    def performance(
        self, accuracy_list, macro_f1_list, weighted_f1_list, y_test, predictions
    ):
        # Performance of the decision tree
        conf_matrix = confusion_matrix(y_test, predictions)
        precision = precision_score(y_test, predictions, average=None) * 100
        recall = recall_score(y_test, predictions, average=None) * 100
        f1 = f1_score(y_test, predictions, average=None) * 100
        macro_f1 = f1_score(y_test, predictions, average="macro") * 100
        weighted_f1 = f1_score(y_test, predictions, average="weighted") * 100
        accuracy_percentage = accuracy_score(y_test, predictions) * 100

        # String representation of the performance
        result = ""
        result += f"(5.B) The Confusion Matrix:\n{conf_matrix}\n"
        result += "(5.C)\n"
        # Iterate through each class
        for i in range(len(precision)):
            result += f"Class {i}: Precision {precision[i]:.2f}% | Recall {recall[i]:.2f}% | F1 {f1[i]:.2f}%\n"
        result += f"(5.D)\n"
        result += f"Accuracy           : {accuracy_percentage:.2f}%\n"
        result += f"Macro-average F1   : {macro_f1:.2f}%\n"
        result += f"Weighted-average F1: {weighted_f1:.2f}%\n"

        # Calculating the average performance
        average_accuracy = sum(accuracy_list) / len(accuracy_list)
        variance_accuracy = statistics.variance(accuracy_list)
        stdev_accuracy = statistics.stdev(accuracy_list)
        average_macro_f1 = sum(macro_f1_list) / len(macro_f1_list)
        variance_macro_f1 = statistics.variance(macro_f1_list)
        stdev_macro_f1 = statistics.stdev(macro_f1_list)
        average_weighted_f1 = sum(weighted_f1_list) / len(weighted_f1_list)
        variance_weighted_f1 = statistics.variance(weighted_f1_list)
        stdev_weighted_f1 = statistics.pstdev(weighted_f1_list)
        result += "\nRun: 5\n"
        result += f"(6.A) Average Accuracy           : {average_accuracy:.2f}% | Variance: {variance_accuracy:.2f} | Standard Deviation: {stdev_accuracy:.2f}\n"
        result += f"(6.B) Average Macro-average F1   : {average_macro_f1:.2f}% | Variance: {variance_macro_f1:.2f} | Standard Deviation: {stdev_macro_f1:.2f}\n"
        result += f"(6.C) Average Weighted-average F1: {average_weighted_f1:.2f}% | Variance: {variance_weighted_f1:.2f} | Standard Deviation: {stdev_weighted_f1:.2f}\n"
        return result

    # Base Decision Tree Classifier
    def base_dt(self, x_train, y_train, x_test, y_test):
        # train decision tree
        dtc = tree.DecisionTreeClassifier()
        dtc.fit(x_train, y_train)
        predictions = dtc.predict(x_test)
        # print(predictions)

        # Plot the decision tree
        fig = plt.figure(figsize=(25, 20))
        tree.plot_tree(dtc)
        file_name = "base_DT_graph.png"
        graph_path = os.path.join(os.getcwd(), self.graph_path, file_name)
        fig.savefig(graph_path, format="png")

        # Part 6 Calculating the average performance
        accuracy_list = []
        macro_f1_list = []
        weighted_f1_list = []
        for i in range(5):
            dtc = tree.DecisionTreeClassifier()
            dtc.fit(x_train, y_train)
            predictions = dtc.predict(x_test)
            accuracy_list.append(accuracy_score(y_test, predictions) * 100)
            macro_f1_list.append(f1_score(y_test, predictions, average="macro") * 100)
            weighted_f1_list.append(
                f1_score(y_test, predictions, average="weighted") * 100
            )
        # Get the performance results
        dt_result = self.performance(
            accuracy_list, macro_f1_list, weighted_f1_list, y_test, predictions
        )
        part_a = "(5.A) *** Base-DT ***\n"
        dt_result = part_a + dt_result
        return dt_result

    # Top Decision Tree Classifier
    def top_dt(self, x_train, y_train, x_test, y_test):
        dtc = tree.DecisionTreeClassifier()
        params_dict = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
        }
        grid = None
        if self.data_option == 1:
            grid = GridSearchCV(
                dtc, param_grid=params_dict, scoring="f1_weighted", cv=10, n_jobs=-1
            )
        elif self.data_option == 2:
            grid = GridSearchCV(
                dtc, param_grid=params_dict, scoring="f1_macro", cv=10, n_jobs=-1
            )
        grid.fit(x_train, y_train)
        # print(grid.best_params_)

        # train decision tree with the best parameters
        top_dtc = tree.DecisionTreeClassifier(**grid.best_params_)
        top_dtc.fit(x_train, y_train)
        predictions = top_dtc.predict(x_test)
        # print(predictions)

        # Plot the decision tree
        fig = plt.figure(figsize=(25, 20))
        tree.plot_tree(top_dtc)
        file_name = "top_DT_graph.png"
        graph_path = os.path.join(os.getcwd(), self.graph_path, file_name)
        fig.savefig(graph_path, format="png")

        # Part 6 Calculating the average performance
        accuracy_list = []
        macro_f1_list = []
        weighted_f1_list = []
        for i in range(5):
            top_dtc = tree.DecisionTreeClassifier(**grid.best_params_)
            top_dtc.fit(x_train, y_train)
            predictions = top_dtc.predict(x_test)
            accuracy_list.append(accuracy_score(y_test, predictions) * 100)
            macro_f1_list.append(f1_score(y_test, predictions, average="macro") * 100)
            weighted_f1_list.append(
                f1_score(y_test, predictions, average="weighted") * 100
            )
        # Get the performance results
        top_dt_result = self.performance(
            accuracy_list, macro_f1_list, weighted_f1_list, y_test, predictions
        )
        part_a = "(5.A) *** Top-DT ***\n"
        top_dt_result = part_a + top_dt_result
        return top_dt_result

    # Write into file
    def write_to_file(self, dt_result, top_dt_result):
        output = "Jackey Weng 40130001\n"
        output += "\n"
        output += "Performance report \n"
        output += "----------------------------------------------------\n"
        output += dt_result
        output += "----------------------------------------------------\n"
        output += top_dt_result
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

    # Step 4, 5 & 6 train model and get performance
    # A) Base Decision Tree Classifier
    dt_result = MLP.base_dt(x_train, y_train, x_test, y_test)
    print("Performance Report")
    print_separator()
    print(dt_result)
    # B) Top Decision Tree Classifier
    top_dt_result = MLP.top_dt(x_train, y_train, x_test, y_test)
    print_separator()
    print(top_dt_result)

    # Write to file: performance of the model
    MLP.write_to_file(dt_result, top_dt_result)


if __name__ == "__main__":
    main()
