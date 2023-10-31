"""
Author      : Jackey Weng
Student ID  : 40130001
Description : 
"""
from sklearn.datasets import load_wine
from sklearn import datasets
import os
import pandas as pd
import matplotlib.pyplot as plt
from Model.Penguin import *
from Model.Abalone import *


# Return a list of the data set as an object


def get_data_set(data_option: int) -> []:
    current_directory = os.getcwd()
    penguin_path = "Data/penguins.csv"
    abalone_path = "Data/abalone.csv"
    data_path = ""
    data_list = []

    # Retrieve penguin data set
    if data_option == 1:
        # Read the csv file
        data_path = os.path.join(current_directory, penguin_path)
        df = pd.read_csv(data_path)

        # Count the occurrences of each species
        species_counts = df["species"].value_counts()
        print(species_counts)

        # Plot the percentage of instances for each species
        plt.figure(figsize=(8, 6))
        species_counts.plot(kind="pie", autopct="%1.1f%%")
        plt.title("Percentage of Penguin Species")
        plt.ylabel("")  # Remove the default ylabel

        # Save the plot as a GIF
        # plt.savefig("penguin-classes.gif", format="gif", dpi=100)

        # Show the plot (optional)
        plt.show()

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
    elif data_option == 2:
        # Read the csv file
        data_path = os.path.join(current_directory, abalone_path)
        df = pd.read_csv(data_path)

        # Iterate through the data rows and create object
        for index, row in df.iterrows():
            abalone = Abalone(**row)
            data_list.append(abalone)
    return data_list


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


def main():
    print_separator()
    data_option = choose_data_set()
    print_separator()
    data_set = get_data_set(data_option)
    count = 0
    for e in data_set:
        print(e)
        count += 1
        if count == 3:
            break


if __name__ == "__main__":
    main()
