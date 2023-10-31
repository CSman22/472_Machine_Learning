"""
Author      : Jackey Weng
Student ID  : 40130001
Description : 
"""
import pandas as pd


class Penguin:
    # Constructor
    def __init__(self, **attributes):
        # Dynamically set the arguments to the penguin class attributes
        for attribute, value in attributes.items():
            setattr(self, attribute, value)

    # return a string representation of penguin
    def __str__(self):
        output = ""
        # Get the max width of the attribute name
        max_attribute_width = 0
        # Iterate through each attribute name of the object
        for attribute in self.__dict__.keys():
            if len(attribute) > max_attribute_width:
                max_attribute_width = len(attribute)
        # Iterate through each attribute of the object and put it in a string
        for attribute, value in self.__dict__.items():
            output += f"{attribute.ljust(max_attribute_width)}: {value}\n"
        return output

    # Return a dictionary of the penguin attributes
    def to_dictionary(self):
        return self.__dict__
