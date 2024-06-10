import numpy as np

# Author: Ben Sheff
# Made for the Optics group in the XSD Division of the APS at Argonne over
# the summer of 2015.
#
# This class is intended to save and load a python list as an asc file, after
# the format used typically throughout the group.
# This is more macro than class though, but to keep with the style of the rest
# of my code, it is kept in object style, with the option to make Ascii-
# converter objects that hold python lists.


class Medium:
    data_list = -1
    data_ascii = -1

    # Initialize the class by setting data_list and data_ascii to the
    # appropriate values for the given list or ascii string.
    def __init__(self, input_data_list):
        if (type(input_data_list) == list or
                type(input_data_list) == np.ndarray):
            self.data_list = input_data_list
            self.data_ascii = self.GetAscii()
        else:
            self.data_ascii = input_data_list
            self.data_list = self.GetList()

    # This method just wraps the ConvertToList method
    def GetList(self, input_data_ascii=data_ascii, split_term="\t"):
        if input_data_ascii == -1:
            input_data_ascii = self.data_ascii

        if type(input_data_ascii) != str and type(input_data_ascii) != ascii:
            return input_data_ascii
        self.data_ascii = input_data_ascii
        self.data_list = ConvertToList(input_data_ascii, split_term)
        return self.data_list

    # Again, just a wrapper for the ConvertToAscii method
    def GetAscii(self, input_data_list=data_list, split_term="\t"):
        if input_data_list == -1:
            input_data_list = self.data_list

        if type(input_data_list) == str or type(input_data_list) == ascii:
            return input_data_list
        self.data_list = input_data_list
        self.data_ascii = ConvertToAscii(input_data_list, split_term)
        return self.data_ascii


# Splits the inputted string based on a specified splitter, assuming rows
# are separated by a "\n". Then transposes the list.
def ConvertToList(input_data_ascii, split_term="\t"):
    if type(input_data_ascii) == ascii:
        input_data_ascii = str(input_data_ascii)
    if type(input_data_ascii) != str:
        return input_data_ascii
    list_form = input_data_ascii.split("\n")
    if list_form[len(list_form)-1] == "":
        list_form.pop()
    for row in range(len(list_form)):
        list_form[row] = list_form[row].split(split_term)
        while True:
            try:
                list_form[row].remove([])
            except:
                break
        while True:
            try:
                list_form[row].remove("")
            except:
                break
    while True:
        try:
            list_form.remove([""])
        except:
            break

    data_list = Transpose(list_form)
    for row in range(len(data_list)):
        data_list[row] = list(data_list[row])
    return data_list


# Converts the list to a string output after flattening it.
def ConvertToAscii(input_data_list, split_term="\t"):
    if type(input_data_list) == str or type(input_data_list) == ascii:
        return input_data_list
    data_ascii = ""
    data_list = __Flatten(input_data_list)
    for line in Transpose(data_list):
        data_ascii += "\n"
        first_element = True
        for datum in line:
            if first_element:
                first_element = False
            else:
                data_ascii += split_term
            data_ascii += str(datum)
    return data_ascii


# Flattens the given data list so it is only a list of lists, each list
# being of data.
def __Flatten(input_data_list):
    while True:
        try:
            input_data_list.remove([])
        except:
            break
    output_list = []
    if (type(input_data_list) != list and
            type(input_data_list) != np.ndarray):
        return None
    if (type(input_data_list[0]) != list and
            type(input_data_list[0]) != np.ndarray):
        output_list.append(input_data_list)
    else:
        for element in input_data_list:
            output_list += __Flatten(element)
    return output_list

def Transpose(ls):
    output = []
    mx = 0
    for row in ls:
        if len(row) > mx:
            mx = len(row)
    for column in range(mx):
        output.append([])
        for row in ls:
            try:
                tmp = row[column]
                output[column].append(tmp)
            except IndexError:
                output[column].append(None)
    return output
