"""
@Filename: parse_arg.py
@Author: Yize Wang
@Date: 2020/11/30
@Description: parse the project folder path from the input argument
"""

import os

from settings import *
from error_codes import no_input_file, not_a_file


def parse_arg(sys_argv):
    """
    :param sys_argv: sys.argv of the main.py
    :return: case name - only contains the filename
             project folder path: is the absolute father path of the file
    """

    # main.py accepts one input argument being the configuration file for a new case
    # if there is no input, exit the codes
    if len(sys_argv) <= 1:
        print(ERROR + "Error: No input file specified." + RESET)
        exit(no_input_file)

    # input argument should be the case name, here we judge the file type first
    case_name = None
    if os.path.isfile(sys_argv[1]):
        case_name = os.path.basename(sys_argv[1])
    else:
        print(ERROR + "Error: {} is not a file.".format(sys_argv[1]) + RESET)
        exit(not_a_file)

    # calculate the father path of the specified input file as the project folder
    # all outputs will be stored in this folder later
    proj_dir = os.path.dirname(os.path.abspath(sys_argv[1]))

    return case_name, proj_dir


if __name__ == "__main__":
    import sys

    print(parse_arg(sys.argv))
