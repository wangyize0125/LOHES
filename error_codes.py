"""
@Filename: error_codes.py
@Author: Yize Wang
@Date: 2020/11/30
@Description: all defined errors and their corresponding codes
"""

# main.py accepts one input argument as the input filename. Commonly, it ends with ".ini".
# if there is no input argument for main.py or no input filename specified, then:
no_input_file = 1

# if the designated argument is not a file, then:
not_a_file = 2

# if the .ini file is organized in wrong format, then:
wrong_ini_file = 3
