"""
@Filename: parse_settings.py
@Author: Yize Wang
@Date: 2020/11/30
@Description: parse case settings stored in the .ini file
"""

import os

from error_codes import not_ini_file


def parse_settings(filename):
    """
    :param filename: .ini file containing the case settings
    :return: case settings in a dictionary
    """

    pass


if __name__ == "__main__":
    print(parse_settings("./debug_case.config.ini"))
