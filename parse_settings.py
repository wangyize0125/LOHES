"""
@Filename: parse_settings.py
@Author: Yize Wang
@Date: 2020/11/30
@Description: parse case settings stored in the .ini file
"""

import configparser

from settings import *
from error_codes import wrong_ini_file


def parse_settings(filename):
    """
    :param filename: .ini file containing the case settings
    :return: case settings in a dictionary
    """

    # instantiate a config object
    config = configparser.ConfigParser()
    config.read(filename)

    settings = {}
    # configparser has no check method for .ini files
    # here, we use try-except grammar to check
    try:
        # for global settings
        settings["length_x"] = config.getfloat("global", "length_x")
        settings["length_y"] = config.getfloat("global", "length_y")

        # for wind turbine layout optimization
        settings["num_turbine"] = config.getint("wind_turbine", "num_turbine")
        settings["sx_turbine"] = config.getfloat("wind_turbine", "sx_turbine")
        settings["sy_turbine"] = config.getfloat("wind_turbine", "sy_turbine")
        # I hope these codes can be utilized to optimize the layouts of wave energy converters for existing
        # offshore wind turbine farm. Hence, if predefined layouts are given, only converters will be optimized
        settings["pre_layouts"] = config.get("wind_turbine", "pre_layouts")

        # for wave energy converter layout optimization
        settings["num_converter"] = config.getint("wave_energy_converter", "num_converter")
        settings["sx_converter"] = config.getfloat("wave_energy_converter", "sx_converter")
        settings["sy_converter"] = config.getfloat("wave_energy_converter", "sy_converter")
    except configparser.NoSectionError as arg:
        # no section in .ini file
        print(ERROR + "Error: {} in {}.".format(arg, filename) + RESET)
        exit(wrong_ini_file)
    except configparser.NoOptionError as arg:
        # no option in .ini file
        print(ERROR + "Error: {} in {}.".format(arg, filename) + RESET)
        exit(wrong_ini_file)
    except ValueError as arg:
        # value error means str was given for functions int(), float(), boolean()
        print(ERROR + "Error: {} in {}.".format(arg, filename) + RESET)
        exit(wrong_ini_file)

    return settings


if __name__ == "__main__":
    print(parse_settings("./debug_case/config.ini"))
