"""
@Filename: parse_settings.py
@Author: Yize Wang
@Date: 2020/11/30
@Description: parse case settings stored in the .ini file
"""

import configparser

from settings import ERROR, RESET, LOG
from error_codes import wrong_ini_file


def parse_settings(filename, default_settings):
    """
    :param filename: .ini file containing the case settings
    :param default_settings: default settings of the case
    :return: case settings in a dictionary
    """

    # instantiate a config object
    config = configparser.ConfigParser()
    config.read(filename)

    settings = {}
    # configparser has no check method for .ini files
    # here, we use try-except grammar to check manually
    try:
        # There are several issues should be encountered:
        # 1. check whether the sections and options in the .ini file are legal
        # 2. if they are legal, read their values into settings; if not, exit
        # 3. if options are not defined by the users, use default values
        sections = list(default_settings.keys())
        for sec in config.sections():
            # judge the designated section is legal in these codes
            if not (sec in sections):
                # if the section is illegal, raise a exception
                raise Exception("Section '{}' is illegal and unaccepted.".format(sec))
            else:
                # if the section is legal, delete section from the sections list
                sections.remove(sec)

            # use a nested dict to store the sub-settings
            tmp = {}
            options = list(default_settings[sec].keys())
            for opt in config.options(sec):
                # judge the designated option is legal in these codes
                if not (opt in options):
                    # if the option is illegal, raise a exception
                    raise Exception("Option '{}' in section '{}' is illegal and unaccepted.".format(opt, sec))
                else:
                    # if the option is legal, delete option from the options list
                    options.remove(opt)

                # legal section and option, record its value
                tmp[opt] = config.get(sec, opt)
            # if there are options that are not defined by user, use the default values
            if options:
                for opt in options:
                    tmp[opt] = default_settings[sec][opt]
                    print(LOG + "'{}': '{}' use the default value of '{}'".format(sec, opt, tmp[opt]) + RESET)

            # record this section
            settings[sec] = tmp
        # if there are sections that are not defined by user, use the default values
        if sections:
            for sec in sections:
                settings[sec] = default_settings[sec]
                print(LOG + "'{}' use the default value of '{}'".format(sec, settings[sec]) + RESET)
    except Exception as arg:
        # illegal sections and options
        print(ERROR + "Error: {} in {}.".format(arg, filename) + RESET)
        exit(wrong_ini_file)

    return settings


if __name__ == "__main__":
    import json
    import settings

    print(json.dumps(parse_settings("./debug_case/config.ini", settings.default_settings), indent=4))
