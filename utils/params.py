import json
import operator
import sys
from distutils.util import strtobool
from functools import reduce
from pathlib import Path
from typing import List, Optional, Union

from utils.singleton import Singleton


class Params(dict, metaclass=Singleton):
    """
    Singelton object to parse parameter file.
    Reads the provided parameters file once, and saves it internally as a dictionary.

    Provides the following functionality:
        * single read
        * write/update file
        * allow overwriting the keys from command line arguments by providing a
          key-value pair (sub dictionaries accessible with the slash char)
    """

    def __init__(self, json_file: Union[Path, str, None] = None) -> None:
        """
        Instantiates the object and reads the parameters from the provided file

        :param json_file: JSON file containing the parameters
        :return: None
        """
        if json_file is None:
            raise RuntimeError("Params singleton was never initialized!")
        super().__init__()
        self.read(json_file)

    def read(self, json_file: Union[str, Path]) -> None:
        """
        Reads the parameters provided in the JSON file and stores them in a dict

        :param json_file: JSON file containing the parameters
        :return: None
        """
        self.clear()
        self.update(json.load(open(json_file)))

    def write(self, json_file: Union[str, Path]) -> None:
        """
        Stores the parameters in a JSON file

        :param json_file: JSON file where the parameters should be stored
        :return: None
        """
        json.dump(self, open(json_file, "w"), indent=4, sort_keys=True)

    def parse_args(self, set_vals: Optional[List[str]] = None) -> None:
        """
        Sets the stored parameters to the provided values

        :param set_vals: List of key-value pairs corresponding to the parameter to be set.
        Nested parameters can be set by separating the levels with a "/"
        :return: None
        """
        if set_vals is None:
            set_vals = sys.argv[1:]

        if len(set_vals) % 2 != 0:
            raise RuntimeError("Odd number of arguments provided")

        set_vals = dict(zip(set_vals[::2], set_vals[1::2]))
        for key, value in set_vals.items():
            key_levels = key.split("/")

            try:
                last_node = reduce(operator.getitem, key_levels[:-1], self)

                last_node[key_levels[-1]] = type(last_node[key_levels[-1]])(value)
            except Exception:
                raise KeyError(f"Could not set key {key} ")

    def __str__(self):
        return json.dumps(self, indent=4, sort_keys=True)
