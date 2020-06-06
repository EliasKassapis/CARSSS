import inspect
from utils.constants import *
from utils.training_helpers import instance_checker

"""
Code adapted from https://github.com/StijnVerdenius/DeepFaceImageSynthesis
"""

def setup_directories():
    stamp = DATA_MANAGER.stamp
    dirs = OUTPUT_DIRS
    for dir_to_be in dirs:
        DATA_MANAGER.create_dir(f"{stamp}/{dir_to_be}")

def assert_type(expectedType, content):
    """ makes sure type is respected"""
    assert instance_checker(content, expectedType), f"{content} is not {expectedType}"

def assert_non_empty(content):
    """ makes sure not None or len()==0 """

    func = inspect.stack()[1][3]
    assert not content == None, "Content is null in {}".format(func)
    if (type(content) is list or type(content) == str):
        assert len(content) > 0, "Empty {} in {}".format(type(content), func)


def get_loss_weights(arguments):
    """ returns a dictionary with the right loss weights given parsed arguments """

    default_returnvalue = {key: value for key, value in arguments.__dict__.items() if
                           (("weight" in key) and (not value == -1))}
    if (not arguments.loss_gen == TOTAL_G_LOSS):
        for key in default_returnvalue:
            if (not arguments.loss_gen in key):
                default_returnvalue[key] = 0.0

    print(default_returnvalue)
    return default_returnvalue
