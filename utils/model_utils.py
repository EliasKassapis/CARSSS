from utils.constants import *
import importlib
import os
from pathlib import Path

LOSS_DIR = "losses"
GEN_DIR = "generators"
DIS_DIR = "discriminators"

types = [DIS_DIR, GEN_DIR, LOSS_DIR]
models = {x: {} for x in types}

def _read_all_classnames():
    for typ in types:
        for name in os.listdir(f"./models/{typ}"):
            if (not "__" in name):
                if os.path.isdir(f"./models/{typ}/{name}"):
                    for name2 in os.listdir(f"./models/{typ}/{name}"):
                        if (not "__" in name2):
                            current_directory = f"models.{typ}.{name}"
                            short_name = name2.split(".")[0]
                            module = importlib.import_module(f"{current_directory}.{short_name}")
                            class_reference = getattr(module, short_name)
                            models[typ][short_name] = class_reference
                else:
                        current_directory = f"models.{typ}"
                        short_name = name.split(".")[0]
                        module = importlib.import_module(f"{current_directory}.{short_name}")
                        class_reference = getattr(module, short_name)
                        models[typ][short_name] = class_reference

def find_model(type, name, **kwargs):
    """
    returns model with arguments given a string name-tag
    """

    return models[type][name](**kwargs)


def save_models(discriminator, generator, calibration_net, suffix):
    """
    Saves current state of models
    """
    save_dict = {"discriminator": discriminator.state_dict(), "generator": generator.state_dict(), "calibration_net": calibration_net.state_dict()}

    DATA_MANAGER.save_python_obj(save_dict, f"{DATA_MANAGER.stamp}/{MODELS_DIR}/{suffix}")


def load_models_and_state(discriminator, generator, calibration_net, models_to_load, suffix, stamp):
    """
    Loads saved models given a suffix and then also loads in state dicts already
    """
    assert Path(f"./results/output/{stamp}/{MODELS_DIR}").exists(), f"{stamp}/{MODELS_DIR} does not exist"

    models = DATA_MANAGER.load_python_obj(f"{stamp}/{MODELS_DIR}/{suffix}")

    if "calibration_net" in models_to_load:
        calibration_net.load_state_dict(models["prior"])
        calibration_net.to(DEVICE)

    if "generator" in models_to_load:
        generator.load_state_dict(models["generator"])
        generator.to(DEVICE)

    if "discriminator" in models_to_load:
        discriminator.load_state_dict(models["discriminator"])
        discriminator.to(DEVICE)


# needed to load in class references
_read_all_classnames()
