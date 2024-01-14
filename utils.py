import os
import shutil
import random
import pickle
import logging
from pathlib import Path

from constants import CLASS_NAMES_10, CLASS_NAMES_15, CLASS_NAMES_21


def set_global_log_level(level):
    LOG_LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    if isinstance(level, str):
        level = level.strip().lower()
        level = LOG_LEVEL_MAP[level]
    logging.getLogger().setLevel(level)
    logging.StreamHandler().setLevel(level)


def get_logger(name):
    """
    Get a logger with the given name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():  # Only if logger has not been set up before
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def split_dataset(data_list, tables_dir, include_test=True, seed=57):
    """
    Splits a dataset into "train", "validation" and "test".

    Args:
        data_list (list): List of rows, each element is an instance-dict.
        tables_dir (str): The path to save the data-tables.
        include_test (bool): If True, will split in train, val and test. If False, will split in train and val.
        seed (int, optional): Seed for the rng. Defaults to 57.
    """
    random.seed(seed)
    n_images = len(data_list)
    random.shuffle(data_list)
    if include_test:
        train_size = int(0.5 * n_images)
        val_size = int(0.3 * n_images)
    else:
        train_size = int(0.6 * n_images)

    train_data = data_list[: train_size]
    if include_test:
        val_data = data_list[train_size: train_size + val_size]
        test_data = data_list[train_size + val_size:]
    else:
        val_data = data_list[train_size:]

    if os.path.exists(tables_dir):
        shutil.rmtree(tables_dir)  # Delete previous folder and re-create
    os.makedirs(tables_dir)
    with open(Path(tables_dir) / "train_data.pkl", "wb") as outfile:
        pickle.dump(train_data, outfile)
    with open(Path(tables_dir) / "val_data.pkl", "wb") as outfile:
        pickle.dump(val_data, outfile)
    if include_test:
        with open(Path(tables_dir) / "test_data.pkl", "wb") as outfile:
            pickle.dump(test_data, outfile)


def get_class_names_and_combinations(n_classes):
    """
    Given the amount of classes, returns the class names and shape combinations.

    Args:
        n_classes (int): The amount of classes. Must be in [10, 15, 21].

    Raises:
        ValueError: If `n_classes` is not in [10, 15, 21].

    Returns:
        list of str: list of the string names of the classes.
        list of list: List of the corresponding figures to generate for each class.
    """
    if n_classes == 10:
        class_names = CLASS_NAMES_10
    elif n_classes == 15:
        class_names = CLASS_NAMES_15
    elif n_classes == 21:
        class_names = CLASS_NAMES_21
    else:
        message = "Current implementation of ConceptShapes are only compatible with 10, 15 or 21 classes. "
        message += f"Got {n_classes=}"
        raise ValueError(message)

    shape_combinations = []
    for shape_name in class_names:
        shape_combinations.append([shape_name.split("_")[0], shape_name.split("_")[1]])

    return class_names, shape_combinations
