import os
import shutil
import random
import pickle
import logging
from pathlib import Path

from constants import CLASS_NAMES_10, CLASS_NAMES_15, CLASS_NAMES_21, DATA_ROOT_DIR, CONCEPTSHAPES_DIR, TABLES_DIR


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


def _check_just_file(filename):
    """
    Checks that `filename` does not contain a folder, for example `plots/plot.png`. Raises ValueError if it does.
    Also checks that `filename` is either `str` or `pathtlib.Path`.

    Args:
        filename (str or pathlib.Path): The filename to check.

    Raises:
        ValueError: If `filename` contains a folder.
        TypeError: If `filename` is not of type `str` or `pathlib.Path`.
    """
    message = f"Filename must not be inside a directory, but only be the filename. Was {filename}. "
    if isinstance(filename, Path):
        filename = filename.as_posix()  # Convert to string
    if not isinstance(filename, str):
        raise TypeError(f"Filename must be of type `str` or `pathlib.Path`. Was {type(filename)}. ")
    if "/" in filename or "\\" in filename:
        raise ValueError(message)


def create_folder(folder_path, exist_ok=True):
    """
    Create a folder.

    Args:
        folder_path (str or pathlib.Path): The folder-path, including the foldername.
        exist_ok (bool, optional): If True, will not raise Exception if folder already exists. Defaults to True.
    """
    os.makedirs(folder_path, exist_ok=exist_ok)


def make_file_path(folder, filename, check_folder_exists=True):
    """
    Merges a path to a folder `folder` with a filename `filename`.
    If `check_folder_exists` is True, will create the folder `folder` if it is not there.
    Argument `filename` can not be inside a folder, it can only be a filename (to ensure that the correct and full
    folder path gets created).

    Args:
        folder (str or pathlib.Path): Path to the folder.
        filename (str or pathlib.Path): Filename of the file. Must not be inside a folder, for example `plots/plot.png`
            is not allowed, `plots/` should be a part of `folder`.
        check_folder_exists (bool): If `True`, will check that `folder` exists, and create it if it does not.

    Returns:
        pathlib.Path: The merged path.
    """
    _check_just_file(filename)  # Check that filename does not have a folder.
    folder_path = Path(folder)
    if check_folder_exists:
        create_folder(folder_path, exist_ok=True)
    file_path = folder_path / filename
    return file_path


def load_data_list(n_classes, n_attr, signal_strength, n_subset=None, n_images_class=None, mode="train"):
    """
    Loads a data-list, a list of dictionaries saved as a pickle file with labels and images-paths for the dataset.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_subset (int, optional): The subset of data to load data-list for. If `None`, will use full dataset.
        n_images_class (int): Amount of images per class. Do not need to be provided if one uses the most
            common variants of datasets, which is 1k for all datasets exepct c10, a9 and s98, which is 2k.
        mode (str, optional): Should be in ["train", "val", "test"]. Determines which
            data-list to load. Defaults to "train".

    Returns:
        list: The data list
    """
    dataset_path = get_shapes_dataset_path(n_classes, n_attr, signal_strength, n_images_class=n_images_class)
    filename = Path(f"{mode}_data.pkl")
    if n_subset is not None:
        data_list_path = dataset_path / TABLES_DIR / f"sub{n_subset}" / filename
    else:  # Load the full dataset-data-list
        data_list_path = dataset_path / TABLES_DIR / filename
    try:
        data_list = pickle.load(open(data_list_path, "rb"))
    except Exception:
        message = f"Data-list {data_list_path} not found. Try creating datalist with make_subset_shapes(), "
        message += "or create datasets with make_shapes_datasets.py. "
        raise FileNotFoundError(message)
    return data_list


def get_shapes_dataset_path(n_classes, n_attr, signal_strength, n_images_class=None, check_already_exists=True):
    """
    Given classes, attributes and the signal-strength, makes the path to a shapes-dataset.
    This does not create the dataset, so it is assumed that it is already created (with `make_shapes_datasets.py`).
    Datasets should be stored in "data/shapes".
    Names of the dataset are on the form "shapes_1k_c10_a5_s100" for 1k images in each class, 10 classes, 5 attributes
    and signal-strength 100. Note that all datasets has 1k images in each class, except from 10-classes 5-attributes
    signal-strenght 98, which has 2k. Also note that datasets with signal-strength 98 will omit the signal-strength
    in the name, for example "shapes_1k_c21_a9" for 21 classes, 9 attributes and signal_strength 98.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_images_class (int): Amount of images per class. Do not need to be provided if one uses the most
            common variants of datasets, which is 1k for all datasets exepct c10, a9 and s98, which is 2k.
        check_already_exists (bool, optional): If True, will raise ValueError if dataset does not exist.
            Defaults to True.

    Raises:
        ValueError: If `check_already_exists` is True and dataset_path does not exit.

    Returns:
        pathlib.Path: Path to the dataset.
    """
    base_path = Path(DATA_ROOT_DIR) / CONCEPTSHAPES_DIR
    if n_images_class is not None:
        base_folder_name = Path(f"shapes_{round(n_images_class / 1000)}k_")
    elif n_classes == 10 and n_attr == 5 and signal_strength == 98:
        base_folder_name = Path("shapes_2k_")  # Dataset c10_a5 (s98) has 2k images of each class. The rest has 1k.
    else:  # Assume 1k if it is not specified
        base_folder_name = Path("shapes_1k_")

    signal_string = f"_s{int(round(signal_strength * 100))}"
    folder_name = Path(f"{base_folder_name}c{n_classes}_a{n_attr}{signal_string}/")
    dataset_path = base_path / folder_name
    if check_already_exists:
        if not dataset_path.exists():
            message = f"Path {dataset_path} for ConceptShapes dataset for {n_classes=}, {n_attr=}, {signal_strength=} "
            message += " does not exist. Check for typo or create the dataset with `make_shapes_datasets.py.` "
            raise ValueError(message)
    return dataset_path


def write_data_list(n_classes, n_attr, signal_strength, n_subset, train_data, val_data, n_images_class=None):
    """
    Given a train and validation data-list for a subset of a shapes-dataset, saves or overwrites the data-lists.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int, optional): Signal-strength used to make dataset.
        n_subset (int, optional): The subset of data to load data-list for. If `None`, will use full dataset.
        train_data (list): The list of the train-data dictionaries.
        val_data (list): The list of the validation-data dictionaries.
        n_images_class (int): Amount of images per class. Do not need to be provided if one uses the most
            common variants of datasets, which is 1k for all datasets exepct c10, a9 and s98, which is 2k.

    Returns:
        list: The data list
    """
    dataset_path = get_shapes_dataset_path(n_classes, n_attr, signal_strength, n_images_class)
    tables_path = dataset_path / "tables" / f"sub{n_subset}"
    if os.path.exists(tables_path):
        shutil.rmtree(tables_path)  # Delete previous folder and re-create
    os.makedirs(tables_path)
    train_filename = make_file_path(tables_path, "train_data.pkl")
    val_filename = make_file_path(tables_path, "val_data.pkl")
    with open(train_filename, "wb") as outfile:
        pickle.dump(train_data, outfile)
    with open(val_filename, "wb") as outfile:
        pickle.dump(val_data, outfile)
