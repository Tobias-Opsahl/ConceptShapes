import argparse

from conceptshapes_datasets import make_specific_shapes_dataset
from utils import get_logger, set_global_log_level, parse_int_list


def parse_my_args():
    """
    Function for ascing for and returning all of the arguments from user.
    Remember that path variables are saved in `constants.py`.

    Returns:
        args: The argparse arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, choices=[10, 15, 21], required=True,
                        help="Number of classes (must be in [10, 15, 21]).")
    parser.add_argument("--n_concepts", type=int, choices=[5, 9], required=True,
                        help="Number of concepts/attributes (must be in [5, 9]).")
    parser.add_argument("--signal_strength", type=float, required=True,
                        help="Signal strength (must be from 0.5, to 1).")
    parser.add_argument("--n_images_class", type=int, default=1,  # TODO: Change to 1000
                        help="Number of images per class (integer).")
    parser.add_argument("--do_not_add_descriptions", action="store_true",
                        help="Do not add descriptions for classes and concepts. ")
    parser.add_argument("--subsets_to_add", type=parse_int_list, default=0,
                        help="List of subsets to make (list of ints, for example `50,100,150,200,250`).")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_global_log_level("debug")
    logger = get_logger(__name__)
    args = parse_my_args()
    if (args.subsets_to_add) == 0 and args.n_images_class >= 500:  # Default subset sizes
        args.subsets_to_add = [50, 100, 150, 200, 250]
    elif args.subsets_to_add == 0:
        args.subsets_to_add = None
    if args.signal_strength > 1 or args.signal_strength < 0:
        raise ValueError(f"Argument `signal_strength` must be in the interval [0.5, 1]. Was {args.signal_strength}")

    # Call the function that makes the dataset.
    make_specific_shapes_dataset(
        n_classes=args.n_classes, n_attr=args.n_concepts, signal_strength=args.signal_strength,
        n_images_class=args.n_images_class, add_descriptions=(not args.do_not_add_descriptions),
        subsets_to_add=args.subsets_to_add)
