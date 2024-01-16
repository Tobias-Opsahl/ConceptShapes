import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils import load_data_list


class ShapesDataset(Dataset):
    """
    Dataset for shapes dataset. The `shapes` datasets are created in `make_shapes_datasets.py`.
    """

    def __init__(self, data_list, transform=None):
        """

        Args:
            data_list (str): The path to the list of dictionaries with table-data.
            transform (torch.transform, optional): Optional transformation on the data.
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]["img_path"]
        class_label = self.data_list[idx]["class_label"]
        # Convert from dict to list to tensor
        attribute_label = torch.tensor(list(self.data_list[idx]["attribute_label"].values())).to(torch.float32)

        with Image.open(img_path) as img:  # Open and close properly
            image = img.convert("RGB")  # Images have alpa channel

        if self.transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.ToTensor()(image)

        return image, class_label, attribute_label, img_path


def get_transforms():
    """
    Transforms for shapes dataset. No color changing or flips are done, since these can be meaningful
    concepts in the data. The images are only turned into tensors and normalized.
    Note that there is not really a need for data augmentation, since more data can be created.

    Returns:
        torchvision.tranforms.Compose: The transforms.
    """
    normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Turns image into [0, 1] float tensor
        normalize,
    ])
    return transform


def normalize_conceptshapes(input_image, mean=0.5, std=0.5):
    """
    Normalizes an input images, with the normalization that was done with the Shapes datasets.

    Args:
        input_image (Tensor): The input image to normalise.
        mean (float, optional): The mean of the normalisation. Defaults to 0.5.
        std (int, optional): The standard deviation of the normalisation. Defaults to 0.5.

    Returns:
        Tensor: The normalised image.
    """
    normalize = torchvision.transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])
    normalized = normalize(input_image)
    return normalized


def denormalize_conceptshapes(input_image, mean=0.5, std=0.5):
    """
    Denormalizes an input images, with the normalization parameters that was done with the Shapes datasets.

    Args:
        input_image (Tensor): The input image to denormalise.
        mean (float, optional): The mean of the normalisation. Defaults to 0.5.
        std (int, optional): The standard deviation of the normalisation. Defaults to 0.5.

    Returns:
        Tensor: The denormalised image.
    """
    new_mean = - mean / std
    new_std = 1 / std
    denormalize = torchvision.transforms.Normalize(mean=[new_mean, new_mean, new_mean], std=[new_std, new_std, new_std])
    denormalized = denormalize(input_image)
    return denormalized


def load_data_conceptshapes(n_classes, n_attr, signal_strength, n_subset=None, n_images_class=1000, mode="train-val",
                            batch_size=4, shuffle=True, drop_last=False, num_workers=0, pin_memory=False,
                            persistent_workers=False):
    """
    Makes dataloaders for the Shapes dataset.
    Finds correct path based on `n_classes`, `n_attr`, `signal_strength` and `n_subset`.

    Will either just load "train", "val" or "test" loader (depending on argument `mode`),
    or a list of all of them if `mode == "all"`.

    Args:
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amonut of attributes (concepts) in the dataset.
        signal_strength (int): Signal-strength used to make dataset.
        n_subset (int): The amount of images per class to load. Set to `None` for full dataset.
        n_images_class (int): The amount of images used in each class in the dataset.
        mode (str, optional): The datasetmode to loader. If "all", will return a tuple of (train, val, test).
        batch_size (int, optional): Batch size to use. Defaults to 4.
        shuffle (bool, optional): Determines wether the sampler will shuffle or not.
            It is recommended to use `True` for training and `False` for validating.
        drop_last (bool, optional): Determines wether the last iteration of an epoch
            is dropped when the amount of elements is less than `batch_size`.
        num_workers (int): The amount of subprocesses used to load the data from disk to RAM.
            0, default, means that it will run as main process.
        pin_memory (bool): Whether or not to pin RAM memory (make it non-pagable).
            This can increase loading speed from RAM to VRAM (when using `to("cuda:0")`,
            but also increases the amount of RAM necessary to run the job. Should only
            be used with GPU training.
        persistent_workers (bool): If `True`, will not shut down workers between epochs.

    Returns:
        Dataloader: The dataloader, or the list of the two or three dataloaders.
    """
    if mode.lower() == "all":
        modes = ["train", "val", "test"]
    elif mode.lower() in ["train-val", "train val", "train-validation" "train validation"]:
        modes = ["train", "val"]
    else:
        modes = [mode]
    dataloaders = []
    for mode in modes:
        data_list = load_data_list(n_classes, n_attr, signal_strength, n_images_class=n_images_class,
                                   n_subset=n_subset, mode=mode)
        transform = get_transforms()
        dataset = ShapesDataset(data_list, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        dataloaders.append(dataloader)

    if len(dataloaders) == 1:  # Just return the datalaoder, not list
        return dataloaders[0]
    return dataloaders  # List of (train, val, test) dataloaders
