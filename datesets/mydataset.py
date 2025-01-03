import cv2, torch, glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io
from .ext_transforms import get_transforms


class TrainDataset(Dataset):
    def __init__(self, graph_list=None, cfg=None, transforms=None, mode='train'):
        self.graph_list = graph_list
        # remove that one faulty image from train_csv
        self.mode = mode
        self.cfg = cfg
        self.transforms = transforms
        if cfg.train_dataset == "hap":
            prefix = "../input/hubmap/"
        elif cfg.train_dataset == "hubmap":
            prefix = "../hubmap/"
        else:
            prefix = "../all_256/"
        self.image_paths = [prefix + "train/" + i for i in graph_list]
        self.mask_paths = [prefix + "masks/" + i.replace("train", "mask") for i in graph_list]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = io.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if self.transforms:
            data = self.transforms(image=img, mask=mask)
            img = data['image']
            mask = data['mask']
        img = np.transpose(img, (2, 0, 1)) / 255.0
        return torch.tensor(img), torch.tensor(mask)


class DiceDataset(Dataset):
    def __init__(self, graph_list=None, cfg=None):
        self.graph_list = graph_list
        # remove that one faulty image from train_csv
        self.cfg = cfg
        if cfg.train_dataset == "hap":
            prefix = "../input/hubmap/"
        elif cfg.train_dataset == "hubmap":
            prefix = "../hubmap/"
        else:
            prefix = "../all_256/"
        self.image_paths = [prefix + "train/" + i for i in graph_list]
        self.mask_paths = [prefix + "masks/" + i.replace("train", "mask") for i in graph_list]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = io.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = np.transpose(img, (2, 0, 1)) / 255.0
        return torch.tensor(img), torch.tensor(mask)



def prepare_train_loaders(fold, df, cfg, debug=False):
    train_list = df[df['fold'] == fold].reset_index(drop=True)["graph_name"].values
    valid_list = df[df['fold'] != fold].reset_index(drop=True)["graph_name"].values

    if debug:
        train_list = train_list[:20]
        valid_list = valid_list[:20]

    train_dataset = TrainDataset(train_list, transforms=get_transforms(train=True, cfg=cfg), cfg=cfg, mode='train')
    valid_dataset = TrainDataset(valid_list, transforms=get_transforms(train=False, cfg=cfg), cfg=cfg, mode='valid')

    #     print(get_statistics(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_bs if not cfg.debug else 20,
                              num_workers=0, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.valid_bs if not cfg.debug else 20,
                              num_workers=0, shuffle=True, pin_memory=True)

    return train_loader, valid_loader



def prepare_valid_loaders(cfg):
    if cfg.train_dataset == "hap":
        prefix = "../input/hubmap/"
    elif cfg.train_dataset == "hubmap":
        prefix = "../hubmap/"
    else:
        prefix = "../all_256/"
    dice_graph_path_list = glob.glob(prefix + "train/*")
    dice_graph_name_list = [i[i.rindex("/") + 1:] for i in dice_graph_path_list]
    dice_dataset = DiceDataset(dice_graph_name_list, cfg=cfg)
    dice_loader = DataLoader(dice_dataset,  num_workers=0, shuffle=True, batch_size=1, pin_memory=True)
    return dice_loader