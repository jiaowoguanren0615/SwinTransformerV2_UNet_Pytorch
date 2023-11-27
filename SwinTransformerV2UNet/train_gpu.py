import argparse
import copy
import gc
import glob
import os
import random
import time
import cv2
from collections import defaultdict
import pandas as pd
from torch.optim import lr_scheduler
from .model_configs import CFG
from fastprogress import progress_bar
from matplotlib import pyplot as plt
from .engine import train_one_epoch, valid_one_epoch
from .datesets import prepare_valid_loaders, prepare_train_loaders
from sklearn.model_selection import KFold
from .utils import Dice_th_pred
from .models import *


def build_model(cfg):
    model = unet_swin(img_size=256, size=cfg.size, config=cfg)
    model.to(cfg.device)
    return model


def load_model(path, cfg=None):
    model = build_model(cfg)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def save_img(data, name, out):
    data = data.float().cpu().numpy()
    img = cv2.imencode('.png', (data * 255).astype(np.uint8))[1]
    out.writestr(name, img)


def fetch_scheduler(optimizer, cfg):
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max,
                                                   eta_min=cfg.min_lr)
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0,
                                                             eta_min=cfg.min_lr)
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=cfg.min_lr, )
    elif cfg.scheduer == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    else:
        scheduler = None

    return scheduler


class Model_pred:
    def __init__(self, model, dl, tta: bool = True, half: bool = False, config=None):
        self.model = model
        self.dl = dl
        self.tta = tta
        self.half = half
        self.config = config

    def __iter__(self):
        self.model.eval()
        name_list = self.dl.dataset.graph_list
        count = 0
        with torch.no_grad():
            for x, y in iter(self.dl):
                if self.config.device != "cpu":
                    x = x.to(self.config.device)
                if self.half:
                    x = x.half()
                x = x.type(torch.float)
                p = self.model(x)
                py = torch.sigmoid(p).detach()
                if self.tta:
                    # x,y,xy flips as TTA
                    flips = [[-1], [-2], [-2, -1]]
                    for f in flips:
                        p = self.model(torch.flip(x, f))
                        p = torch.flip(p, f)
                        py += torch.sigmoid(p).detach()
                    py /= (1 + len(flips))
                if y is not None and len(y.shape) == 4 and py.shape != y.shape:
                    py = F.upsample(py, size=(y.shape[-2], y.shape[-1]), mode="bilinear")
                py = py.permute(0, 2, 3, 1).float().cpu()
                batch_size = len(py)
                for i in range(batch_size):
                    taget = y[i].detach().cpu() if y is not None else None
                    yield py[i], taget, name_list[count]
                    count += 1

    def __len__(self):
        return len(self.dl.dataset)



def score(weight_path):
    score_lindex = weight_path.rindex("_") + 1
    score_rindex = weight_path.rindex(".")
    return float(weight_path[score_lindex:score_rindex])

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    print(f"Setting seed as {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


def initialise_config(debug=False, train_bs=4, fold=0, pretrained_path=None):
    cfg = CFG(fold=fold, train_bs=train_bs, debug=debug, pretrained_path=pretrained_path)
    set_seed(cfg.seed)
    return cfg


def create_folds(cfg=None):
    if cfg.train_dataset == "hap":
        image_name_list = [ i[i.rindex("/"):]for i in glob.glob("../input/hubmap-2022-256x256/train/*.png")]
    elif cfg.train_dataset == "hubmap":
        image_name_list = [i[i.rindex("/"):] for i in glob.glob("../hubmap-256x256/train/*.png")]
    else:
        image_name_list = [i[i.rindex("/"):] for i in glob.glob("../all_256/train/*.png")]

    df = pd.DataFrame({"graph_name":image_name_list})
    skf = KFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    for fold, idxes in enumerate(skf.split(range(len(df)))):
        df.loc[idxes[1], 'fold'] = fold
    return df


def run_training(model, optimizer, scheduler, device, num_epochs, fold, train_loader, valid_loader, cfg):
    if device != "cpu":
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = 0
    history = defaultdict(list)

    if cfg.only_dice != 1:
        for epoch in range(2):

            print(f'Epoch {epoch}/{num_epochs}', end='')
            train_loss, train_score = train_one_epoch(model, optimizer, scheduler,
                                                      dataloader=train_loader,
                                                      device=cfg.device, epoch=epoch, cfg=cfg)

            val_loss, val_score = valid_one_epoch(model, valid_loader,
                                                  device=cfg.device,
                                                  epoch=epoch,
                                                  optimizer=optimizer, cfg=cfg)

            history['epoch'].append(epoch)
            history['Train Loss'].append(train_loss)
            history['Valid Loss'].append(val_loss)
            history['Valid Scores'].append(val_score)

            print(f'Train Loss: {train_loss} | Valid Loss: {val_loss}')
            print(f'Train Score: {train_score} | Valid Dice Score: {val_score}')

            # deep copy the model
            if val_score >= best_dice:
                os.system(f"rm models/fold_{fold}/{cfg.size}_*")

                print(f"Valid Score Improved ({best_dice:0.4f} ---> {val_score:0.4f})")
                best_dice = val_score
                best_model_wts = copy.deepcopy(model.state_dict())
                PATH = f"Models/fold_{fold}/{cfg.size}_{val_score:0.4f}.pth"
                torch.save(model.state_dict(), PATH)

                print(f"Model Saved")


        end = time.time()
        time_elapsed = end - start
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))

        model.load_state_dict(best_model_wts)

        plt.subplot(1, 2, 1, frameon=False)
        plt.title(f'fold_{fold}_train_loss')
        plt.xlabel('Epoch')
        plt.plot(history['epoch'], history['Train Loss'], "r")

        plt.subplot(1, 2, 2, frameon=False)
        plt.title(f'fold_{fold}_test_dice')
        plt.xlabel('Epoch')
        plt.plot(history['epoch'], history['Valid Scores'], "b")

        plt.savefig(f"Models/fold_{fold}/metric_fold_{fold}.jpg")
        plt.close()

    dice_loader = prepare_valid_loaders(cfg)
    mp = Model_pred(model, dice_loader, config=cfg)
    dice = Dice_th_pred(np.arange(0.2, 0.7, 0.01))
    for p in progress_bar(mp):
        dice.accumulate(p[0], p[1])
    # save_img(p[0], p[2], out)
    gc.collect()
    dices = dice.value
    noise_ths = dice.ths
    best_dice = dices.max()
    best_thr = noise_ths[dices.argmax()]
    plt.figure(figsize=(8, 4))
    plt.plot(noise_ths, dices, color='blue')
    plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max(), colors='black')
    d = dices.max() - dices.min()
    plt.text(noise_ths[-1] - 0.1, best_dice - 0.1 * d, f'DICE = {best_dice:.3f}', fontsize=12)
    plt.text(noise_ths[-1] - 0.1, best_dice - 0.2 * d, f'TH = {best_thr:.3f}', fontsize=12)
    plt.savefig(f'models/fold_{fold}/save.jpg')
    plt.close()

    weight_path = glob.glob(f"Models/fold_{fold}/{cfg.size}*.pth")[0]
    down_index = weight_path.rindex("_")
    new_weight_path = weight_path[:down_index] + f"_{best_thr:.3f}" + weight_path[down_index:]
    os.rename(weight_path, new_weight_path)

    return model, history


def main(cfg):
    cfg.display()
    print(f'#' * 30)
    print(f'### Fold: {cfg.fold}')
    print(f'#' * 30)

    train_loader, valid_loader = prepare_train_loaders(fold=cfg.fold,
                                                       df=create_folds(cfg),
                                                       debug=cfg.debug,
                                                       cfg=cfg)

    if cfg.load_best_model:
        models = glob.glob(f"Models/fold_{cfg.fold}/{cfg.size}_*.pth")
        models = sorted(models, key=lambda i: score(i), reverse=True)
        model = load_model(models[0], cfg=cfg).to(cfg.device)
        print("Load Pretrained Model: " + models[0])
    elif cfg.pretrained_path is None:
        model = unet_swin(img_size=256, size=cfg.size, config=cfg).to(cfg.device)
    else:
        model = load_model(cfg.pretrained_path, cfg=cfg).to(cfg.device)
        print("Load pretrained Model: " + cfg.pretrained_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = fetch_scheduler(optimizer, cfg=cfg)
    run_training(model, optimizer, scheduler,
                 device=cfg.device,
                 num_epochs=cfg.epochs, fold=cfg.fold,
                 train_loader=train_loader,
                 valid_loader=valid_loader,
                 cfg=cfg)


if __name__ == '__main__':
    cfg = initialise_config(train_bs=8, fold=0)
    if not os.path.exists('./Models'):
        os.makedirs('./Models')
    if not os.path.exists(f"Models/fold_{cfg.fold}"):
        os.mkdir(f"Models/fold_{cfg.fold}")
    main(cfg)
