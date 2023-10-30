from tqdm import tqdm
import torch
from torch.cuda import amp
import gc
from utils.losses import DiceScore, DiceBCELoss


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch, cfg):
    model.eval()
    total_val_loss = 0
    total_val_score = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')

    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        y_pred = model(images)
        criterion = DiceBCELoss()
        loss = criterion(y_pred, masks)
        dice_score = DiceScore()(y_pred, masks).detach().item()

        loss = loss.detach().item()
        total_val_loss += loss
        total_val_score += dice_score

    print(f'\nTesting epoch {epoch} ')
    print(f'Total DiceBCE loss: {total_val_loss / len(dataloader):.4f}')
    print(f'Total average Dice Score: {total_val_score / len(dataloader):.4f}')

    torch.cuda.empty_cache()
    gc.collect()

    return total_val_loss / len(dataloader), total_val_score / len(dataloader)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, cfg):

    model.train()
    scaler = amp.GradScaler()
    total_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ', leave=False)
    data_size = 0
    total_dice_score = 0

    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)
        data_size += batch_size

        with amp.autocast(enabled=True):
            y_pred = model(images)
            criterion = DiceBCELoss()
            loss = criterion(y_pred, masks)
            dice_score = DiceScore()(y_pred, masks).detach().item()

        scaler.scale(loss / cfg.n_accumulate).backward()

        if ((step + 1) % cfg.n_accumulate == 0 or (step + 1) == len(dataloader)):

            scaler.step(optimizer)
            scaler.update()
            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        loss = loss.detach().item()
        total_loss += loss
        total_dice_score += dice_score

        pbar.set_postfix(desc=f'Loss={loss:.4f} DiceScore= {dice_score:.4f}  Batch_id={step}')

    print(f'\nTraining epoch {epoch} ')
    print(f'Total DiceBCE loss: {total_loss / len(dataloader):.4f}')
    print(f'Total average Dice Score: {total_dice_score / len(dataloader):.4f}')

    torch.cuda.empty_cache()
    gc.collect()

    return (total_loss / len(dataloader), total_dice_score / len(dataloader))