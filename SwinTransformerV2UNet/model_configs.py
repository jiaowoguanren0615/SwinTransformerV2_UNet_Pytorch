import torch


class CFG:
    def __init__(self, fold=0., seed=2020, train_bs=32, debug=False, pretrained_path=None):
        self.n_fold = 5
        self.fold = fold
        self.seed = seed
        self.train_bs = train_bs
        self.valid_bs = train_bs  # may need to change this
        self.debug = debug
        self.img_size = [256, 256]
        self.exp_name = 'Hubmap256-training'
        self.epochs = 1000
        self.lr = 2e-3
        self.pretrained_path = pretrained_path
        self.scheduler = "CosineAnnealingLR"
        self.min_lr = 2e-4
        self.T_max = int(30000 / self.train_bs * self.epochs) + 50
        self.T_0 = 25
        self.warmup_epochs = 10
        self.wd = 1e-6
        self.n_accumulate = max(1, 32 // self.train_bs)
        self.num_classes = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.backbone = "swinunet"
        self.model_name = "swinunet"
        self.model_urls = {
            "swinv2_tiny_window16_256": "../input/swinv2w/swinv2_tiny_patch4_window8_256.pth",
            "swinv2_small_window8_256": "../input/swinv2w/swinv2_small_patch4_window8_256.pth",
            "swinv2_small_window16_256": "../input/swinv2w/swinv2_small_patch4_window16_256.pth",
            "swinv2_base_window16_256": "../input/swinv2w/swinv2_base_patch4_window16_256.pth",
        }

        self.size = "swinv2_base_window16_256"

        self.load_best_model = False

        self.train_dataset = "hap"  # only all, hap, hubmap

        self.dice_dataset = "hap"  # only all, hap, hubmap

        self.only_dice = 0

    def display(self):
        print(f"{self.exp_name}")
        print(f"debug is {self.debug}")
        print(f"seed is {self.seed}")
        print(f"train_bs is {self.train_bs}")
        print(f"img_size is {self.img_size}")
        print(f"fold_no is {self.fold}")
        print(f"backbone is {self.backbone}")
        print(f"epochs is {self.epochs}")
        print(f"Pretrained: {self.pretrained_path}")