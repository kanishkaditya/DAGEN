import pytorch_lightning as pl
from torch.utils.data import DataLoader

class DAGENDataModule(pl.LightningDataModule):
    def __init__(self, src_dataset, tgt_dataset, batch_size=64, num_workers=4):
        """
        src_dataset: returns (img, gaze_xy) with ground truth
        tgt_dataset: returns (img, gaze_xy) but gaze_xy only used for LLR neighbor selection
        """
        super().__init__()
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        # During stage 2, zip source/target for joint training
        src_loader = DataLoader(self.src_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.num_workers)
        tgt_loader = DataLoader(self.tgt_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.num_workers)
        return zip(src_loader, tgt_loader)

    def pretrain_dataloader(self):
        # Stage 1: source-only
        return DataLoader(self.src_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.tgt_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
