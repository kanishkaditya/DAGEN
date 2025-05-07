import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models import DAGEN
from datamodules import DAGENDataModule

# === Your plug-and-play components ===
from your_code import MyBackbone, MyGazeHead, SourceDataset, TargetDataset

backbone = MyBackbone()     # φ
gaze_head = MyGazeHead()    # h
src_ds = SourceDataset(...) # returns (img, gaze_xy)
tgt_ds = TargetDataset(...) # returns (img, gaze_xy) – gaze_xy only for neighbor search

dm = DAGENDataModule(src_ds, tgt_ds, batch_size=64, num_workers=8)

model = DAGEN(backbone, gaze_head,
              pretrain_epochs=5,
              k=5, mu=0.1, lambda_llr=1e-3,
              lambda_epc=1.0, lambda_gaze=1.0,
              lr=1e-3)

checkpoint_cb = ModelCheckpoint(monitor="val/mse", mode="min", save_top_k=3)
lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
    max_epochs=50,
    gpus=1,
    callbacks=[checkpoint_cb, lr_monitor],
)

# Run both stages in one call
trainer.fit(model, datamodule=dm)
