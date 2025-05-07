import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class DAGEN(pl.LightningModule):
    def __init__(self, backbone, gaze_head,
                 pretrain_epochs=10,
                 k=5, mu=0.1, lambda_llr=0.001,
                 lambda_epc=1.0, lambda_gaze=1.0,
                 lr=1e-3):
        """
        backbone: φ(I) → embedding (B×D)
        gaze_head: h(φ) → gaze pred (B×2)
        """
        super().__init__()
        self.save_hyperparameters()

        self.backbone = backbone
        self.gaze_head = gaze_head

    def forward(self, x):
        feat = self.backbone(x)
        pred = self.gaze_head(feat)
        return feat, pred

    def gaze_loss(self, pred, gt):
        # Eq. (10): arccos cosine similarity
        cos = F.cosine_similarity(pred, gt, dim=1).clamp(-1+1e-6,1-1e-6)
        return torch.acos(cos).mean()

    def llr_weights(self, gt_src, pred_tgt):
        # gt_src: (Bs×2), pred_tgt: (2,)
        diff = pred_tgt.unsqueeze(0) - gt_src  # (Bs×2)
        # select neighbors within mu
        mask = (diff.abs().max(dim=1).values < self.hparams.mu)
        idx = torch.where(mask)[0]
        if idx.numel() < self.hparams.k:
            return None, None  # skip sample
        idx = idx[torch.randperm(idx.numel())[:self.hparams.k]]
        D = diff[idx]  # (k×2)
        S = D @ D.T  # (k×k)
        A = S + self.hparams.lambda_llr * torch.eye(self.hparams.k, device=S.device)
        invA = torch.inverse(A)
        ones = torch.ones(self.hparams.k, device=S.device)
        w = invA @ ones
        w = w / (ones @ invA @ ones)
        return idx, w  # both length k

    def epc_loss(self, feat_src, gt_src, feat_tgt, pred_tgt):
        losses = []
        for j in range(feat_tgt.size(0)):
            idx, w = self.llr_weights(gt_src, pred_tgt[j])
            if idx is None:
                continue
            hypo = (w.unsqueeze(1) * feat_src[idx]).sum(dim=0)
            losses.append(F.l1_loss(feat_tgt[j], hypo, reduction="mean"))
        return torch.stack(losses).mean() if losses else torch.tensor(0., device=self.device)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # Determine stage by epoch
        if self.current_epoch < self.hparams.pretrain_epochs:
            # Stage 1: pre-train on source only
            (img_s, gt_s) = batch
            feat_s, pred_s = self(img_s)
            loss = self.gaze_loss(pred_s, gt_s)
            self.log("train/pre_gaze_loss", loss, on_step=True, on_epoch=True)
            return loss

        # Stage 2: joint adaptation
        (img_s, gt_s), (img_t, gt_t) = batch
        feat_s, pred_s = self(img_s)
        feat_t, pred_t = self(img_t)

        L_gaze = self.gaze_loss(pred_s, gt_s)
        L_epc  = self.epc_loss(feat_s, gt_s, feat_t, pred_t)

        loss = self.hparams.lambda_gaze * L_gaze + self.hparams.lambda_epc * L_epc
        self.log_dict({
            "train/gaze_loss": L_gaze,
            "train/epc_loss":  L_epc,
            "train/total_loss": loss
        }, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch
        _, pred = self(img)
        val_loss = F.mse_loss(pred, gt)
        self.log("val/mse", val_loss, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),
                               lr=self.hparams.lr,
                               momentum=0.9,
                               weight_decay=5e-4)
