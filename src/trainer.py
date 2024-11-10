from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os

class Trainer:
    def __init__(self, ckpt_path="checkpoints/", device="cuda", dtype=torch.float32):
        self.ckpt_path = ckpt_path
        self.device = device
        self.dtype = dtype

    def config_trainer(self, model, optimizer, wandb_logger):
        self.model = model
        self.optimizer = optimizer
        self.wandb_logger = wandb_logger

    def _delete_older_checkpoints(self):
        checkpoints_dir = os.path.join(self.ckpt_path, "checkpoints")
        checkpoints_files_list = os.listdir(checkpoints_dir)
        # Ignore backup files from list
        checkpoints_files_list = filter(lambda f: not f.endswith("_bkup.ckpt"), checkpoints_files_list)
        # Sort by global step (descending)
        checkpoints_files_list = sorted(checkpoints_files_list, key=lambda f: int(f.split('-')[1][6:]), reverse=True)

        # Delete older checkpoints in the dir, leaving only
        # the backups and the most recent checkpoint
        for filename in checkpoints_files_list[1:]:
            os.remove(os.path.join(self.ckpt_path, "checkpoints", filename))


    def save_model(self,):
        self._delete_older_checkpoints()

        torch.save({
            'epoch': self.current_epoch,
            'vit_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train/loss': self._train_loss,
            'val/loss': self._val_loss,
        }, os.path.join(self.ckpt_path, "checkpoints", f"epoch={self.current_epoch}-vloss{'{:.2f}'.format(self._val_loss)}.ckpt"))


    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")

        self.model.load_state_dict(checkpoint["vit_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self._train_loss = checkpoint["train/loss"]
        self._val_loss = checkpoint["val/loss"]

        print("Loaded: ", self.epoch, self._train_loss, self._val_loss)


    def shared_step(self, batch, margin: float = 0.0) -> float:
        face1, face2, stranger = batch["face1"], batch["face2"], batch["stranger"]
        
        face1 = face1.to(device=self.device, dtype=self.dtype)
        face2 = face2.to(device=self.device, dtype=self.dtype)
        stranger = stranger.to(device=self.device, dtype=self.dtype)

        face1_outputs = self.model(face1)
        face2_outputs = self.model(face2)
        stranger_outputs = self.model(stranger)

        positive_distance = (face1_outputs - face2_outputs) ** 2
        negative_distance = (face1_outputs - stranger_outputs) ** 2

        loss = (margin + positive_distance - negative_distance).mean()

        return loss
    
    def validation_step(self, val_loader: DataLoader, margin: int = 0) -> float:
        val_loss = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                loss = self.shared_step(data, margin)
                val_loss.append(loss.detach().cpu())

        val_loss = sum(val_loss) / len(val_loss)

        return val_loss

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            margin: float = 0.0
        ):
        for e in range(epochs):
            self.current_epoch = e
            progress_bar = tqdm(total=len(train_loader))
            progress_bar.set_description(f"Epoch {e}")
            train_losses = []
            self.wandb_logger.log({"epoch": e})

            for i, data in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                loss = self.shared_step(data, margin=margin)

                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.detach().cpu())

                progress_bar.set_postfix({"train/loss": train_losses[-1]})
                progress_bar.update()

            val_loss = self.validation_step(val_loader)
            train_losses = sum(train_losses) / len(train_losses)
            progress_bar.set_postfix({"val/loss": val_loss})
            progress_bar.update()

            self._train_loss = train_losses
            self._val_loss = val_loss

            self.wandb_logger.log({"train/loss": train_losses, "val/loss": val_loss})

            self.save_model()
