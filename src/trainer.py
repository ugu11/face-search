from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class Trainer:
    def __init__(self, ckpt_path="checkpoints/", device="cuda", dtype=torch.float32, save_ckpt=False, distance="mse"):
        self.ckpt_path = ckpt_path
        self.device = device
        self.dtype = dtype
        self.save_ckpt = save_ckpt
        self.current_epoch = 0
        self.distance = distance

        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

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

    def mse_distance(self, x, y):
        return ((x - y) ** 2).mean(1)
    
    def cosine_similarity(self, x, y):
        x = x / x.norm(dim=1, keepdim=True)
        y = y / y.norm(dim=1, keepdim=True)
        
        return self.logit_scale.exp() * x @ y.t()

    def calc_distance(self, x, y):
        if self.distance == "cosine_similarity":
            return self.cosine_similarity(x, y)
        else:
            return self.mse_distance(x, y)


    def save_model(self,):
        self._delete_older_checkpoints()

        torch.save({
            'epoch': self.current_epoch,
            'vit_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train/loss': self._train_loss,
            'val/loss': self._val_loss,
        }, os.path.join(self.ckpt_path, f"epoch={self.current_epoch}-vloss{'{:.2f}'.format(self._val_loss)}.ckpt"))


    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")

        self.model.load_state_dict(checkpoint["vit_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self._train_loss = checkpoint["train/loss"]
        self._val_loss = checkpoint["val/loss"]

        print("Loaded: ", self.current_epoch, self._train_loss, self._val_loss)


    def shared_step(self, batch, margin: float = 0.0) -> float:
        face1, face2, stranger = batch["face1"], batch["face2"], batch["stranger"]
        
        face1 = face1.to(device=self.device, dtype=self.dtype)
        face2 = face2.to(device=self.device, dtype=self.dtype)
        stranger = stranger.to(device=self.device, dtype=self.dtype)

        face1_outputs = self.model(face1)
        face2_outputs = self.model(face2)
        stranger_outputs = self.model(stranger)

        positive_distance = self.calc_distance(face1_outputs, face2_outputs)
        negative_distance = self.calc_distance(face1_outputs, stranger_outputs)
        negative_distance_swap = self.calc_distance(face2_outputs, stranger_outputs)
        hard_negative_distance = torch.min(negative_distance, negative_distance_swap)

        loss = torch.max((margin + positive_distance - hard_negative_distance).mean(), torch.tensor(0))

        return loss, positive_distance, hard_negative_distance
    
    def validation_step(self, val_loader: DataLoader, margin: float = 0.0) -> float:
        val_loss = []
        positive_distances = []
        negative_distances = []
        with torch.no_grad():
            for _, data in enumerate(val_loader):
                loss, positive_distance, negative_distance = self.shared_step(data, margin)
                val_loss.append(loss.detach().cpu())
                positive_distances.append(positive_distance.detach().cpu())
                negative_distances.append(negative_distance.detach().cpu())

        val_loss = sum(val_loss) / len(val_loss)
        positive_distance = sum(positive_distance) / len(positive_distance)
        negative_distances = sum(negative_distances) / len(negative_distances)

        return val_loss, positive_distance, negative_distances

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            margin: float = 0.0
        ):
        for e in range(self.current_epoch, epochs):
            self.current_epoch = e
            progress_bar = tqdm(total=len(train_loader))
            progress_bar.set_description(f"Epoch {e}")
            train_losses = []
            positive_distances = []
            negative_distances = []
            self.wandb_logger.log({"epoch": e})

            for i, data in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                loss, positive_distance, negative_distance = self.shared_step(data, margin=margin)

                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.detach().cpu())
                positive_distances.append(positive_distance.detach().cpu())
                negative_distances.append(negative_distance.detach().cpu())

                progress_bar.set_postfix({
                    "train/loss": train_losses[-1],
                    "train/positive_distance": positive_distances[-1],
                    "train/negative_distance": negative_distances[-1],
                })
                progress_bar.update()
                tqdm._instances.clear()

            val_loss, val_positive_distance, val_negative_distance = self.validation_step(val_loader, margin=margin)
            train_losses = sum(train_losses) / len(train_losses)
            positive_distances = sum(positive_distances) / len(positive_distances)
            negative_distances = sum(negative_distances) / len(negative_distances)

            logging_data = {
                "train/loss": train_losses,
                "train/positive_distance": positive_distances,
                "train/negative_distance": negative_distances,
                "val/loss": val_loss,
                "val/positive_distance": val_positive_distance,
                "val/negative_distance": val_negative_distance,
            }

            progress_bar.set_postfix(logging_data)
            progress_bar.update()
            tqdm._instances.clear()

            self._train_loss = train_losses
            self._val_loss = val_loss

            self.wandb_logger.log(logging_data)

            if self.save_ckpt:
                self.save_model()
