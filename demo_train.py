from torch.optim import AdamW
from src.data.face_dataset import FaceDataset
from src.models.vit import ViT
from src.trainer import Trainer
import wandb
import json
from torch.utils.data import DataLoader

def load_train_val_datasets(dataset: dict):
    people = list(dataset.keys())
    val_size = 0.2
    max_train_idx = int(len(people) * (1 - val_size))

    train_split = {}
    val_split = {}

    for i in range(len(people)):
        k = people[i]
        if i < max_train_idx:
            train_split[k] = dataset[k]
        else:
            val_split[k] = dataset[k]
            
    len(list(train_split.keys())), len(list(val_split.keys()))

    return train_split, val_split


def run_demo_train():

    trainer = Trainer(device="cpu", distance="cosine_similarity")
    LR = 1e-4
    EPOCHS = 3
    BATCH_SIZE = 4

    
    vit = ViT(
        transformer_depth=2,
        attn_heads=2,
        mlp_dim=768,
        output_dim=16,
        patch_width = 16,
        patch_height = 16,
        image_width = 128,
        image_height = 128,
        patch_embeddings_dim = 64
    )

    optimizer = AdamW(vit.parameters(), lr=LR)
    trainer.config_trainer(vit, optimizer, wandb_logger=None)

    with open("dataset/face_dataset.json", 'r') as f:
        dataset_dict = json.load(f)

    train_split_dataset, val_split_dataset = load_train_val_datasets(dataset_dict)

    train_loader = DataLoader(FaceDataset(train_split_dataset), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(FaceDataset(val_split_dataset), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    len(train_loader), len(val_loader)

    trainer.fit(train_loader, val_loader, EPOCHS)



if __name__ == "__main__":
    run_demo_train()