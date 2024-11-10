import random
from typing import Optional
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image
import json

class FaceDataset(Dataset):
    def __init__(self, dataset: Optional[dict] = None, dataset_path: Optional[str] = None):
        assert dataset != None or dataset_path != None

        if dataset == None:
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)
        else:
            self.dataset = dataset
            
        self.keys = list(self.dataset.keys())
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(35),
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        person = self.keys[idx]
        random_face1_idx = random.randint(0, len(self.dataset[person])-1)
        random_face2_idx = random.randint(0, len(self.dataset[person])-1)
        random_stranger_idx = random.randint(0, len(self.keys)-1)
        
        while random_face2_idx == random_face1_idx:
            random_face2_idx = random.randint(0, len(self.dataset[person])-1)
            
        while random_stranger_idx == idx:
            random_stranger_idx = random.randint(0, len(self.keys)-1)
            
        stranger = self.keys[random_stranger_idx]
        random_stranger_face_idx = random.randint(0, len(self.dataset[stranger])-1)
        
        face1 = (read_image(self.dataset[person][random_face1_idx]).float() / 255.0) * 2.0 - 1.0
        face2 = (read_image(self.dataset[person][random_face2_idx]).float() / 255.0) * 2.0 - 1.0
        stranger_face = (read_image(self.dataset[stranger][random_stranger_face_idx]).float() / 255.0) * 2.0 - 1.0

        return {
            "face1": self.transforms(face1),
            "face2": self.transforms(face2),
            "stranger": self.transforms(stranger_face)
        }