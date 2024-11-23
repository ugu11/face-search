import torch

def mse_distance(x, y):
    return ((x - y) ** 2).mean(1)

def cosine_similarity(x, y, eps=1e-8, device="cpu", dtype=torch.float32):
    # x = x / x.norm(dim=1, keepdim=True)
    # y = y / y.norm(dim=1, keepdim=True)
    x_norm = torch.max(x.norm(dim=1, keepdim=True), torch.ones(x.shape, device=device, dtype=dtype) * eps)
    y_norm = torch.max(y.norm(dim=1, keepdim=True), torch.ones(y.shape, device=device, dtype=dtype) * eps)
    
    similarity = (x @ y.t()) / (x_norm @ y_norm.t())

    return 1 - similarity


class TripletLoss:
    def __init__(self, distance: str = 'mse'):
        # Change later when new distances are added
        if distance == 'mse':
            self.distance = mse_distance
        else:
            self.distance = mse_distance

    def __call__(self, margin, anchor, positive, negative, model):
        anchor_outputs = model(anchor)
        positive_outputs = model(positive)
        negative_outputs = model(negative)

        positive_distance = self.distance(anchor_outputs, positive_outputs)
        negative_distance = self.distance(anchor_outputs, negative_outputs)
        negative_distance_swap = self.distance(positive_outputs, negative_outputs)
        hard_negative_distance = torch.min(negative_distance, negative_distance_swap)

        loss = torch.max((margin + positive_distance - hard_negative_distance).mean(), torch.tensor(0))

        return loss, positive_distance.mean(), hard_negative_distance.mean()