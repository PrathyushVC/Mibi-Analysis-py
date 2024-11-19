import torch
import random
#The lack of default p is intended
class RandomHorizontalFlip3D:
    def __init__(self, p):
        if not (0 <= p <= 1):
            raise ValueError("Parameter p must be between 0 and 1.")
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            tensor = torch.flip(tensor, dims=[2])  
        return tensor

class RandomVerticalFlip3D:
    def __init__(self,p):
        if not (0 <= p <= 1):
            raise ValueError("Parameter p must be between 0 and 1.")
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            tensor = torch.flip(tensor, dims=[1])  # Flip height axis
        return tensor

class RandomRotation3D:
    def __init__(self, p):
        if not (0 <= p <= 1):
            raise ValueError("Parameter p must be between 0 and 1.")
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            num_rotations = random.choice([1, 2, 3])  # Rotate by 90, 180, or 270 degrees
            tensor = torch.rot90(tensor, k=num_rotations, dims=[1, 2])
        return tensor

class Compose3D:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tensor):
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor
