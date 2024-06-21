import tables
import torch
from torch.utils.data import Dataset


class PyTableDataset(Dataset):
    def __init__(self, pt_path):
        self.file = tables.open_file(pt_path, mode="r")
        self.images = self.file.root.img
        self.masks = self.file.root.mask
        self.numpixels = self.file.root.numpixels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        numpixels = self.numpixels[idx]

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        numpixels = torch.tensor(numpixels, dtype=torch.int32)

        return image, mask, numpixels

    def __del__(self):
        # https://teddykoker.com/2020/12/dataloader/
        self.file.close()


if __name__ == "__main__":
    pt_path = "Mibi_trial_val.pytable"
    dataset = PyTableDataset(pt_path)
    image, mask, numpixels = dataset[0]
    print(image.shape, mask.shape, numpixels)
