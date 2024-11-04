import torch
import pytorch_lightning as pl
import glob
import os
from torchvision.io import read_video

class LRW_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, path="path/to/LRW", num_workers=0):
        super(LRW_DataModule, self).__init__()
        self.batch_size = batch_size
        self.path = path
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = self.get_dataset(self.path, "train")
        self.val_dataset = self.get_dataset(self.path, "val")
        self.test_dataset = self.get_dataset(self.path, "test")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def get_dataset(self, path, mode):
        # path is "path/lipread_mp4/{word}/{mode}/{word}_{index}.mp4"
        # glob path, end with .mp4
        glob_path = os.path.join(path, "*", mode, "*.mp4")
        files_list = glob.glob(glob_path)

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, files_list):
                self.files_list = files_list

            def __len__(self):
                return len(self.files_list)

            def __getitem__(self, index):
                filename = self.files_list[index]
                # read video and audio
                file = read_video(filename, pts_unit="sec")
                video, audio, _ = file
                video = video.permute(0, 3, 1, 2).float()
                audio = audio.float()
                return video, audio

        return Dataset(files_list)
