import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
import glob
import os
from torchvision.io import read_video, read_audio


class Picture2Feature(nn.Module):
    def __init__(self):
        super(Picture2Feature, self).__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.fc = nn.Identity()

    def forward(self, x):
        return self.resnet18(x)


class Feature2Classifier(nn.Module):
    def __init__(self, num_classes=32, num_features=512, num_hidden=256):
        super(Feature2Classifier, self).__init__()
        self.classifer = nn.Sequential(nn.Linear(num_features, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_classes))

    def forward(self, x):
        return self.classifer(x)


class Feature2Audio(nn.Module):
    def __init__(self, video_length=29, video_feature=512, audio_length=16000, audio_feature=1):
        super(Feature2Audio, self).__init__()
        self.preconv = nn.Sequential(
            nn.ConvTranspose1d(video_feature, video_feature // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(video_feature // 2, video_feature // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(video_feature // 4, video_feature // 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.mid_length = ((video_length * 2 - 1) * 2 - 1) * 2 - 1
        self.audio_length = audio_length
        self.lastconv = nn.Sequential(
            nn.Conv1d(video_feature // 8, audio_feature * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(audio_feature * 4, audio_feature * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(audio_feature * 2, audio_feature, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.preconv(x)
        # 把 x 的维度从 [batch, feature, mid_length] 插值 [batch, feature, audio_length]
        x = F.interpolate(x, size=self.audio_length, mode="linear")
        x = self.lastconv(x)
        return x


class Model(pl.LightningModule):
    def __init__(self, num_classes=32, num_features=512, num_hidden=256, video_length=29, audio_length=16000):
        super(Model, self).__init__()
        self.picture2feature = Picture2Feature()
        self.feature2classifier = Feature2Classifier(num_classes=num_classes, num_features=num_features, num_hidden=num_hidden)
        self.feature2audio = Feature2Audio(video_length=video_length, video_feature=num_features, audio_length=audio_length, audio_feature=1)
        self.num_features = num_features

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feature = self.picture2feature(x)
        output_classifier = self.feature2classifier(feature)
        feature = feature.reshape(b, t, self.num_features)
        output_audio = self.feature2audio(feature)
        return output_classifier, output_audio

    def training_step(self, batch, batch_idx):
        video, audio = batch
        # video: [batch, 29, 3, 256, 256], audio: [batch, 19456]
        classifier, audio_hat = self(video)
        predict_class = torch.argmax(classifier, dim=1)
        loss_classifier = F.cross_entropy(classifier, predict_class)
        loss_audio = F.mse_loss(audio_hat, audio)
        loss = 0.1 * loss_classifier + loss_audio
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("loss", avg_loss)


class LRW_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, path="path/to/LRW"):
        super(LRW_DataModule, self).__init__()
        self.batch_size = batch_size
        self.path = path

    def setup(self, stage=None):
        self.train_dataset = self.get_dataset(self.path, "train")
        self.val_dataset = self.get_dataset(self.path, "val")
        self.test_dataset = self.get_dataset(self.path, "test")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def get_dataset(self, path, mode):
        # path is "path/lipread_mp4/{word}/{mode}/{word}_{index}.mp4"
        # glob path, end with .mp4
        glob_path = os.path.join(path, "lipread_mp4", "*", mode, "*.mp4")
        files_list = glob.glob(glob_path)
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, files_list):
                self.files_list = files_list

            def __len__(self):
                return len(self.files)

            def __getitem__(self, index):
                filename = self.files[index]
                # read video and audio
                file = read_video(filename)
                video, audio, _ = file
                video = video.permute(0, 3, 1, 2)
                return video, audio
        return Dataset(files_list)

if __name__ == "__main__":
    # LRW video is (29, 3, 256, 256), audio is (1, 19456)
    model = Model()
    # dataset is LRW dataset
    datamodule = LRW_DataModule(path="path/to/LRW")
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, datamodule)
    trainer.test()