import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
from dataset import LRW_DataModule


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
    def forward(self, x):
        flat_input = x.view(-1, self.embedding_dim)
        distances = torch.cdist(flat_input, self.embeddings.weight)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.size(0), self.num_embeddings, device=x.device
        )
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(x.shape)
        quantized = x + (quantized - x).detach()
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        return quantized, loss, encoding_indices


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
        self.classifer = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_classes),
        )

    def forward(self, x):
        return self.classifer(x)


class Feature2Audio(nn.Module):
    def __init__(
        self, video_length=29, video_feature=512, audio_length=16000, audio_feature=1
    ):
        super(Feature2Audio, self).__init__()
        self.preconv = nn.Sequential(
            nn.ConvTranspose1d(
                video_feature, video_feature // 2, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                video_feature // 2,
                video_feature // 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                video_feature // 4,
                video_feature // 8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
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
    def __init__(
        self,
        num_classes=32,
        num_features=512,
        num_hidden=256,
        video_length=29,
        audio_length=16000,
    ):
        super(Model, self).__init__()
        self.picture2feature = Picture2Feature()
        self.embedding = VectorQuantizer(
            num_embeddings=num_classes, embedding_dim=num_features, commitment_cost=0.25
        )
        self.feature2audio = Feature2Audio(
            video_length=video_length,
            video_feature=num_features,
            audio_length=audio_length,
            audio_feature=1,
        )
        self.num_features = num_features

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feature = self.picture2feature(x)
        feature = feature.reshape(b, t, self.num_features)
        quantized, vq_loss, _ = self.embedding(feature)
        audio_hat = self.feature2audio(quantized)
        return audio_hat, feature, vq_loss

    def training_step(self, batch, batch_idx):
        video, audio = batch
        # video: [batch, 29, 3, 256, 256], audio: [batch, 19456]
        audio_hat, feature, vq_loss = self(video)
        loss_audio = F.mse_loss(audio_hat, audio)
        loss = loss_audio + vq_loss
        return loss

    def validation_step(self, batch, batch_idx):
        video, audio = batch
        # video: [batch, 29, 3, 256, 256], audio: [batch, 19456]
        audio_hat, feature, vq_loss = self(video)
        loss_audio = F.mse_loss(audio_hat, audio)
        loss = loss_audio + vq_loss * 0.5
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_training_batch_end(self, outputs, batch, batch_idx):
        self.log("loss_tra", outputs, prog_bar=True)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log("loss_val", outputs, prog_bar=True)


if __name__ == "__main__":
    # LRW video is (29, 3, 256, 256), audio is (1, 19456)
    torch.set_float32_matmul_precision('medium')
    model = Model(
        num_classes=32,
        num_features=512,
        num_hidden=256,
        video_length=29,
        audio_length=19456,
    )
    # dataset is LRW dataset
    datamodule = LRW_DataModule(
        path="/ai/storage/LRW", batch_size=16, num_workers=4
    )
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule, ckpt_path='best')
