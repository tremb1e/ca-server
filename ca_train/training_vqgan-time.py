import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import weights_init
from mutils import AverageMeter
import time

# 假设传感器数据为时序数据
class SensorDataset(Dataset):
    def __init__(self, sensor_data, labels, seq_length):
        self.sensor_data = sensor_data
        self.labels = labels
        self.seq_length = seq_length

    def __len__(self):
        return len(self.sensor_data)

    def __getitem__(self, idx):
        data = self.sensor_data[idx]
        label = self.labels[idx]
        # 截取固定长度的序列
        data = data[:self.seq_length]
        return torch.FloatTensor(data), torch.LongTensor([label])

class VQGAN_LSTM(VQGAN):
    def __init__(self, args):
        super(VQGAN_LSTM, self).__init__(args)
        
        # 修改编码器，使用LSTM处理时序数据
        self.lstm_encoder = nn.LSTM(input_size=args.num_features, hidden_size=128, num_layers=2, batch_first=True)
        
        # 修改解码器，使用LSTM解码
        self.lstm_decoder = nn.LSTM(input_size=128, hidden_size=args.num_features, num_layers=2, batch_first=True)

        # 修改后继续使用VQGAN的其他部分
        self.codebook = nn.Embedding(args.num_codebook_vectors, args.latent_dim)
        self.quant_conv = nn.Conv1d(args.latent_dim, args.latent_dim, 1)
        self.post_quant_conv = nn.Conv1d(args.latent_dim, args.latent_dim, 1)

    def forward(self, x):
        # 使用LSTM作为编码器处理时序数据
        lstm_out, _ = self.lstm_encoder(x)
        
        # 量化步骤
        z_e_x = lstm_out[:, -1, :]  # 使用LSTM的最后一个时间步的输出
        quantized, q_loss = self.vector_quantize(z_e_x)
        
        # 使用LSTM作为解码器进行时序数据的生成
        lstm_out_decoded, _ = self.lstm_decoder(quantized.unsqueeze(1).repeat(1, x.size(1), 1))
        return lstm_out_decoded, q_loss

class TrainVQGAN:
    def __init__(self, args, sensor_data, sensor_labels):
        self.vqgan = VQGAN_LSTM(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        # 初始化数据集和数据加载器
        self.seq_length = 100  # 假设每个数据序列长度为100
        self.train_dataset = SensorDataset(sensor_data, sensor_labels, self.seq_length)
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)

        self.prepare_training()
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.lstm_encoder.parameters()) +
            list(self.vqgan.lstm_decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def temporal_smoothness_loss(self, decoded, original):
        """
        时间一致性损失：确保生成数据在时间维度上的一致性。
        这里我们简单地使用了相邻时间步之间的L2损失。
        """
        loss = torch.mean(torch.abs(decoded[:, 1:, :] - decoded[:, :-1, :]))
        return loss

    def train(self, args):
        steps_per_epoch = len(self.train_loader)
        for epoch in range(args.epochs):
            with tqdm(range(len(self.train_loader))) as pbar:
                for i, (imgs, _) in zip(pbar, self.train_loader):
                    imgs = imgs.to(device=args.device)

                    # 使用VQGAN生成数据
                    decoded_images, q_loss = self.vqgan(imgs)

                    # 计算损失
                    rec_loss = torch.abs(imgs - decoded_images)
                    rec_loss = rec_loss.mean()

                    # 添加时间一致性损失
                    temporal_loss = self.temporal_smoothness_loss(decoded_images, imgs)

                    # VQGAN总损失
                    vq_loss = rec_loss + q_loss + temporal_loss

                    # 更新优化器
                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    self.opt_disc.step()

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        Temporal_Loss=np.round(temporal_loss.cpu().detach().numpy().item(), 5)
                    )
                    pbar.update(0)

                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))

    def test(self, args, testloader, vqgan):
        vqgan.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device=args.device)
            targets = targets.to(device=args.device)
            decoded_images, q_loss = vqgan(inputs)

            scores = torch.mean(torch.pow((inputs - decoded_images), 2), dim=[1,2,3])
            prec1 = torch.mean(scores).item()  # 简单计算均方误差

            top1.update(prec1, inputs.size(0))
            batch_time.update(time.time() - batch_time.avg)

        return top1.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN for Sensor Data")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z')
    parser.add_argument('--seq-length', type=int, default=100, help='Sequence length of sensor data')
    parser.add_argument('--num-features', type=int, default=6, help='Number of sensor features')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to train on')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta2 parameter')

    args = parser.parse_args()
    args.dataset_path = r"./Data"

    # 假设传感器数据已经预处理为numpy数组，sensor_data为形状(batch_size, seq_length, num_features)
    sensor_data = np.random.randn(1000, 100, 6)  # 示例：1000个样本，序列长度为100，6个特征
    sensor_labels = np.random.randint(0, 2, size=1000)  # 示例标签：0或1

    train_vqgan = TrainVQGAN(args, sensor_data, sensor_labels)
