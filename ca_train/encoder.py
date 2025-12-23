import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # Sensor windows are treated as a "1×H×T image" (H=12 sensor rows, T=time axis).
        # Keep more spatial resolution early to avoid collapsing the 12 sensor rows too aggressively.
        base_channels = int(getattr(args, "base_channels", 128))
        channels = [
            base_channels,
            int(base_channels * 1.5),
            int(base_channels * 2.0),
            int(base_channels * 2.5),
        ]
        use_nonlocal = bool(getattr(args, "use_nonlocal", True))
        attn_resolutions = []  # 依赖瓶颈处的 NonLocal 即可
        num_res_blocks = 2
        resolution = 12  # 粗略跟踪特征维度高度
        layers = [nn.Conv2d(args.image_channels, channels[0], 3, 1, 1)]
        downsample_plan = [
            {"stride": (2, 2), "pad": (0, 1, 0, 1)},  # 12x50 -> 6x25
            {"stride": (1, 2), "pad": (0, 1, 1, 1)},  # 6x25 -> 6x12（仅压缩时间轴）
            {"stride": (1, 2), "pad": (0, 1, 1, 1)},  # 6x12 -> 6x6  得到 36 个 token
        ]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i < len(downsample_plan):
                cfg = downsample_plan[i]
                layers.append(DownSampleBlock(channels[i + 1], stride=cfg["stride"], pad=cfg["pad"]))
                # 粗略跟踪分辨率，主要用于是否添加 NonLocalBlock
                resolution = resolution // cfg["stride"][0]
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        if use_nonlocal:
            layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
