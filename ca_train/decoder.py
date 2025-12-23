import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        base_channels = int(getattr(args, "base_channels", 128))
        channels = [
            int(base_channels * 2.5),
            int(base_channels * 2.0),
            int(base_channels * 1.5),
            base_channels,
        ]
        use_nonlocal = bool(getattr(args, "use_nonlocal", True))

        # Expected input "image" size for sensor windows: (1, H=12, W=target_width).
        # The encoder downsamples with:
        #   1) stride (2,2)  : H -> H1, W -> W1
        #   2) stride (1,2)  : H1 stays, W1 -> W2
        #   3) stride (1,2)  : H1 stays, W2 -> W3
        #
        # We compute matching upsample targets dynamically so the same VQGAN can
        # be used with different `target_width` values.
        input_height = int(getattr(args, "input_height", 12))
        input_width = int(getattr(args, "input_width", 50))

        def _downsample_w(width: int) -> int:
            # Encoder uses manual pad_right=1, kernel=3, stride=2 on the time axis.
            return max(1, (width - 2) // 2 + 1)

        height_1 = max(1, (input_height - 2) // 2 + 1)
        width_1 = _downsample_w(input_width)
        width_2 = _downsample_w(width_1)
        # width_3 is the latent width, kept for clarity.
        width_3 = _downsample_w(width_2)
        _ = width_3

        size = [
            (height_1, width_2),
            (height_1, width_1),
            (input_height, input_width),
        ]
        attn_resolutions = []
        num_res_blocks = 2
        resolution = 6

        in_channels = channels[0]
        layers = [
            nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels),
        ]
        if use_nonlocal:
            layers.append(NonLocalBlock(in_channels))
        layers.append(ResidualBlock(in_channels, in_channels))

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels, size[i-1]))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

