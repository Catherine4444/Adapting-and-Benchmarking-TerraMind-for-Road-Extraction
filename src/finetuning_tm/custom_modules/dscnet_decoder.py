import torch
from torch import nn, cat
from terratorch.registry import TERRATORCH_DECODER_REGISTRY
from finetuning_tm.custom_modules.dscnet.S3_DSConv import DSConv
from finetuning_tm.custom_modules.dscnet.S3_DSCNet import EncoderConv, DecoderConv

@TERRATORCH_DECODER_REGISTRY.register
class TerraTorchDSCNetDecoder(nn.Module):
    """
        TerraMind's embed_dim arrives as [D, D, D, D]
        DSCNet's Decoder is adapted to accept uniform amount of channels from ViT encoder
        before only accepting 24, 12, 6, 3 in channels per convolution.

        now e.g.:
        Plugging in your decoder_channels: [384, 64, 32, 16]:
            conv7: D → 384
            Block 1: (384 + D) → 64
            Block 2: (64 + D) → 32
            Block 3: (32 + D) → 16
        
        Args:
            embed_dim (list[int]): Input channel width at each of the 4 pyramidal scales
            handed to forward(), ordered shallow -> deep (e.g. after SelectIndices +
            ReshapeTokensToImage + LearnedInterpolateToPyramidal for a ViT backbone
            like TerraMind).

            channels (list[int]): Decoder widths, ordered deep -> shallow (bottleneck-first,
            final-output-last), matching TerraTorch/SMP convention. channels[0] sizes the
            bottleneck squeeze applied to the raw deepest feature; channels[1:] size each
            of the 3 upsample/fuse stages in turn. channels[-1] becomes self.out_channels.
            
            kernel_size: the size of kernel
        
            extend_scope: the range to expand (default 1 for this method)

            if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
            
            device: set on gpu
        
            dim: the dimension to concatenate, default 1 for channel dimension
    """
    def __init__(
        self, 
        embed_dim: list[int],
        channels: list[int],
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        if_offset: bool = True,
        device: str = "cuda",
        dim: int = 1,
        **kwargs                
    ):

        super().__init__()

        self.embed_dim = embed_dim
        self.channels = channels
        self.device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.dim = dim

        self._validate_shapes(self.embed_dim, self.channels)
        self.out_channels = channels[3]

        # model
        self.conv7 = EncoderConv(embed_dim[3], channels[0])

        in_ch_1 = channels[0] + embed_dim[2]
        self.conv120 = EncoderConv(in_ch_1, channels[1])
        self.conv12x = DSConv(in_ch_1, channels[1], self.kernel_size, self.extend_scope, 0, self.if_offset, self.device)
        self.conv12y = DSConv(in_ch_1, channels[1], self.kernel_size, self.extend_scope, 1, self.if_offset, self.device)
        self.conv13 = EncoderConv(3 * channels[1], channels[1])

        in_ch_2 = channels[1] + embed_dim[1]
        self.conv140 = DecoderConv(in_ch_2, channels[2])
        self.conv14x = DSConv(in_ch_2, channels[2], self.kernel_size, self.extend_scope, 0, self.if_offset, self.device)
        self.conv14y = DSConv(in_ch_2, channels[2], self.kernel_size, self.extend_scope, 1, self.if_offset, self.device)
        self.conv15 = DecoderConv(3 * channels[2], channels[2])

        in_ch_3 = channels[2] + embed_dim[0]
        self.conv160 = DecoderConv(in_ch_3, channels[3])
        self.conv16x = DSConv(in_ch_3, channels[3], self.kernel_size, self.extend_scope, 0, self.if_offset, self.device)
        self.conv16y = DSConv(in_ch_3, channels[3], self.kernel_size, self.extend_scope, 1, self.if_offset, self.device)
        self.conv17 = DecoderConv(3 * channels[3], channels[3])

        # 3. Final prediction head
        # Assuming your fusion brings everything back to a unified scale
        #self.out_conv = nn.Conv2d(channels[3], num_classes, 1)
        self.up = nn.Upsample(scale_factor=2,
                            mode="bilinear",
                            align_corners=True)

        # self.dropout = nn.Dropout(0.5)   

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: A list of multi-scale tensors from the TerraTorch neck.
                      Expected format: [Scale 1/4, Scale 1/8, Scale 1/16, Scale 1/32]
        """

        x_0_1, x_1_1, x_2_1, x_60xy_0 = features
        

        x_3_1 = self.conv7(x_60xy_0)

        x = self.up(x_3_1)
        x_120_2 = self.conv120(cat([x, x_2_1], dim=self.dim))
        x_12x_2 = self.conv12x(cat([x, x_2_1], dim=self.dim))
        x_12y_2 = self.conv12y(cat([x, x_2_1], dim=self.dim))
        x_2_3 = self.conv13(cat([x_120_2, x_12x_2, x_12y_2], dim=self.dim))

        # block5
        x = self.up(x_2_3)
        x_140_2 = self.conv140(cat([x, x_1_1], dim=self.dim))
        x_14x_2 = self.conv14x(cat([x, x_1_1], dim=self.dim))
        x_14y_2 = self.conv14y(cat([x, x_1_1], dim=self.dim))
        x_1_3 = self.conv15(cat([x_140_2, x_14x_2, x_14y_2], dim=self.dim))

        # block6
        x = self.up(x_1_3)
        x_160_2 = self.conv160(cat([x, x_0_1], dim=self.dim))
        x_16x_2 = self.conv16x(cat([x, x_0_1], dim=self.dim))
        x_16y_2 = self.conv16y(cat([x, x_0_1], dim=self.dim))
        x_0_3 = self.conv17(cat([x_160_2, x_16x_2, x_16y_2], dim=self.dim))
        # x = self.dropout(x)
        
        return x_0_3
    
    def _validate_shapes(self, embed_dim: list[int], channels: list[int]):
        if len(embed_dim) != 4 or len(channels) != 4:
            raise ValueError(
                f"DSCNetDecoder expects exactly 4 pyramidal scales, "
                f"got embed_dim={len(embed_dim)}, channels={len(channels)}"
            )
        for name, ch in zip(["channels[0]", "channels[1]", "channels[2]", "channels[3]"], channels):
            if ch % 4 != 0:
                raise ValueError(
                    f"{name}={ch} must be divisible by 4 (EncoderConv/DecoderConv use "
                    f"nn.GroupNorm(out_ch // 4, out_ch))"
                )