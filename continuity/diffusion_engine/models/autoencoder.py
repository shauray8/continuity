import torch
import torch.cuda

# Copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py:DiagonalGaussianOriginal
class DiagonalGaussianOriginal(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean

class FastDiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim
        self.stream = torch.cuda.Stream()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            with torch.cuda.stream(self.stream):
                noise = torch.randn_like(mean, memory_format=torch.preserve_format)
                return mean + std * noise
        return mean

class AutoEncoder(torch.nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder=Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.decoder=Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg=FastDiogonalGaussian()

        self.scale_factor=params.scale_factor
        self.shift_factor=params.shift_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x=self.reg(self.encoder(x))
        return self.scale_factor*(x-self.shift_factor)

    def decode(self, x:torch.Tensor) -> torch.Tensor:
        x=x/self.scale_factor+self.shift_factor
        return self.decoder(x)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.decode(self.encoder(x))

