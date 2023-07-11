#-*- coding:utf-8 -*-

from diffusion.model import UNetModel, timestep_embedding
from torchvision import models
from typing import Optional
from enum import Enum
from typing import Union
import torch 

class ModelType(Enum):
    CNN = "C"
    TRANSFORMER = "T"

NUM_CLASSES = 1

def rand_log_normal(shape, loc=0., scale=1., device='cuda', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()

class KarrasUnet(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, sigmas, global_cond=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (global_cond is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        c_noises = sigmas.log() / 4
        emb = self.time_embed(timestep_embedding(c_noises, self.model_channels))

        if self.num_classes is not None:
            assert global_cond.shape == (x.shape[0],)
            emb = torch.cat([emb, self.label_emb(global_cond)], 1)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

def create_edm_model(
    image_size:int,
    num_channels:int,
    num_res_blocks:int,
    channel_mult:Optional[str]="",
    learn_sigma:bool=False,
    class_cond:bool=False,
    use_checkpoint:bool=False,
    attention_resolutions:str="16",
    num_heads:int=1,
    num_head_channels:int=-1,
    num_heads_upsample:int=-1,
    use_scale_shift_norm:bool=False,
    dropout:int=0,
    resblock_updown:bool=False,
    use_fp16:bool=False,
    use_new_attention_order:bool=False,
    in_channels:int=1,
    out_channels:int=1,
    dims:int=2
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return KarrasUnet(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(1*out_channels if not learn_sigma else 2*out_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        dims=dims
    )

if __name__ == '__main__':
    pass 