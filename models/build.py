# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .mlpmixer import MlpMixer

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'mixer':
        model = MlpMixer(num_classes=config.MODEL.NUM_CLASSES,
                        img_size=config.DATA.IMG_SIZE,
                        in_chans=config.MODEL.MIXER.IN_CHANS,
                        patch_size=config.MODEL.MIXER.PATCH_SIZE,
                        num_blocks=config.MODEL.MIXER.NUM_BLOCKS,
                        hidden_dim=config.MODEL.MIXER.HIDDEN_DIM,
                        mlp_ratio=config.MODEL.MIXER.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        stem_norm=config.MODEL.MIXER.STEM_NORM)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
