
from .mlpmixer import MlpMixer
from .gmlp import gmlp


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'mixer':
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
    elif model_type == 'gmlp':
        model = gmlp(attention=config.MODEL.GMLP.ATTENTION)
    else:    
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
