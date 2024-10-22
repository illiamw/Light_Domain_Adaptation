from .deeplab_multi import DeeplabMulti
from .liteseg import LiteSeg


def get_model(cfg):
    if cfg.model.backbone == "deeplabv2_multi":
        model = DeeplabMulti(num_classes=cfg.data.num_classes, init=cfg.model.imagenet_pretrained)
    elif cfg.model.backbone == "LiteSeg":
        model = LiteSeg(n_classes=cfg.data.num_classes,PRETRAINED_WEIGHTS=cfg.model.imagenet_pretrained)
        params = model.optim_parameters(lr=cfg.opt.lr)
    else:
        raise NotImplementedError()
    return model, params

def get_model_test(n):
    return LiteSeg(n_classes=n)
