from .enet import ENet
from .mobilenetV2_unet import MobileNetV2_unet
from .mobilenetV2 import MobileNetV2
from .unet2D import UNet
from .unet3D import Unet3D


def get_model(name, args):

    if name == "enet":
        return ENet(num_classes=args.out_dim, encoder_relu=False, decoder_relu=True)

    elif name == "mobilev2unet":
        return MobileNetV2_unet(n_classes=args.out_dim)  # , pre_trained=args.pretrained_model)

    elif name == "mobilev2":
        return MobileNetV2(n_channels=args.in_dim,
                           n_classes=args.out_dim,
                           input_size=args.input_size,
                           width_mult=args.width_mult)

    elif name == "unet":
        return UNet(n_channels=args.in_dim, n_classes=args.out_dim)  # , bilinear=True

    elif name == "unet3D":
        return Unet3D(n_channels=args.in_dim,
                      n_classes=args.out_dim,
                      final_activation=args.final_activation)

    else:
        raise ValueError("Model not found")
