import argparse


def parse_args():
    """Parse input arguments."""
    desc = ('CT Segmentation Project\n')

    parser = argparse.ArgumentParser(description=desc)
    # ### ! [General] ! ###
    parser.add_argument('--project_tag', type=str, default="CTSeg_")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--NAS', type=str, default="//192.168.0.000/XXX/")
    parser.add_argument('--DTYPE', type=str, default="CBCT",
                        choices=["CT", "MDCT", "CBCT"])
    parser.add_argument('--source_company', type=str, default='YYY')
    parser.add_argument('--gpu', type=bool, default=False)
    # ### ! [Inference] ! ###
    parser.add_argument('--inference_eval', type=bool, default=True)
    parser.add_argument('--postprocess', type=bool, default=False)
    parser.add_argument('--pati_num', type=str, default="00000")
    parser.add_argument('--predicted_log_dir', type=str, default="/Users/dyanlee/workspace/CTSegmentation/Logs")
    parser.add_argument('--model_path', type=str, default="checkpoints.pth")
    # ### ! [Dataset Loading] ! ###
    # parser.add_argument('--source', type=str, default="/data/CT/")
    # parser.add_argument('--converted_dir', type=str, default="/data/CT/CBCT/YYY/vti/")
    parser.add_argument('--input_size', type=int, default=224)
    # ### ! [Training] ! ###
    parser.add_argument('--net', type=str, default="unet",
                        choices=["enet", "mobilev2unet", "mobilev2", "unet", "unet3D"])
    parser.add_argument('--loss', type=str, default="ce",
                        choices=["ce", 'softiou', 'ohem', 'focal', 'softce', 'kl',
                                 'dice', 'jaccard', 'tversky'])

    # ### Transfer Learning
    parser.add_argument('--transfer', type=bool, default=False)
    parser.add_argument('--pretrained_model', type=str, default="checkpoints.pth")
    # ### Model Hyper-params
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--in_dim', type=int, default=5)
    parser.add_argument('--out_dim', type=int, default=3)  # N_CLASS = 3
    parser.add_argument('--width_mult', type=float, default=1.,
                        choices=[0.75, 1.0])  # mobilev2
    parser.add_argument('--final_activation', type=str, default='softmax',
                        choices=["softmax", "sigmoid"])  # unet3D
    # ### Loader
    parser.add_argument('--load_mode', type=str, default="ORI",
                        choices=["NORM", "ORI"])
    parser.add_argument('--slice_batch', type=int, default=2)
    parser.add_argument('--val_split', type=float, default=.2)
    parser.add_argument('--pati_batch', type=int, default=1)
    parser.add_argument('--pati_shuffle', type=bool, default=True)
    parser.add_argument('--num_slices', type=int, default=5)
    parser.add_argument('--slice_shuffle', type=bool, default=True)
    # ### preprocessing
    parser.add_argument('--augmentation', type=bool, default=False)
    parser.add_argument('--scale', type=float, default=0.5)  # Downsampling == preprocessing for smaller resizing
    # ### Optimizer
    parser.add_argument('--optimizer', type=str, default="adam",
                        choices=["adam", "sgd", 'rmsprop'])

    # ### ! [MISC] ! ###
    parser.add_argument("--pthfile", type=str, default="checkpoint.pth")
    args = parser.parse_args()
    return args
