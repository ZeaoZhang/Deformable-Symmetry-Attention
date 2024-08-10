import argparse

parser = argparse.ArgumentParser()
dir_name = ''
name = ''


parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')
parser.add_argument("--model", type=str, help="voxelmorph 1 or 2",
                    dest="model", choices=['vm1', 'vm2'], default='vm2')
parser.add_argument("--func", type=str, help="reg function",
                    dest="func", choices=['stf', 'rstf2d', 'rstf3d'], default='rstf2d')
parser.add_argument("--dataset", type=str, help="dataset",
                    dest="dataset", choices=['wbs', 'brats', 'chexray'], default='wbs')


parser.add_argument("--front", type=str, help="The registration can be performed individually on the anterior and posterior images or collectively as a whole.",
                    dest="front", choices=["True", "False", "both"], default="both")
parser.add_argument("--sim_loss", type=list, help="image similarity loss: mse or ncc or ssim",
                    dest="sim_loss", default=['ssim'])
parser.add_argument("--weight", type=list, help="weight of each image similarity loss",
                    dest="weight", default=[1.0])
parser.add_argument("--aux_loss", type=str, help="image auxiliary loss: nj or bp or None",
                    dest="aux_loss", default='nj')
parser.add_argument("--alpha", type=float, help="regularization loss parameter",
                    dest="alpha", default=1.0)  
parser.add_argument("--beta", type=float, help="auxiliary loss parameter",
                    dest="beta", default=1e-7)  
parser.add_argument("--checkpoint", type=str, help="model weight file",
                    dest="checkpoint", default='')

parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default=f'./{dir_name}/{name}/Result')
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default=f'./{dir_name}/{name}/Checkpoint')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default=f'./{dir_name}/{name}/Log')

parser.add_argument("--root_path", type=str, help="gpu id number",
                    default='')
parser.add_argument("--suffix", type=str, help="suffix of the image file",
                    default='.png')
parser.add_argument("--train_splits", type=str, help="train splits file path",
                    default='')
parser.add_argument("--val_splits", type=str, help="val splits file path",
                    default='')
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=1e-4)
parser.add_argument("--epoch", type=int, help="number of epoch",
                    dest="epoch", default=300)
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--n_save_iter", type=int, help="frequency of model saves",
                    dest="n_save_iter", default=20)


args = parser.parse_args()