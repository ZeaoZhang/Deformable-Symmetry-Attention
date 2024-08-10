import os
import glob
import warnings

import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import AdamW
import torch.utils.data as Data

from Model import losses
from Model.config import args
from Model.datagenerators import Dataset
from Model.model import U_Network, SpatialTransformer
import cv2

def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(input_moving, input_fixed, m2f, flow, name):
    input_moving = np.squeeze(input_moving.cpu().detach().numpy())
    input_fixed = np.squeeze(input_fixed.cpu().detach().numpy())
    m2f = np.squeeze(m2f.cpu().detach().numpy())
    # flow = np.squeeze(flow.cpu().detach().numpy())
    # flows = [flow[i] for i in range(flow.shape[0])]
    img = np.concatenate([input_moving, input_fixed, m2f], axis=1)  
    # flow_img = np.concatenate(flows, axis=1)  
    cv2.imwrite(os.path.join(args.result_dir, name), img)
    # cv2.imwrite(os.path.join(args.result_dir, "flow_" + name), flow_img)

def save_args():
    argsDict = args.__dict__
    with open(os.path.join(args.log_dir, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------' + '\n')


def train():
    # Create the required folder and specify the gpu
    make_dirs()
    save_args()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # log file
    log_name = str(args.epoch) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # Get all the names of the training data
    DS = Dataset(root_path=args.root_path, splits=args.train_splits, suffix=args.suffix, da=args.dataset) # type: ignore
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Read fixed image
    # f_img = sitk.ReadImage(args.atlas_file)
    # input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    # img = next(iter(DL))
    if args.front == "both":
        vol_size = (1024, 512)
    else:
        vol_size = (1024, 256)
    # # [B, C, D, W, H]
    # input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)
    # input_fixed = torch.from_numpy(input_fixed).to(device).float()

    # Creating Alignment Networks (UNet) and STNs
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    if args.checkpoint is not None:
        UNet.load_state_dict(torch.load(args.checkpoint))
    STN = SpatialTransformer(vol_size, func=args.func).to(device)
    UNet.train()
    STN.train()

    # Set optimizer and losses
    opt = AdamW(UNet.parameters(), lr=args.lr, weight_decay=0.05, eps=1e-08, betas=(0.9, 0.999))
    schlor = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

    sim_loss_fns = []
    for sim in args.sim_loss:
        if sim == "ncc":
            sim_loss_fns.append(losses.ncc())
        elif sim == "mse":
            sim_loss_fns.append(losses.mse_loss())
        elif sim == "ssim":
            sim_loss_fns.append(losses.ssim_loss())
        else:
            raise ValueError("Unknown similarity loss: %s" % args.sim_loss)
    
    if args.aux_loss == "nj":
        aux_loss_fn = losses.nj_loss(func=args.func)
    elif args.aux_loss == "bp":
        aux_loss_fn = losses.Bend_Penalty()
    elif args.aux_loss == "None":
        print("No auxiliary loss")
        aux_loss_fn = None
    else:
        raise ValueError("Unknown auxiliary loss: %s" % args.aux_loss)

    grad_loss_fn = losses.Grad()


    # Training loop.
    for i in range(1, args.epoch):

        epoch_loss = []
        epoch_total_loss = []
        for input_img in DL:  
            # Generate the moving images and convert them to tensors.
            # input_img = next(iter(DL))
            # [B, C, H, W]
            input_img = input_img.to(device).float()
            s = input_img.shape[3] // 2
            input_reverse = torch.cat([torch.flip(input_img[..., :s], dims=[3]), torch.flip(input_img[..., s:], dims=[3])], dim=3)
            
            if args.front == "True":
                input_fixed, input_moving = input_img[..., :s], input_reverse[..., :s]
            elif args.front == "False":
                input_fixed, input_moving = input_img[..., s:], input_reverse[..., s:]
            elif args.front == "both":
                input_fixed, input_moving = input_img, input_reverse
            else:
                raise ValueError("Unknown front: %s" % args.front)
            

            # Run the data through the model to produce warp and flow field
            flow_m2f = UNet(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)

            # Calculate loss
            loss_list = []
            loss = None
            for j, sim_loss_fn in enumerate(sim_loss_fns):
                if args.sim_loss == "ssim2":
                    sim_loss = sim_loss_fn(m2f, input_fixed, flow_m2f, input_moving) * args.weight[j]
                else:
                    sim_loss = sim_loss_fn(m2f, input_fixed) * args.weight[j]
                if j == 0:
                    loss = sim_loss
                else:
                    loss += sim_loss
                loss_list.append(sim_loss.item())

            grad_loss = grad_loss_fn(flow_m2f) * args.alpha
            loss += grad_loss
            loss_list.append(grad_loss.item())

            if aux_loss_fn is not None:
                aux_loss = aux_loss_fn(flow_m2f) * args.beta
                loss_list.append(aux_loss.item())
                loss += aux_loss

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

        schlor.step(np.mean(epoch_total_loss))

        epoch_info = 'Epoch %d/%d' % (i, args.epoch)
        # loss_info = 'loss: %.4e  sim_loss: %.4e  aux_loss: %.4e  grid_loss: %.4e' % (np.mean(epoch_total_loss), *np.mean(epoch_loss, axis=0))
        loss_names = args.sim_loss + ['grid_loss']
        loss_names = loss_names + ['aux_loss'] if aux_loss_fn is not None else loss_names
        loss_infos = ['%s: %.4e' % (name, value) for name, value in zip(loss_names, np.mean(epoch_loss, axis=0))]
        # loss_info = 'loss: %.4e  ' % np.mean(epoch_total_loss) + ', '.join(loss_infos)
        loss_info = 'loss: %.4e  ' % np.mean(epoch_total_loss) + 'lr: %.8e' % opt.param_groups[0]['lr'] + ', '.join(loss_infos)

        print('%s - %s' % (epoch_info, loss_info), file=f, flush=True)

        print(' - '.join((epoch_info, loss_info)), flush=True)
        # f.write(' - '.join((epoch_info, loss_info)) + '\n')

        if (i + 1) % args.n_save_iter == 0:
            # Save model checkpoint
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(UNet.state_dict(), save_file_name)

    f.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
