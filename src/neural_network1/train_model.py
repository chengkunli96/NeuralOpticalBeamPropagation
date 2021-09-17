import argparse
import logging
import os
import sys
import numpy as np
import time
from os.path import join as opj

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import shutil

from model.net.model import Net as Net
from utils.eval import eval_net
from utils.dataset import BasicDataset



def train_net(net,
              device,
              dir_dataset: list,
              dir_out,
              writer,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              tensorboard_txt=None):

    if save_cp:
        dir_checkpoint = opj(dir_out, 'checkpoints')
    net.to(device=device)

    dataset = BasicDataset(dir_dataset, mode='normal')
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    batch = next(iter(train_loader))
    amplitude_input = batch['amplitude_input']
    phase_input = batch['phase_input']
    input_data = torch.cat((amplitude_input, phase_input), 1)
    input_data = input_data.to(device=device, dtype=torch.float32)
    writer.add_graph(net, input_data)

    global_step = 0
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    logtxt = f'''Training Setting:\n
        Epochs:          {epochs}\n
        Batch size:      {batch_size}\n
        Learning rate:   {lr}\n
        Training size:   {n_train}\n
        Validation size: {n_val}\n
        Checkpoints:     {save_cp}\n
        Device:          {device.type}\n
    '''

    writer.add_text('train_detail', logtxt)
    if tensorboard_txt is not None:
        writer.add_text('addition', tensorboard_txt)

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = MyLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                amplitude_input = batch['amplitude_input']  # shape: BCHW
                phase_input = batch['phase_input']
                amplitude_target = batch['amplitude_target']
                phase_target = batch['phase_target']

                input_data = torch.cat((amplitude_input, phase_input), 1)
                target_data = torch.cat((amplitude_target, phase_target), 1)

                assert input_data.shape[1] == net.in_channels, \
                    f'Network has been defined with {net.in_channels} image-input channels, ' \
                    f'but loaded images have {input_data.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                input_data = input_data.to(device=device, dtype=torch.float32)
                target_data = target_data.to(device=device, dtype=torch.float32)

                pred_data, pred_supplement = net(input_data)
                loss = criterion(pred_data, target_data)
                supplement_loss = criterion(pred_supplement, target_data)
                loss = loss + 0.1 * supplement_loss
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Epoch', epoch, global_step)
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # BP
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                # show in tensorboard
                pbar.update(input_data.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:  # print for every 10 percent
                    val_score = eval_net(net, val_loader, device, criterion, middleout=True)
                    scheduler.step(val_score)  # adjust the learning rate
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation Loss: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)

                    # tensorboard(tb) images
                    tb_amplitude_input = input_data.data[:, 0, :, :].unsqueeze(1)
                    tb_phase_input = input_data.data[:, 1, :, :].unsqueeze(1)
                    tb_amplitude_target = target_data.data[:, 0, :, :].unsqueeze(1)
                    tb_phase_target = target_data.data[:, 1, :, :].unsqueeze(1)
                    tb_amplitude_pred = pred_data.data[:, 0, :, :].unsqueeze(1)
                    tb_phase_pred = pred_data.data[:, 1, :, :].unsqueeze(1)
                    writer.add_images('input/amplitude', tb_amplitude_input, global_step)
                    writer.add_images('input/phase', tb_phase_input, global_step)
                    writer.add_images('target/amplitude', tb_amplitude_target, global_step)
                    writer.add_images('target/phase', tb_phase_target, global_step)
                    writer.add_images('predict/amplitude', tb_amplitude_pred, global_step)
                    writer.add_images('predict/phase', tb_phase_pred, global_step)

        # save for each epoch
        if save_cp:
            if not os.path.exists(dir_checkpoint):
                os.makedirs(dir_checkpoint)
            torch.save(net.state_dict(),
                       opj(dir_checkpoint, f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')  # pre-trained model
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-d', '--device', dest='device', type=int, default=0,
                        help='gpu device')
    return parser.parse_args()


CURR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
dir_output = opj(CURR_FILE_PATH, '../../output')

dataset0_pth = opj(CURR_FILE_PATH, '../../data/processed/lfw_nn1')

# in terminal, you can run
# python train_model.py --epochs 10  --batch-size 4  --learning-rate 0.001 --validation 20 --device 0

if __name__ == '__main__':
    args = get_args()
    # basic logging config
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # ===== get the experiment number =====
    # get all dirname under dir_output folder
    dirs = os.listdir(dir_output)
    dir_exps = []
    for dirname in dirs:
        if len(dirname) == 5 and dirname[:-2] == 'exp':
            dir_exps.append(dirname)
    # set the experiment number:
    ep = -1
    for dirname in dir_exps:
        num = int(dirname[-2:])
        if num > ep:
            ep = num
    ep = ep + 1
    # dir_checkpoint name
    dir_exp_out = opj(dir_output, 'exp{:02d}'.format(ep))

    # ===== training config =====
    # set gpu and cpu setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logging.info(f'Using device {device} {args.device}')
    else:
        logging.info(f'Using device cpu')
    # tensorboard setting
    expnum = int(dir_exp_out[-2:])
    logdir = opj(dir_exp_out, 'runs/exp' + f'_LR_{args.lr}_BS_{args.batchsize}')
    writer = SummaryWriter(log_dir=logdir)
    logging.info(f'Running experiment {expnum}.')
    # The additional information you want to add in the tensorboard page.
    additiontxt = '''Addition:\n
               Model: res-unet + (middle supervisor) + dbpnet. 2 channels in and 2 channels out.\n
               Training: angle 10, distance 180\n
               Loss: MSE\n
           '''
    # dataset set
    dataset_pth = dataset0_pth
    logging.info(f'Dataset is :{dataset_pth}.')

    # ===== network setting =====
    # set neural network
    net = Net(in_channels=2, out_channels=2)
    logging.info(f'Residual-U-Network + DBP-Network:\n'
                 f'\t{net.in_channels} input img channels\n'
                 f'\t{net.out_channels} output channels')

    if args.load:  # load the pre-trained net
        if os.path.isabs(args.load):
            model_file = args.load
        else:
            model_file = opj(CURR_FILE_PATH, args.load)
        net.load_state_dict(
            torch.load(model_file, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    # faster convolutions, but more memory
    # cudnn.benchmark = True

    # ===== starting training =====
    start = time.time()
    try:
        train_net(net=net,
                  dir_dataset=[dataset_pth],
                  dir_out=dir_exp_out,
                  writer=writer,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  save_cp=True,
                  tensorboard_txt=additiontxt)
        end = time.time()
        running_time = end - start
        logging.info("The program's running time is: {:.2f} hours".format(running_time / 60 / 60))
    except KeyboardInterrupt:
        writer.close()
        end = time.time()
        running_time = (end - start) / 60 / 60
        logging.info("The program's running time is: {:.2f} hours".format(running_time))
        if running_time > 0.5:
            if not os.path.exists(dir_exp_out):
                os.makedirs(dir_exp_out)
            torch.save(net.state_dict(), opj(dir_exp_out, 'checkpoints', 'INTERRUPTED.pth'))
            logging.info('Saved interrupt')
        else:
            if os.path.exists(dir_exp_out):
                shutil.rmtree(dir_exp_out)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
