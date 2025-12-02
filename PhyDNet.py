# file: PhyDNet.py
import os
import h5py
import logging
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from einops import rearrange
import argparse
import random  
from utilities import * 
from rnn_models import ConvLSTM, PhyCell, PhyDNet, FrameLoss
from constrain_moments import K2M


# hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model_name", type = str, default = "PhyDNet")
parser.add_argument('-id', "--model_id", type = int, default = 0)
parser.add_argument('-r', "--random_seed", type = int, default = 0)
parser.add_argument('-bz', "--batch_size", type = int, default = 32)
parser.add_argument('-lr', "--learning_rate", type = float, default = 1e-4)
parser.add_argument('-e', "--num_epochs", type = int, default = 300)
parser.add_argument('-strd', "--stride", type = int, default = 1)
parser.add_argument('-hd1', "--hidden_dim1", type = int, default = 128)
parser.add_argument('-hd2', "--hidden_dim2", type = int, default = 128)

parser.add_argument('-df', "--disp_every_batch", type = int, default = 100)
parser.add_argument('-sf', "--save_pred_every_epoch", type = int, default = 10)
parser.add_argument('-smd', "--save_model_every_epoch", type = int, default = 5)
parser.add_argument('-ldm', "--load_model", type = str, default = '')
parser.add_argument('-msg', "--message", type = str, default = '')
args = parser.parse_args()


# test code
if os.name == 'nt':
    args.batch_size = 4
    args.disp_every_batch = 1
    args.save_pred_every_epoch = 1
    args.hidden_dim1 = 8
    args.hidden_dim2 = 8

# setup logger
script_name = os.path.basename(__file__).split('.')[0] + f"_{args.model_id:02}"
setup_logger(script_name, args)
# save hyperparameters
save_hyperparam(args)
# setup random seed and device
set_random_seed(args.random_seed)
device = select_device(req_mem = 5000)

# constant matrix for physical differential constraints
constraints = torch.zeros((49,7,7)).to(device)
ind = 0
for i in range(0,7):
    for j in range(0,7):
        constraints[ind,i,j] = 1
        ind +=1 

dataset = SkyVideoDataset(stride=args.stride)

train_dataset, val_dataset, test_dataset = split_dataset_by_date(dataset)

logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Val   dataset size: {len(val_dataset)}")
logging.info(f"Test  dataset size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)


sampling_step_1 = 15
sampling_step_2 = 30
r_exp_alpha = 2.5


def reserve_schedule_sampling_exp(epoch_idx, log_length):
    real_input_flag_encoder = np.zeros(log_length, dtype=bool)
    if epoch_idx < sampling_step_1:
        r_eta = 0.5
    elif epoch_idx < sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(epoch_idx - sampling_step_1) / r_exp_alpha)
    else:
        r_eta = 1.0

    for i in range(log_length):
        real_input_flag_encoder[i] = True if random.random() < r_eta else False
    return r_eta, real_input_flag_encoder


def schedule_sampling(epoch_idx,pred_length):
    real_input_flag_decoder = np.zeros(pred_length, dtype=bool)
    if epoch_idx < sampling_step_1:
        eta = 0.5
    elif epoch_idx < sampling_step_2:
        eta = 0.5 - (0.5 / (sampling_step_2 - sampling_step_1)) * (epoch_idx - sampling_step_1)
    else:
        eta = 0

    for i in range(pred_length):
        real_input_flag_decoder[i] = True if random.random() < eta else False
    return eta, real_input_flag_decoder


phycell  =  PhyCell(input_shape=(32,32), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
convcell =  ConvLSTM(input_shape=(32,32), input_dim=64, hidden_dims=[args.hidden_dim1,args.hidden_dim2,64], n_layers=3, kernel_size=(3,3), device=device)   
encoder  = PhyDNet(phycell, convcell, device)
encoder = encoder.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.info(f'phycell  #param: {count_parameters(phycell)}')    
logging.info(f'convcell #param: {count_parameters(convcell)}') 
logging.info(f'encoder  #param: {count_parameters(encoder)}') 

if args.load_model:
    try:
        encoder.load_state_dict(torch.load(args.load_model, map_location=device, weights_only=True))
        logging.info(f'Load encoder model from {args.load_model}')
    except:
        logging.info(f'Fail to load model')

# criterion_mae = nn.L1Loss()
criterion_mse = nn.MSELoss().to(device)
criterion_FrameLoss = FrameLoss().to(device)


def train_on_batch(batch_idx, epoch_idx, loaddata, encoder, encoder_optimizer):                
    encoder.train()
    encoder_optimizer.zero_grad()
    
    # [B T C H W]
    stidx, input_frames, target_frames, _, _ = loaddata

    input_length  = input_frames.size(1)
    target_length = target_frames.size(1)
    batch_loss = 0.0

    r_eta,real_input_flag_encoder = reserve_schedule_sampling_exp(epoch_idx, input_length)
    eta,real_input_flag_decoder = schedule_sampling(epoch_idx, target_length)
    
    encoder_input_img = input_frames[:,0]
    
    for ei in range(input_length-1): 

        encoder_output_img = encoder(encoder_input_img, first_timestep=(ei==0))

        encoder_target_img = input_frames[:,ei+1]
        batch_loss += criterion_FrameLoss(encoder_output_img, encoder_target_img[:,:3])
        
        if real_input_flag_encoder[ei]:  # Teacher forcing
            encoder_input_img = encoder_target_img
        else:
            sun_mask = encoder_target_img[:, 3].unsqueeze(1)
            encoder_input_img = torch.cat([encoder_output_img, sun_mask], dim=1)
    
    if real_input_flag_encoder[-1]:  # Teacher forcing
        decoder_input_img = input_frames[:, -1] 
    else:
        sun_mask = input_frames[:, -1, 3].unsqueeze(1)
        decoder_input_img = torch.cat([encoder_output_img, sun_mask], dim=1)
    
    for di in range(target_length):
        decoder_output_img = encoder(decoder_input_img)

        decoder_target_img = target_frames[:, di]
        batch_loss += criterion_FrameLoss(decoder_output_img, decoder_target_img[:,:3])
        
        if real_input_flag_decoder[di]:  # Teacher forcing
            decoder_input_img = decoder_target_img
        else:
            sun_mask = decoder_target_img[:, 3].unsqueeze(1)
            decoder_input_img = torch.cat([decoder_output_img, sun_mask], dim=1)
    
    k2m = K2M([7,7]).to(device)
    for b in range(0,encoder.phycell.cell_list[0].input_dim):
        filters = encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)     
        m = k2m(filters.double()) 
        m  = m.float()   
        batch_loss += criterion_mse(m, constraints) # constrains is a precomputed matrix   
    
    batch_loss.backward()
    encoder_optimizer.step()
    
    batch_loss = batch_loss.item() / (input_length + target_length)

    return batch_loss


def trainIters(encoder):
    trian_loss_list = []

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=3, factor=0.3)
    
    for epoch_idx in range(0, args.num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, loaddata in enumerate(train_loader, 0):

            loaddata = [data.to(device) for data in loaddata]
            batch_loss = train_on_batch(batch_idx, epoch_idx, loaddata, encoder, encoder_optimizer)  

            epoch_loss += batch_loss

            if batch_idx==0 or (batch_idx+1) % args.disp_every_batch == 0:
                logging.info(f'Epoch [{epoch_idx+1}/{args.num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {batch_loss:8f}')
        
        epoch_loss /= len(train_loader)

        trian_loss_list.append(epoch_loss)     
        logging.info(f'Epoch [{epoch_idx+1}/{args.num_epochs}] Epoch Loss: {epoch_loss:8f}')
        
        prediction_imgs, eval_stidx, eval_loss = evaluate(encoder, val_loader)
        scheduler_enc.step(eval_loss)

        if (epoch_idx+1) % args.save_model_every_epoch == 0:
            model_save_path = os.path.join(os.path.dirname(__file__), 'weights', f'{args.model_name}_{args.model_id:02}')
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(encoder.state_dict(), os.path.join(model_save_path, f'encoder_{(epoch_idx+1):03}.pth')) 
            torch.save(encoder.state_dict(), os.path.join(model_save_path, 'encoder.pth')) 
            logging.info(f"Saved encoder models to {model_save_path}")

        if (epoch_idx+1) % args.save_pred_every_epoch == 0:
            prediction_imgs = rearrange(prediction_imgs, 'n_samples n_target_frames ch h w -> n_samples n_target_frames h w ch')
            prediction_imgs = prediction_imgs * 255.0
            prediction_imgs = prediction_imgs.astype(np.uint8) 

            pred_result_save_file = os.path.join(os.path.dirname(__file__), 'results', f'{args.model_name}_{args.model_id:02}', f'predicted_frames_val.h5')
            os.makedirs(os.path.dirname(pred_result_save_file), exist_ok=True)
            with h5py.File(pred_result_save_file, "w") as f:
                f.create_dataset("prediction_imgs", data = prediction_imgs, compression = "lzf")
                f.create_dataset("eval_stidx", data = eval_stidx, compression = "lzf")

            logging.info(f'Save validation prediction results to {pred_result_save_file}')
    
    test_prediction_imgs, test_eval_stidx, test_loss = evaluate(encoder, test_loader)
    logging.info(f'Final Test Loss: {test_loss:.8f}')

    test_prediction_imgs = rearrange(test_prediction_imgs, 'n_samples n_target_frames ch h w -> n_samples n_target_frames h w ch')
    test_prediction_imgs = (test_prediction_imgs * 255.0).astype(np.uint8)
    test_result_file = os.path.join(os.path.dirname(__file__), 'results', f'{args.model_name}_{args.model_id:02}', f'predicted_frames_test.h5')
    os.makedirs(os.path.dirname(test_result_file), exist_ok=True)
    with h5py.File(test_result_file, "w") as f:
        f.create_dataset("prediction_imgs", data = test_prediction_imgs, compression = "lzf")
        f.create_dataset("eval_stidx", data = test_eval_stidx, compression = "lzf")
    logging.info(f'Save test prediction results to {test_result_file}')

    return trian_loss_list


@torch.no_grad()
def evaluate(encoder, test_loader):
    eval_loss = 0.0

    eval_stidx = []
    prediction_imgs = []

    for i, loaddata in enumerate(test_loader, 0):

        loaddata = [data.to(device) for data in loaddata]
        stidx, input_frames, target_frames, _, _ = loaddata

        input_length  = input_frames.size(1)
        target_length = target_frames.size(1)

        for ei in range(input_length-1):
            encoder_input_img = input_frames[:,ei]
            _ = encoder(encoder_input_img, first_timestep=(ei==0))

        decoder_input_img = input_frames[:,-1] # first decoder input= last image of input sequence

        prediction_img = []
        batch_loss = 0.0
        
        for di in range(target_length):
            decoder_output_img = encoder(decoder_input_img)

            decoder_target_img = target_frames[:, di]
            batch_loss += criterion_FrameLoss(decoder_output_img, decoder_target_img[:,:3])

            sun_mask = decoder_target_img[:, 3].unsqueeze(1)
            decoder_input_img = torch.cat([decoder_output_img, sun_mask], dim=1)

            prediction_img.append(decoder_output_img.cpu())


        prediction_img = np.stack(prediction_img) # (T, B, 3, H, W)
        prediction_img = rearrange(prediction_img, 'T B C H W -> B T C H W')
        
        eval_stidx.append(stidx.cpu()) # store start_index of each batch
        prediction_imgs.append(prediction_img) # store prediction of each batch

        eval_loss += batch_loss.item() / target_length

    eval_stidx = np.concatenate(eval_stidx, axis=0)
    prediction_imgs =  np.concatenate(prediction_imgs, axis=0)

    eval_loss /= len(test_loader)

    logging.info(f'Eval Loss: {eval_loss:.8f}')

    return prediction_imgs, eval_stidx, eval_loss


trian_loss_list = trainIters(encoder)
