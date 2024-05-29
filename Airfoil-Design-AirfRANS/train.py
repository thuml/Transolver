import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time, json

import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from pathlib import Path
import os.path as osp


def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, criterion='MSE', reg=1):
    model.train()
    avg_loss_per_var = torch.zeros(4, device=device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4, device=device)
    avg_loss_vol_var = torch.zeros(4, device=device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iter = 0

    for data in train_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)
        optimizer.zero_grad()
        out = model(data_clone)
        targets = data_clone.y

        if criterion == 'MSE' or criterion == 'MSE_weighted':
            loss_criterion = nn.MSELoss(reduction='none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction='none')
        loss_per_var = loss_criterion(out, targets).mean(dim=0)
        total_loss = loss_per_var.mean()
        loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim=0)
        loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim=0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()

        if criterion == 'MSE_weighted':
            (loss_vol + reg * loss_surf).backward()
        else:
            total_loss.backward()

        optimizer.step()
        scheduler.step()
        avg_loss_per_var += loss_per_var
        avg_loss += total_loss
        avg_loss_surf_var += loss_surf_var
        avg_loss_vol_var += loss_vol_var
        avg_loss_surf += loss_surf
        avg_loss_vol += loss_vol
        iter += 1

    return avg_loss.cpu().data.numpy() / iter, avg_loss_per_var.cpu().data.numpy() / iter, avg_loss_surf_var.cpu().data.numpy() / iter, avg_loss_vol_var.cpu().data.numpy() / iter, \
           avg_loss_surf.cpu().data.numpy() / iter, avg_loss_vol.cpu().data.numpy() / iter


@torch.no_grad()
def test(device, model, test_loader, criterion='MSE'):
    model.eval()
    avg_loss_per_var = np.zeros(4)
    avg_loss = 0
    avg_loss_surf_var = np.zeros(4)
    avg_loss_vol_var = np.zeros(4)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iter = 0

    for data in test_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)
        out = model(data_clone)

        targets = data_clone.y
        if criterion == 'MSE' or 'MSE_weighted':
            loss_criterion = nn.MSELoss(reduction='none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction='none')

        loss_per_var = loss_criterion(out, targets).mean(dim=0)
        loss = loss_per_var.mean()
        loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim=0)
        loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim=0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()

        avg_loss_per_var += loss_per_var.cpu().numpy()
        avg_loss += loss.cpu().numpy()
        avg_loss_surf_var += loss_surf_var.cpu().numpy()
        avg_loss_vol_var += loss_vol_var.cpu().numpy()
        avg_loss_surf += loss_surf.cpu().numpy()
        avg_loss_vol += loss_vol.cpu().numpy()
        iter += 1

    return avg_loss / iter, avg_loss_per_var / iter, avg_loss_surf_var / iter, avg_loss_vol_var / iter, avg_loss_surf / iter, avg_loss_vol / iter


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, val_dataset, Net, hparams, path, criterion='MSE', reg=1, val_iter=10,
         name_mod='GraphSAGE', val_sample=True):
    '''
        Args:
        device (str): device on which you want to do the computation.
        train_dataset (list): list of the data in the training set.
        val_dataset (list): list of the data in the validation set.
        Net (class): network to train.
        hparams (dict): hyper parameters of the network.
        path (str): where to save the trained model and the figures.
        criterion (str, optional): chose between 'MSE', 'MAE', and 'MSE_weigthed'. The latter is the volumetric MSE plus the surface MSE computed independently. Default: 'MSE'.
        reg (float, optional): weigth for the surface loss when criterion is 'MSE_weighted'. Default: 1.
        val_iter (int, optional): number of epochs between each validation step. Default: 10.
        name_mod (str, optional): type of model. Default: 'GraphSAGE'.
    '''
    Path(path).mkdir(parents=True, exist_ok=True)

    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
    )
    val_loader = DataLoader(val_dataset, batch_size=1)
    start = time.time()

    train_loss_surf_list = []
    train_loss_vol_list = []
    loss_surf_var_list = []
    loss_vol_var_list = []
    val_surf_list = []
    val_vol_list = []
    val_surf_var_list = []
    val_vol_var_list = []

    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
            idx = torch.tensor(idx)

            data_sampled.pos = data_sampled.pos[idx]
            data_sampled.x = data_sampled.x[idx]
            data_sampled.y = data_sampled.y[idx]
            data_sampled.surf = data_sampled.surf[idx]

            if name_mod != 'PointNet' and name_mod != 'MLP':
                data_sampled.edge_index = nng.radius_graph(x=data_sampled.pos.to(device), r=hparams['r'], loop=True,
                                                           max_num_neighbors=int(hparams['max_neighbors'])).cpu()

            train_dataset_sampled.append(data_sampled)
        train_loader = DataLoader(train_dataset_sampled, batch_size=hparams['batch_size'], shuffle=True)
        del (train_dataset_sampled)

        train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = train(device, model, train_loader, optimizer,
                                                                                lr_scheduler, criterion, reg=reg)
        print('epoch: ' + str(epoch))
        print('train_loss： ' + str(train_loss))
        print('loss_vol： ' + str(loss_vol))
        print('loss_surf： ' + str(loss_surf))

        if criterion == 'MSE_weighted':
            train_loss = reg * loss_surf + loss_vol
        del (train_loader)

        train_loss_surf_list.append(loss_surf)
        train_loss_vol_list.append(loss_vol)
        loss_surf_var_list.append(loss_surf_var)
        loss_vol_var_list.append(loss_vol_var)

        if val_iter is not None:
            if epoch % val_iter == val_iter - 1 or epoch == 0:
                if val_sample:
                    val_surf_vars, val_vol_vars, val_surfs, val_vols = [], [], [], []
                    for i in range(20):
                        val_dataset_sampled = []
                        for data in val_dataset:
                            data_sampled = data.clone()
                            idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
                            idx = torch.tensor(idx)

                            data_sampled.pos = data_sampled.pos[idx]
                            data_sampled.x = data_sampled.x[idx]
                            data_sampled.y = data_sampled.y[idx]
                            data_sampled.surf = data_sampled.surf[idx]

                            if name_mod != 'PointNet' and name_mod != 'MLP':
                                data_sampled.edge_index = nng.radius_graph(x=data_sampled.pos.to(device),
                                                                           r=hparams['r'], loop=True,
                                                                           max_num_neighbors=int(
                                                                               hparams['max_neighbors'])).cpu()

                                # if name_mod == 'GNO' or name_mod == 'MGNO':
                                #     x, edge_index = data_sampled.x, data_sampled.edge_index
                                #     x_i, x_j = x[edge_index[0], 0:2], x[edge_index[1], 0:2]
                                #     v_i, v_j = x[edge_index[0], 2:4], x[edge_index[1], 2:4]
                                #     p_i, p_j = x[edge_index[0], 4:5], x[edge_index[1], 4:5]
                                #     v_inf = torch.linalg.norm(v_i, dim = 1, keepdim = True)
                                #     sdf_i, sdf_j = x[edge_index[0], 5:6], x[edge_index[1], 5:6]
                                #     normal_i, normal_j = x[edge_index[0], 6:8], x[edge_index[1], 6:8]

                                #     data_sampled.edge_attr = torch.cat([x_i - x_j, v_i - v_j, p_i - p_j, sdf_i, sdf_j, v_inf, normal_i, normal_j], dim = 1)

                            val_dataset_sampled.append(data_sampled)
                        val_loader = DataLoader(val_dataset_sampled, batch_size=1, shuffle=True)
                        del (val_dataset_sampled)

                        val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(device, model, val_loader,
                                                                                         criterion)
                        del (val_loader)
                        val_surf_vars.append(val_surf_var)
                        val_vol_vars.append(val_vol_var)
                        val_surfs.append(val_surf)
                        val_vols.append(val_vol)
                    val_surf_var = np.array(val_surf_vars).mean(axis=0)
                    val_vol_var = np.array(val_vol_vars).mean(axis=0)
                    val_surf = np.array(val_surfs).mean(axis=0)
                    val_vol = np.array(val_vols).mean(axis=0)
                else:
                    val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(device, model, val_loader,
                                                                                     criterion)
                print("=====validation=====")
                print('epoch: ' + str(epoch))
                print('val_vol： ' + str(val_vol))
                print('val_surf： ' + str(val_surf))
                if criterion == 'MSE_weigthed':
                    val_loss = reg * val_surf + val_vol
                val_surf_list.append(val_surf)
                val_vol_list.append(val_vol)
                val_surf_var_list.append(val_surf_var)
                val_vol_var_list.append(val_vol_var)
                pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf, val_loss=val_loss, val_surf=val_surf)
            else:
                pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf, val_loss=val_loss, val_surf=val_surf)
        else:
            pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf)

    loss_surf_var_list = np.array(loss_surf_var_list)
    loss_vol_var_list = np.array(loss_vol_var_list)
    val_surf_var_list = np.array(val_surf_var_list)
    val_vol_var_list = np.array(val_vol_var_list)

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    torch.save(model, osp.join(path, 'model'))

    sns.set()
    fig_train_surf, ax_train_surf = plt.subplots(figsize=(20, 5))
    ax_train_surf.plot(train_loss_surf_list, label='Mean loss')
    ax_train_surf.plot(loss_surf_var_list[:, 0], label=r'$v_x$ loss')
    ax_train_surf.plot(loss_surf_var_list[:, 1], label=r'$v_y$ loss')
    ax_train_surf.plot(loss_surf_var_list[:, 2], label=r'$p$ loss')
    ax_train_surf.plot(loss_surf_var_list[:, 3], label=r'$\nu_t$ loss')
    ax_train_surf.set_xlabel('epochs')
    ax_train_surf.set_yscale('log')
    ax_train_surf.set_title('Train losses over the surface')
    ax_train_surf.legend(loc='best')
    fig_train_surf.savefig(osp.join(path, 'train_loss_surf.png'), dpi=150, bbox_inches='tight')

    fig_train_vol, ax_train_vol = plt.subplots(figsize=(20, 5))
    ax_train_vol.plot(train_loss_vol_list, label='Mean loss')
    ax_train_vol.plot(loss_vol_var_list[:, 0], label=r'$v_x$ loss')
    ax_train_vol.plot(loss_vol_var_list[:, 1], label=r'$v_y$ loss')
    ax_train_vol.plot(loss_vol_var_list[:, 2], label=r'$p$ loss')
    ax_train_vol.plot(loss_vol_var_list[:, 3], label=r'$\nu_t$ loss')
    ax_train_vol.set_xlabel('epochs')
    ax_train_vol.set_yscale('log')
    ax_train_vol.set_title('Train losses over the volume')
    ax_train_vol.legend(loc='best')
    fig_train_vol.savefig(osp.join(path, 'train_loss_vol.png'), dpi=150, bbox_inches='tight')

    if val_iter is not None:
        fig_val_surf, ax_val_surf = plt.subplots(figsize=(20, 5))
        ax_val_surf.plot(val_surf_list, label='Mean loss')
        ax_val_surf.plot(val_surf_var_list[:, 0], label=r'$v_x$ loss')
        ax_val_surf.plot(val_surf_var_list[:, 1], label=r'$v_y$ loss')
        ax_val_surf.plot(val_surf_var_list[:, 2], label=r'$p$ loss')
        ax_val_surf.plot(val_surf_var_list[:, 3], label=r'$\nu_t$ loss')
        ax_val_surf.set_xlabel('epochs')
        ax_val_surf.set_yscale('log')
        ax_val_surf.set_title('Validation losses over the surface')
        ax_val_surf.legend(loc='best')
        fig_val_surf.savefig(osp.join(path, 'val_loss_surf.png'), dpi=150, bbox_inches='tight')

        fig_val_vol, ax_val_vol = plt.subplots(figsize=(20, 5))
        ax_val_vol.plot(val_vol_list, label='Mean loss')
        ax_val_vol.plot(val_vol_var_list[:, 0], label=r'$v_x$ loss')
        ax_val_vol.plot(val_vol_var_list[:, 1], label=r'$v_y$ loss')
        ax_val_vol.plot(val_vol_var_list[:, 2], label=r'$p$ loss')
        ax_val_vol.plot(val_vol_var_list[:, 3], label=r'$\nu_t$ loss')
        ax_val_vol.set_xlabel('epochs')
        ax_val_vol.set_yscale('log')
        ax_val_vol.set_title('Validation losses over the volume')
        ax_val_vol.legend(loc='best')
        fig_val_vol.savefig(osp.join(path, 'val_loss_vol.png'), dpi=150, bbox_inches='tight')

        if val_iter is not None:
            with open(osp.join(path, name_mod + '_log.json'), 'a') as f:
                json.dump(
                    {
                        'regression': 'Total',
                        'loss': 'MSE',
                        'nb_parameters': params_model,
                        'time_elapsed': time_elapsed,
                        'hparams': hparams,
                        'train_loss_surf': train_loss_surf_list[-1],
                        'train_loss_surf_var': loss_surf_var_list[-1],
                        'train_loss_vol': train_loss_vol_list[-1],
                        'train_loss_vol_var': loss_vol_var_list[-1],
                        'val_loss_surf': val_surf_list[-1],
                        'val_loss_surf_var': val_surf_var_list[-1],
                        'val_loss_vol': val_vol_list[-1],
                        'val_loss_vol_var': val_vol_var_list[-1],
                    }, f, indent=12, cls=NumpyEncoder
                )

    return model
