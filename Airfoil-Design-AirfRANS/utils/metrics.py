import os.path as osp
import pathlib

import numpy as np
import scipy as sc
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader

import pyvista as pv
import json
import seaborn as sns
import random
import time

import utils.metrics_NACA as metrics_NACA
from utils.reorganize import reorganize
from dataset.dataset import Dataset

from tqdm import tqdm

NU = np.array(1.56e-5)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def rsquared(predict, true):
    '''
    Args:
        predict (tensor): Predicted values, shape (N, *)
        true (tensor): True values, shape (N, *)

    Out:
        rsquared (tensor): Coefficient of determination of the prediction, shape (*,)
    '''
    mean = true.mean(dim=0)
    return 1 - ((true - predict) ** 2).sum(dim=0) / ((true - mean) ** 2).sum(dim=0)


def rel_err(a, b):
    return np.abs((a - b) / a)


def WallShearStress(Jacob_U, normals):
    S = .5 * (Jacob_U + Jacob_U.transpose(0, 2, 1))
    S = S - S.trace(axis1=1, axis2=2).reshape(-1, 1, 1) * np.eye(2)[None] / 3
    ShearStress = 2 * NU.reshape(-1, 1, 1) * S
    ShearStress = (ShearStress * normals[:, :2].reshape(-1, 1, 2)).sum(axis=2)

    return ShearStress


@torch.no_grad()
def Infer_test(device, models, hparams, data, coef_norm=None):
    # Inference procedure on new simulation
    outs = [torch.zeros_like(data.y)] * len(models)
    n_out = torch.zeros_like(data.y[:, :1])
    idx_points = set(map(tuple, data.pos[:, :2].numpy()))
    cond = True
    i = 0
    while cond:
        i += 1
        data_sampled = data.clone()
        idx = random.sample(range(data_sampled.x.size(0)), hparams[0]['subsampling'])
        idx = torch.tensor(idx)
        idx_points = idx_points - set(map(tuple, data_sampled.pos[idx, :2].numpy()))
        data_sampled.pos = data_sampled.pos[idx]
        data_sampled.x = data_sampled.x[idx]
        data_sampled.y = data_sampled.y[idx]
        data_sampled.surf = data_sampled.surf[idx]
        data_sampled.batch = data_sampled.batch[idx]

        # try:
        #     data_sampled.edge_index = nng.radius_graph(x = data_sampled.pos.to(device), r = hparams['r'], loop = True, max_num_neighbors = int(hparams['max_neighbors'])).cpu()
        # except KeyError:
        #     None

        out = [torch.zeros_like(data.y)] * len(models)
        tim = np.zeros(len(models))
        for n, model in enumerate(models):
            try:
                data_sampled.edge_index = nng.radius_graph(x=data_sampled.pos.to(device), r=hparams[n]['r'], loop=True,
                                                           max_num_neighbors=int(hparams[n]['max_neighbors'])).cpu()
            except KeyError:
                data_sampled.edge_index = None

            model.eval()
            data_sampled = data_sampled.to(device)
            start = time.time()
            o = model(data_sampled)
            tim[n] += time.time() - start
            out[n][idx] = o.cpu()

            outs[n] = outs[n] + out[n]
        n_out[idx] = n_out[idx] + torch.ones_like(n_out[idx])

        cond = (len(idx_points) > 0)

    for n, out in enumerate(outs):
        outs[n] = out / n_out
        if coef_norm is not None:
            outs[n][data.surf, :2] = -torch.tensor(coef_norm[2][None, :2]) * torch.ones_like(out[data.surf, :2]) / (
                    torch.tensor(coef_norm[3][None, :2]) + 1e-8)
            outs[n][data.surf, 3] = -torch.tensor(coef_norm[2][3]) * torch.ones_like(out[data.surf, 3]) / (
                    torch.tensor(coef_norm[3][3]) + 1e-8)
        else:
            outs[n][data.surf, :2] = torch.zeros_like(out[data.surf, :2])
            outs[n][data.surf, 3] = torch.zeros_like(out[data.surf, 3])

    return outs, tim / i


def Airfoil_test(internal, airfoil, outs, coef_norm, bool_surf):
    # Produce multiple copies of a simulation for different predictions.
    # stocker les internals, airfoils, calculer le wss, calculer le drag, le lift, plot pressure coef, plot skin friction coef, plot drag/drag, plot lift/lift
    # calcul spearsman coef, boundary layer
    internals = []
    airfoils = []
    for out in outs:
        intern = internal.copy()
        aerofoil = airfoil.copy()

        point_mesh = intern.points[bool_surf, :2]
        point_surf = aerofoil.points[:, :2]
        out = (out * (coef_norm[3] + 1e-8) + coef_norm[2]).numpy()
        out[bool_surf.numpy(), :2] = np.zeros_like(out[bool_surf.numpy(), :2])
        out[bool_surf.numpy(), 3] = np.zeros_like(out[bool_surf.numpy(), 3])
        intern.point_data['U'][:, :2] = out[:, :2]
        intern.point_data['p'] = out[:, 2]
        intern.point_data['nut'] = out[:, 3]

        surf_p = intern.point_data['p'][bool_surf]
        surf_p = reorganize(point_mesh, point_surf, surf_p)
        aerofoil.point_data['p'] = surf_p

        intern = intern.ptc(pass_point_data=True)
        aerofoil = aerofoil.ptc(pass_point_data=True)

        internals.append(intern)
        airfoils.append(aerofoil)

    return internals, airfoils


def Airfoil_mean(internals, airfoils):
    # Average multiple prediction over one simulation

    oi_point = np.zeros((internals[0].points.shape[0], 4))
    oi_cell = np.zeros((internals[0].cell_data['U'].shape[0], 4))
    oa_point = np.zeros((airfoils[0].points.shape[0], 4))
    oa_cell = np.zeros((airfoils[0].cell_data['U'].shape[0], 4))

    for k in range(len(internals)):
        oi_point[:, :2] += internals[k].point_data['U'][:, :2]
        oi_point[:, 2] += internals[k].point_data['p']
        oi_point[:, 3] += internals[k].point_data['nut']
        oi_cell[:, :2] += internals[k].cell_data['U'][:, :2]
        oi_cell[:, 2] += internals[k].cell_data['p']
        oi_cell[:, 3] += internals[k].cell_data['nut']

        oa_point[:, :2] += airfoils[k].point_data['U'][:, :2]
        oa_point[:, 2] += airfoils[k].point_data['p']
        oa_point[:, 3] += airfoils[k].point_data['nut']
        oa_cell[:, :2] += airfoils[k].cell_data['U'][:, :2]
        oa_cell[:, 2] += airfoils[k].cell_data['p']
        oa_cell[:, 3] += airfoils[k].cell_data['nut']
    oi_point = oi_point / len(internals)
    oi_cell = oi_cell / len(internals)
    oa_point = oa_point / len(airfoils)
    oa_cell = oa_cell / len(airfoils)
    internal_mean = internals[0].copy()
    internal_mean.point_data['U'][:, :2] = oi_point[:, :2]
    internal_mean.point_data['p'] = oi_point[:, 2]
    internal_mean.point_data['nut'] = oi_point[:, 3]
    internal_mean.cell_data['U'][:, :2] = oi_cell[:, :2]
    internal_mean.cell_data['p'] = oi_cell[:, 2]
    internal_mean.cell_data['nut'] = oi_cell[:, 3]

    airfoil_mean = airfoils[0].copy()
    airfoil_mean.point_data['U'][:, :2] = oa_point[:, :2]
    airfoil_mean.point_data['p'] = oa_point[:, 2]
    airfoil_mean.point_data['nut'] = oa_point[:, 3]
    airfoil_mean.cell_data['U'][:, :2] = oa_cell[:, :2]
    airfoil_mean.cell_data['p'] = oa_cell[:, 2]
    airfoil_mean.cell_data['nut'] = oa_cell[:, 3]

    return internal_mean, airfoil_mean


def Compute_coefficients(internals, airfoils, bool_surf, Uinf, angle, keep_vtk=False):
    # Compute force coefficients, if keet_vtk is True, also return the .vtu/.vtp with wall shear stress added over the airfoil and velocity gradient over the volume.

    coefs = []
    if keep_vtk:
        new_internals = []
        new_airfoils = []

    for internal, airfoil in zip(internals, airfoils):
        intern = internal.copy()
        aerofoil = airfoil.copy()

        point_mesh = intern.points[bool_surf, :2]
        point_surf = aerofoil.points[:, :2]

        intern = intern.compute_derivative(scalars='U', gradient='pred_grad')

        surf_grad = intern.point_data['pred_grad'].reshape(-1, 3, 3)[bool_surf, :2, :2]
        surf_p = intern.point_data['p'][bool_surf]

        surf_grad = reorganize(point_mesh, point_surf, surf_grad)
        surf_p = reorganize(point_mesh, point_surf, surf_p)

        Wss_pred = WallShearStress(surf_grad, -aerofoil.point_data['Normals'])
        aerofoil.point_data['wallShearStress'] = Wss_pred
        aerofoil.point_data['p'] = surf_p

        intern = intern.ptc(pass_point_data=True)
        aerofoil = aerofoil.ptc(pass_point_data=True)

        WP_int = -aerofoil.cell_data['p'][:, None] * aerofoil.cell_data['Normals'][:, :2]

        Wss_int = (aerofoil.cell_data['wallShearStress'] * aerofoil.cell_data['Length'].reshape(-1, 1)).sum(axis=0)
        WP_int = (WP_int * aerofoil.cell_data['Length'].reshape(-1, 1)).sum(axis=0)
        force = Wss_int - WP_int

        alpha = angle * np.pi / 180
        basis = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        force_rot = basis @ force
        coef = 2 * force_rot / Uinf ** 2
        coefs.append(coef)
        if keep_vtk:
            new_internals.append(intern)
            new_airfoils.append(aerofoil)

    if keep_vtk:
        return coefs, new_internals, new_airfoils
    else:
        return coefs


def Results_test(device, models, hparams, coef_norm, path_in, path_out, n_test=3, criterion='MSE',
                 x_bl=[.2, .4, .6, .8], s='full_test'):
    '''
    Compute criterion scores for the fields over the volume and the surface, and for the force coefficients. Also compute Spearman's correlation scores
    for the force coefficients and the relative error for the wall shear stress and the pressure over the airfoil. Outputs the true, the mean predicted
    and the std predicted integrated force coefficients, the true and mean predicted local force coefficients (at the surface of airfoils) and the true
    mean predicted boundary layers at chord x_bl.

    Args:
        device (str): Device on which you do the prediction.
        models (torch_geometric.nn.Module): List of models to predict with. It is a list of a list of different training of the same model.
            For example, it can be [model_MLP, model_GraphSAGE] where model_MLP is itself a list of the form [MLP_1, MLP_2].
        hparams (list): List of dictionnaries of hyperparameters of the models.
        coef_norm (tuple): Tuple of the form (mean_in, mean_out, std_in, std_out) for the denormalization of the data.
        path_in (str): Path to find the manifest.json file and the dataset.
        path_out (str): Path to write the scores.
        n_test (int, optional): Number of airfoils on which you want to infer (they will be drawn randomly in the given set). Default: ``3``
        criterion(str, optional): Criterion for the fields scores. Choose between MSE and MAE. Default: ``"MSE"``
        x_bl (list, optional): List of chord where the extract boundary layer prediction will be extracted. Default: ``[.2, .4, .6, .8]``
        s (str, optional): Dataset in which the simulation names are going to be sampled. Default: ``"full_test"``
    '''
    # Compute scores and all metrics for a
    sns.set()
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

    with open(osp.join(path_in, 'manifest.json'), 'r') as f:
        manifest = json.load(f)

    test_dataset = manifest[s]
    idx = random.sample(range(len(test_dataset)), k=n_test)
    idx.sort()

    test_dataset_vtk = Dataset(test_dataset, sample=None, coef_norm=coef_norm, my_path=path_in)
    test_loader = DataLoader(test_dataset_vtk, shuffle=False)

    if criterion == 'MSE':
        criterion = nn.MSELoss(reduction='none')
    elif criterion == 'MAE':
        criterion = nn.L1Loss(reduction='none')

    scores_vol = []
    scores_surf = []
    scores_force = []
    scores_p = []
    scores_wss = []
    internals = []
    airfoils = []
    true_internals = []
    true_airfoils = []
    times = []
    true_coefs = []
    pred_coefs = []
    for i in range(len(models[0])):
        model = [models[n][i] for n in range(len(models))]
        avg_loss_per_var = np.zeros((len(model), 4))
        avg_loss = np.zeros(len(model))
        avg_loss_surf_var = np.zeros((len(model), 4))
        avg_loss_vol_var = np.zeros((len(model), 4))
        avg_loss_surf = np.zeros(len(model))
        avg_loss_vol = np.zeros(len(model))
        avg_rel_err_force = np.zeros((len(model), 2))
        avg_loss_p = np.zeros((len(model)))
        avg_loss_wss = np.zeros((len(model), 2))
        internal = []
        airfoil = []
        pred_coef = []

        for j, data in enumerate(tqdm(test_loader)):
            Uinf, angle = float(test_dataset[j].split('_')[2]), float(test_dataset[j].split('_')[3])
            outs, tim = Infer_test(device, model, hparams, data, coef_norm=coef_norm)
            times.append(tim)
            intern = pv.read(osp.join(path_in, test_dataset[j], test_dataset[j] + '_internal.vtu'))
            aerofoil = pv.read(osp.join(path_in, test_dataset[j], test_dataset[j] + '_aerofoil.vtp'))
            tc, true_intern, true_airfoil = Compute_coefficients([intern], [aerofoil], data.surf, Uinf, angle,
                                                                 keep_vtk=True)
            tc, true_intern, true_airfoil = tc[0], true_intern[0], true_airfoil[0]
            intern, aerofoil = Airfoil_test(intern, aerofoil, outs, coef_norm, data.surf)
            pc, intern, aerofoil = Compute_coefficients(intern, aerofoil, data.surf, Uinf, angle, keep_vtk=True)
            if i == 0:
                true_coefs.append(tc)
            pred_coef.append(pc)

            if j in idx:
                internal.append(intern)
                airfoil.append(aerofoil)
                if i == 0:
                    true_internals.append(true_intern)
                    true_airfoils.append(true_airfoil)

            for n, out in enumerate(outs):
                loss_per_var = criterion(out, data.y).mean(dim=0)
                loss = loss_per_var.mean()
                loss_surf_var = criterion(out[data.surf, :], data.y[data.surf, :]).mean(dim=0)
                loss_vol_var = criterion(out[~data.surf, :], data.y[~data.surf, :]).mean(dim=0)
                loss_surf = loss_surf_var.mean()
                loss_vol = loss_vol_var.mean()

                avg_loss_per_var[n] += loss_per_var.cpu().numpy()
                avg_loss[n] += loss.cpu().numpy()
                avg_loss_surf_var[n] += loss_surf_var.cpu().numpy()
                avg_loss_vol_var[n] += loss_vol_var.cpu().numpy()
                avg_loss_surf[n] += loss_surf.cpu().numpy()
                avg_loss_vol[n] += loss_vol.cpu().numpy()
                avg_rel_err_force[n] += rel_err(tc, pc[n])
                avg_loss_wss[n] += rel_err(true_airfoil.point_data['wallShearStress'],
                                           aerofoil[n].point_data['wallShearStress']).mean(axis=0)
                avg_loss_p[n] += rel_err(true_airfoil.point_data['p'], aerofoil[n].point_data['p']).mean(axis=0)

        internals.append(internal)
        airfoils.append(airfoil)
        pred_coefs.append(pred_coef)

        score_var = np.array(avg_loss_per_var) / len(test_loader)
        score = np.array(avg_loss) / len(test_loader)
        score_surf_var = np.array(avg_loss_surf_var) / len(test_loader)
        score_vol_var = np.array(avg_loss_vol_var) / len(test_loader)
        score_surf = np.array(avg_loss_surf) / len(test_loader)
        score_vol = np.array(avg_loss_vol) / len(test_loader)
        score_force = np.array(avg_rel_err_force) / len(test_loader)
        score_p = np.array(avg_loss_p) / len(test_loader)
        score_wss = np.array(avg_loss_wss) / len(test_loader)

        score = score_surf + score_vol
        scores_vol.append(score_vol_var)
        scores_surf.append(score_surf_var)
        scores_force.append(score_force)
        scores_p.append(score_p)
        scores_wss.append(score_wss)

    scores_vol = np.array(scores_vol)
    scores_surf = np.array(scores_surf)
    scores_force = np.array(scores_force)
    scores_p = np.array(scores_p)
    scores_wss = np.array(scores_wss)
    times = np.array(times)
    true_coefs = np.array(true_coefs)
    pred_coefs = np.array(pred_coefs)
    pred_coefs_mean = pred_coefs.mean(axis=0)
    pred_coefs_std = pred_coefs.std(axis=0)

    spear_coefs = []
    for j in range(pred_coefs.shape[0]):
        spear_coef = []
        for k in range(pred_coefs.shape[2]):
            spear_drag = sc.stats.spearmanr(true_coefs[:, 0], pred_coefs[j, :, k, 0])[0]
            spear_lift = sc.stats.spearmanr(true_coefs[:, 1], pred_coefs[j, :, k, 1])[0]
            spear_coef.append([spear_drag, spear_lift])
        spear_coefs.append(spear_coef)
    spear_coefs = np.array(spear_coefs)

    with open(osp.join(path_out, 'score.json'), 'w') as f:
        json.dump(
            {
                'mean_time': times.mean(axis=0),
                'std_time': times.std(axis=0),
                'mean_score_vol': scores_vol.mean(axis=0),
                'std_score_vol': scores_vol.std(axis=0),
                'mean_score_surf': scores_surf.mean(axis=0),
                'std_score_surf': scores_surf.std(axis=0),
                'mean_rel_p': scores_p.mean(axis=0),
                'std_rel_p': scores_p.std(axis=0),
                'mean_rel_wss': scores_wss.mean(axis=0),
                'std_rel_wss': scores_wss.std(axis=0),
                'mean_score_force': scores_force.mean(axis=0),
                'std_score_force': scores_force.std(axis=0),
                'spearman_coef_mean': spear_coefs.mean(axis=0),
                'spearman_coef_std': spear_coefs.std(axis=0)
            }, f, indent=4, cls=NumpyEncoder
        )

    surf_coefs = []
    true_surf_coefs = []
    bls = []
    true_bls = []
    for i in range(len(internals[0])):
        aero_name = test_dataset[idx[i]]
        true_internal = true_internals[i]
        true_airfoil = true_airfoils[i]
        surf_coef = []
        bl = []
        for j in range(len(internals[0][0])):
            internal_mean, airfoil_mean = Airfoil_mean([internals[k][i][j] for k in range(len(internals))],
                                                       [airfoils[k][i][j] for k in range(len(airfoils))])
            internal_mean.save(osp.join(path_out, test_dataset[idx[i]] + '_' + str(j) + '.vtu'))
            surf_coef.append(np.array(metrics_NACA.surface_coefficients(airfoil_mean, aero_name)))
            b = []
            for x in x_bl:
                b.append(np.array(metrics_NACA.boundary_layer(airfoil_mean, internal_mean, aero_name, x)))
            bl.append(np.array(b))
        true_surf_coefs.append(np.array(metrics_NACA.surface_coefficients(true_airfoil, aero_name)))
        true_bl = []
        for x in x_bl:
            true_bl.append(np.array(metrics_NACA.boundary_layer(true_airfoil, true_internal, aero_name, x)))
        true_bls.append(np.array(true_bl))
        surf_coefs.append(np.array(surf_coef))
        bls.append(np.array(bl))

    true_bls = np.array(true_bls)
    bls = np.array(bls)

    return true_coefs, pred_coefs_mean, pred_coefs_std, true_surf_coefs, surf_coefs, true_bls, bls
