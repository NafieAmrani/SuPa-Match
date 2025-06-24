import igl
import pymeshfix

import numpy as np
from tqdm.auto import tqdm
import fine2coarsedec as f2cd
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn.functional as F
import torch.optim as optim

from networks.diffusion_network import DiffusionNet
from networks.permutation_network import Similarity
from networks.fmap_network import RegularizedFMNet

from losses.fmap_loss import SURFMNetLoss, PartialFmapsLoss, SquaredFrobeniusLoss
from losses.dirichlet_loss import DirichletLoss

from utils.geometry_util import compute_operators
from utils.tensor_util import to_numpy
from utils.misc import robust_lossfun
from utils.shape_util import read_shape


def to_tensor(vert_np, face_np, device):
    vert = torch.from_numpy(vert_np).to(device=device, dtype=torch.float32)
    face = torch.from_numpy(face_np).to(device=device, dtype=torch.long)

    return vert, face

def compute_features(vert_x, face_x, vert_y, face_y, feature_extractor, normalize=True):
    feat_x = feature_extractor(vert_x.unsqueeze(0), face_x.unsqueeze(0))
    feat_y = feature_extractor(vert_y.unsqueeze(0), face_y.unsqueeze(0))
    # normalize features
    if normalize:
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)

    return feat_x, feat_y

def compute_permutation_matrix(feat_x, feat_y, permutation, bidirectional=False, normalize=True):
    # normalize features
    if normalize:
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
    similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

    Pxy = permutation(similarity)

    if bidirectional:
        Pyx = permutation(similarity.transpose(1, 2))
        return Pxy, Pyx
    else:
        return Pxy

def update_network(loss_metrics, feature_extractor, optimizer):
    # compute total loss
    loss = 0.0
    for k, v in loss_metrics.items():
        if k != 'l_total':
            loss += v
    optimizer.zero_grad()
    # backward pass
    loss.backward()
    # clip gradient for stability
    torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), 1.0)
    # update weight
    optimizer.step()

    return loss

def get_feature_extractor(network_path, input_type='wks', num_refine=0, partial=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_channels = 128 if input_type == 'wks' else 3  # 'xyz'
    feature_extractor = DiffusionNet(in_channels=in_channels, out_channels=256, input_type=input_type).to(device)
    feature_extractor.load_state_dict(torch.load(network_path, map_location=torch.device(device))['networks']['feature_extractor'], strict=True)
    return feature_extractor

def get_opts_for_faust():
    network_path = "checkpoints/faust.pth"
    feature_opts = {
        "feature_extractor": get_feature_extractor(network_path, input_type='wks'),
        "num_refine": 0,
        "partial": False,
        "non_isometric": False,
    }
    return feature_opts

def get_opts_for_dt4d_inter():
    network_path = "checkpoints/dt4d.pth"
    feature_opts = {
        "feature_extractor": get_feature_extractor(network_path, input_type='wks'),
        "num_refine": 0,
        "partial": False,
        "non_isometric": True,
    }
    return feature_opts

def get_opts_for_dt4d_intra():
    network_path = "checkpoints/dt4d.pth"
    feature_opts = {
        "feature_extractor": get_feature_extractor(network_path, input_type='wks'),
        "num_refine": 25,
        "partial": False,
        "non_isometric": False,
    }
    return feature_opts

def get_opts_for_smal():
    network_path = "checkpoints/smal.pth"
    feature_opts = {
        "feature_extractor": get_feature_extractor(network_path, input_type='xyz'),
        "num_refine": 25,
        "partial": False,
        "non_isometric": True,
    }
    return feature_opts

def get_opts_for_mano():
    network_path = "checkpoints/mano.pth"
    feature_opts = {
        "feature_extractor": get_feature_extractor(network_path, input_type='wks'),
        "num_refine": 0,
        "partial": False,
        "non_isometric": False,
    }
    return feature_opts

def get_feature_opts(dataset):
    if 'faust' in dataset.lower():
        return get_opts_for_faust()
    if 'smal' in dataset.lower():
        return get_opts_for_smal()
    if 'dt4d_inter' in dataset.lower():
        return get_opts_for_dt4d_inter()
    if 'dt4d_intra' in dataset.lower():
        return get_opts_for_dt4d_intra()
    if 'mano' in dataset.lower():
        return get_opts_for_mano()

def knn_search(x, X, k=1, return_distance=False):
    """
    find indices of k-nearest neighbors of x in X
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    if return_distance:
        d, indices = nbrs.kneighbors(x)
        if k == 1:
            return indices.flatten(), d.flatten()
        else:
            return indices, d
    else:
        _, indices = nbrs.kneighbors(x)
        if k == 1:
            return indices.flatten()
        else:
            return indices

def shape_loader(filename1, filename2, shape_loader_opts):
    """
    load and downsample shapes
    """
    vert_np_x, face_np_x = read_shape(filename1)
    vert_np_y, face_np_y = read_shape(filename2)
    vert_np_x -= np.mean(vert_np_x, axis=0)
    vert_np_y -= np.mean(vert_np_y, axis=0)

    VX_orig = vert_np_x
    FX_orig = face_np_x
    VY_orig = vert_np_y
    FY_orig = face_np_y
    VX, FX = VX_orig, FX_orig
    VY, FY = VY_orig, FY_orig


    downsample = shape_loader_opts['num_faces'] != -1
    if downsample:
        shape_loader_opts['num_faces_x'] = shape_loader_opts.get('num_faces', shape_loader_opts['num_faces'])
        nfacesX = min(shape_loader_opts['num_faces_x'], len(FX_orig))
        nfacesY = min(shape_loader_opts['num_faces'], len(FY_orig))

        use_qslim = False
        VX, FX = f2cd.decimate_no_map(VX, FX, nfacesX, use_qslim)
        VY, FY = f2cd.decimate_no_map(VY, FY, nfacesY, use_qslim)

        # fix triangulation
        n_faces = FX.shape[0]
        VX, FX = igl.upsample(VX, FX, 2)
        VX, FX = f2cd.decimate_no_map(VX, FX, n_faces, False)

        VX, FX = pymeshfix.clean_from_arrays(VX, FX)
        VY, FY = pymeshfix.clean_from_arrays(VY, FY)

        idx_vx_in_orig = knn_search(VX, VX_orig)
        idx_vy_in_orig = knn_search(VY, VY_orig)

        return VX_orig, FX_orig, VX, FX, idx_vx_in_orig, VY_orig, FY_orig, VY, FY, idx_vy_in_orig
    else:
        idx_vx_in_orig = knn_search(VX, VX_orig)
        idx_vy_in_orig = knn_search(VY, VY_orig)
        return (VX_orig, FX_orig, VX, FX, idx_vx_in_orig,
                VY_orig, FY_orig, VY, FY, idx_vy_in_orig)

def get_features(VX, FX, VY, FY, feature_opts, return_numpy=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device used: ", device)
    if isinstance(VX, np.ndarray):
        vert_x, face_x = to_tensor(VX, FX, device)
        vert_y, face_y = to_tensor(VY, FY, device)
    else:
        vert_x, face_x = VX, FX
        vert_y, face_y = VY, FY

    # compute Laplacian
    _, mass_x, Lx, evals_x, evecs_x, _, _ = compute_operators(vert_x, face_x, k=200)
    _, mass_y, Ly, evals_y, evecs_y, _, _ = compute_operators(vert_y, face_y, k=200)
    evecs_trans_x = evecs_x.T * mass_x[None]
    evecs_trans_y = evecs_y.T * mass_y[None]
    feature_extractor = feature_opts['feature_extractor']
    feature_extractor.eval()
    num_refine = feature_opts['num_refine']
    partial = feature_opts['partial']
    non_isometric = feature_opts['non_isometric']
    if num_refine > 0:
        with torch.set_grad_enabled(True):
            permutation = Similarity(tau=0.07, hard=False).to(device)
            fmap_net = RegularizedFMNet(bidirectional=True)
            optimizer = optim.Adam(feature_extractor.parameters(), lr=1e-3)
            fmap_loss = SURFMNetLoss(w_bij=1.0, w_orth=1.0, w_lap=0.0) if not partial else PartialFmapsLoss(w_bij=1.0, w_orth=1.0)
            align_loss = SquaredFrobeniusLoss(loss_weight=1.0)
            if non_isometric:
                w_dirichlet = 5.0
            else:
                if partial:
                    w_dirichlet = 1.0
                else:
                    w_dirichlet = 0.0
            dirichlet_loss = DirichletLoss(loss_weight=w_dirichlet)
            print('Test-time adaptation')
            pbar = tqdm(range(num_refine))
            for _ in pbar:
                feat_x, feat_y = compute_features(vert_x, face_x, vert_y, face_y, feature_extractor, normalize=False)
                Cxy, Cyx = fmap_net(feat_x, feat_y, evals_x.unsqueeze(0), evals_y.unsqueeze(0),
                                    evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0))
                Pxy, Pyx = compute_permutation_matrix(feat_x, feat_y, permutation, bidirectional=True, normalize=True)

                # compute functional map regularisation loss
                loss_metrics = fmap_loss(Cxy, Cyx, evals_x.unsqueeze(0), evals_y.unsqueeze(0))
                # compute C
                Cxy_est = torch.bmm(evecs_trans_y.unsqueeze(0), torch.bmm(Pyx, evecs_x.unsqueeze(0)))

                # compute couple loss
                loss_metrics['l_align'] = align_loss(Cxy, Cxy_est)
                if not partial:
                    Cyx_est = torch.bmm(evecs_trans_x.unsqueeze(0), torch.bmm(Pxy, evecs_y.unsqueeze(0)))
                    loss_metrics['l_align'] += align_loss(Cyx, Cyx_est)

                # compute dirichlet energy
                if non_isometric:
                    loss_metrics['l_d'] = (dirichlet_loss(torch.bmm(Pxy, vert_y.unsqueeze(0)), Lx.to_dense().unsqueeze(0)) +
                                           dirichlet_loss(torch.bmm(Pyx, vert_x.unsqueeze(0)), Ly.to_dense().unsqueeze(0)))

                loss = update_network(loss_metrics, feature_extractor, optimizer)
                pbar.set_description(f'Total loss: {loss:.4f}')

    feature_extractor.eval()
    with torch.no_grad():
        feat_x, feat_y = compute_features(vert_x, face_x, vert_y, face_y, feature_extractor)

    if return_numpy:
        feat_x = to_numpy(feat_x)
        feat_y = to_numpy(feat_y)
    else:
        feat_x = feat_x.squeeze(0)
        feat_y = feat_y.squeeze(0)

    return feat_x, feat_y


def get_edge_cost(feat_x, feat_y, vx2VX, vy2VY):
    edge_costs = torch.zeros((len(vx2VX), len(vy2VY)))
    for i in range(0, len(vx2VX)):
        diff = feat_y[vy2VY, :] - feat_x[vx2VX[i], :]
        edge_costs[i, :]  = torch.sum(robust_lossfun(diff.to(torch.float64),
                              alpha=torch.tensor(2, dtype=torch.float64),
                              scale=torch.tensor(0.3, dtype=torch.float64)), dim=1)
    edge_costs = to_numpy(edge_costs)
    return edge_costs