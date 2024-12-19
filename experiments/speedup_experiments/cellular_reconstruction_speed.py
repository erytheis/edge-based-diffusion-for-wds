import datetime
import os
import time
from os.path import join

import numpy as np
import pandas as pd
import torch
# from numpy.linalg import weigh
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm

from src.model.torch_models.model.diffusion_step import DiffusionStep
from src.model.torch_models.parse_config import ConfigParser
from src.model.torch_models.runners import load_cli_options, load_args, load_datasets
from src.model.torch_models.loader import *

from src.utils.torch_utils import prepare_device
from src.utils.utils import DEFAULT_COLORS, PROJECT_ROOT

from definitions import ROOT_DIR

def fetch_best(path_to_logs, num=0):
    logs = pd.read_json(path_to_logs, lines=True)
    best = logs.sort_values('target', ascending=False).iloc[num]
    params = best['params']
    return params, best['target']



def _plot_headloss_and_flowarates(mask):
    # plot headloss
    x = np.arange(len(h_real[mask]))
    # x = np.arange(num_elements - num_real_nodes + reservoirs.sum().item())
    sortargs = np.argsort(h_real[mask].cpu().squeeze())
    plt.plot(x, h_real.cpu()[mask].cpu()[sortargs], color='b')
    plt.bar(x, h[mask].cpu()[sortargs], color='r')
    plt.show()
    #
    # plot flowrates
    sortargs = np.argsort(y[mask].cpu())

    plt.bar(x, y_mask[mask].cpu()[sortargs], color='r')
    plt.plot(x, batch.edge_y.cpu()[mask].cpu()[sortargs], color='b')
    plt.show()


def runtime_fp(y_mask, mu_down, mu_up,
               momentum_down=0.0, momentum_up=0,
               low_iterations=3, up_iterations=1):
    total_checking_time = 0

    delta_y_mask = torch.zeros_like(y_mask)
    delta_h = torch.zeros_like(y_mask)

    # scatter virtual edges to sparse indices
    mask_ = virtual[batch.upper_laplacian_index[0]].squeeze()

    index = batch.upper_laplacian_index[:, ~mask_]
    weight = batch.upper_laplacian_weight[~mask_].clone() * mu_up

    index_down = batch.lower_laplacian_index
    weight_down = batch.lower_laplacian_weight.clone() * mu_down

    for j in tqdm(range(stop_points[-1])):

        for i in range(low_iterations):
            y_mask_old = y_mask
            y_mask = y_mask - fp(y_mask, index_down, weight_down) + momentum_down * delta_y_mask
            y_mask = torch.where(virtual, y, y_mask)
            delta_y_mask = y_mask - y_mask_old

        # transform
        h = loss_coefficient * y_mask.abs() ** 1.852 * y_mask.sign()

        # propagate with mp
        for i in range(up_iterations):

            if known_nodes.sum() > len(batch):
                h = torch.where(reservoir_connector, h_known, h)

            h_old = h
            h = h - fp(h, index, weight) + momentum_up * delta_h
            delta_h = h - h_old

        # transform back
        y_new = (h.abs() / loss_coefficient) ** (1 / 1.852) * h.sign()
        y_mask = torch.where(~reservoir_connector, y_new, y_mask)

        # log
        if j in stop_points:
            checking_time_st = time.time()

            y_mask = torch.where(virtual, y, y_mask)

            f_score = r2_score(y[mask].cpu(), y_mask[mask].cpu())
            h_score = r2_score(h_real[mask].cpu(), h[mask].cpu())

            print(f_score)
            # check if y is diverging
            if delta_y_mask.abs().max() > 1e2:
                print('Flowrate is Diverging')
                break

            # check if h is diverging
            if delta_h.abs().max() > 1e1:
                print('H is Diverging')
                break

            # stop conditions
            if h_score > 0.99 and 'headloss_99' not in total_time[ds.name][memory_limit]:
                total_time[ds.name][memory_limit]['headloss_99'] = time.time() - st_time

            if f_score > 0.99 and 'flowrate_99' not in total_time[ds.name][memory_limit]:
                total_time[ds.name][memory_limit]['flowrate_99'] = time.time() - st_time

            if h_score > 0.999 and 'headloss_999' not in total_time[ds.name][memory_limit]:
                total_time[ds.name][memory_limit]['headloss_999'] = time.time() - st_time

            if f_score > 0.999 and 'flowrate_999' not in total_time[ds.name][memory_limit]:
                total_time[ds.name][memory_limit]['flowrate_999'] = time.time() - st_time

            # finally
            if f_score > 0.999:
                break

            total_checking_time += time.time() - checking_time_st

    return h, y_mask, total_checking_time, j


if __name__ == '__main__':
    reservoirs = torch.tensor(0)
    config_name = join(PROJECT_ROOT, 'input', 'config_simplex.yaml')

    args = load_args(config_name,
                     )
    options = load_cli_options()
    config = ConfigParser.from_args(args, options)
    device, device_ids = prepare_device(config['n_gpu'], config['device'])

    dataset = load_datasets(config, device=device)

    normalized = False
    # reload = 'out/cellular_reconstruction_speed_optimal_experiment.csv'
    down_iterations = 3
    get_best = True

    # memory_limits = list(np.logspace(-1, 1, 10)*2)
    memory_limits = [0.1]
    suffix = ''

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(dataset)

    stop_points = np.linspace(0, 100, 25).astype(int)
    stop_points = np.concatenate([stop_points, np.linspace(100, 1000, 100).astype(int)])

    if True:
        fp = DiffusionStep().eval()
    # torch jittable
    elif hasattr(torch_geometric, 'compile'):
        fp = torch_geometric.compile(fp)

    torch.no_grad()

    try:
        sim_stats = pd.read_csv('out/cellular_reconstruction_speed_optimal_experiment.csv')
    except FileNotFoundError:
        sim_stats = pd.DataFrame(columns=['dataset',
                                          'num_simulations',
                                          'num_nodes',
                                          'num_pipes',
                                          'total_iterations',
                                          'gpu_taken',
                                          'total_time',
                                          'flowrate_99_time',
                                          'headloss_99_time',
                                          'flowrate_999_time',
                                          'headloss_999_time',
                                          'evaluattion_time',
                                          'up_iterations'])
    # compare it with mp scheme       ] 'total_time',

    total_time = {}

    for ds_i, ds in enumerate(dataset.datasets):

        total_time[ds.name] = {}
        old_num_simulations, num_simulations = 0, 0

        for i, memory_limit in enumerate(memory_limits):

            torch.cuda.empty_cache()

            total_time[ds.name][memory_limit] = {}

            # gpu loader
            if ds.name in ['kl', 'marchirural','net-3']:
                step = 5 * memory_limit / 2
            elif ds.name in ['bak', 'asnet2',  'zj','l-town']:
                step = 100 * memory_limit / 2
            elif ds.name in ['jilin', 'apulia', 'pes']:
                step = 50 * memory_limit / 2
            else:
                step = 10* memory_limit / 2

            data_loader = FillGPULoader(ds, memory_limit=memory_limit, step=step, device=device)
            num_pipes = ds[0].num_edges

            for i, batch in enumerate(data_loader):

                ds_name = ds.name

                num_nodes, num_edges = batch.num_nodes, batch.num_edges
                num_elements = batch.num_edges
                num_real_nodes = batch.num_nodes // 2
                num_real_edges = (batch.num_edges - num_real_nodes) / len(batch)
                num_real_nodes = int(num_real_nodes / len(batch))


                virtual = (batch.edge_attr[:, -2] == 1)
                virtual_nodes = batch.x[:, 2].bool()
                reservoir_connector = (batch.edge_attr[:, -1] == 1)
                real_edges = ~virtual & ~reservoir_connector

                reservoir_connector = reservoir_connector.unsqueeze(-1)
                virtual = virtual.unsqueeze(-1)

                # get known heads
                known_nodes = batch.x[:, 3] == 1
                virtual_reservoir_pipes = torch.zeros_like(batch.edge_index[0], dtype=torch.bool)
                virtual_reservoir_nodes = torch.zeros_like(batch.x[:, 3], dtype=torch.bool)

                edge_attr_slices = batch.slices['edge_attr'].numpy()
                x_slices = batch.slices['x'].int().numpy()

                for i in range(len(edge_attr_slices) - 1):
                    edge_attr_slice = slice(edge_attr_slices[i], edge_attr_slices[i + 1])
                    x_slice = slice(x_slices[i], x_slices[i + 1])

                    virtual_reservoir_nodes[(x_slices[i] + num_real_nodes):x_slices[i + 1]] = \
                        (batch.x[:, 3] == 1)[x_slices[i]:(x_slices[i + 1] - num_real_nodes)]

                virtual_reservoir_pipes = virtual_reservoir_nodes[batch.edge_index[0]] | \
                                          virtual_reservoir_nodes[batch.edge_index[1]]

                # get best data
                if get_best:
                    try:
                        path_to_logs = join(ROOT_DIR,
                                            'input', 'optimal_parameters', ds_name, 'logs_experiment.log.json')
                        param_dict, target = fetch_best(path_to_logs)

                        print(f'Successfully loaded best config, in {target} iterations')
                        print(param_dict)

                        momentum_down = param_dict['momentum_down']
                        momentum_up = param_dict['momentum_up']
                        mu_down = param_dict['mu_down']
                        mu_up = param_dict['mu_up']

                    except FileNotFoundError:
                        print('No best config found, using default values')

                # loss coefficients
                h_real = batch.y[batch.edge_index[1]] - batch.y[batch.edge_index[0]]
                h_real = h_real.unsqueeze(-1)

                # account for multiple reservoirs
                if known_nodes.sum() > len(batch):
                    # release the end of the network
                    # sparse for FP
                    L1_down_sparse_indices, L1_down_sparse_values = batch.lower_laplacian_index, batch.lower_laplacian_weight

                    virtual_reservoir_pipes_sparse_indices = virtual_reservoir_pipes[L1_down_sparse_indices[0]] & \
                                                             virtual_reservoir_pipes[
                                                                 L1_down_sparse_indices[1]]

                    virtual[virtual_reservoir_pipes] = 0

                    L1_down_sparse_values[virtual_reservoir_pipes_sparse_indices] = 1
                    batch.lower_laplacian_index = L1_down_sparse_indices
                    batch.lower_laplacian_weight = L1_down_sparse_values

                    h_known = reservoir_connector * h_real

                # create folders if dont exist
                if not os.path.exists(f'out/{ds.name}'):
                    os.makedirs(f'out/{ds.name}')

                y = batch.edge_y.unsqueeze(-1)

                # remove reservoir from mask
                mask = virtual.squeeze()

                loss_coefficient = batch.edge_attr[:, 1].unsqueeze(-1)
                loss_coefficient = torch.clip(loss_coefficient, 10e-6)

                y_mask = y.clone()
                y_mask[~virtual] = 0

                # total reservoir inflow
                # outflow = y_mask[(virtual & ~virtual_reservoir_pipes.unsqueeze(-1))].abs().sum()
                # y_mask[virtual_reservoir_pipes] = outflow / known_nodes.sum()

                mask = real_edges

                # tets with bathces
                st_time = time.time()
                total_checking_time = 0

                h, y_mask, total_checking_time, total_iterations = runtime_fp(y_mask, mu_down=mu_down, mu_up=mu_up,
                                                                              momentum_down=momentum_down,
                                                                              momentum_up=momentum_up,
                                                                              low_iterations=3)
                total = time.time() - st_time
                sim_stats = pd.concat([sim_stats, pd.DataFrame({'dataset': ds_name,
                                                                'num_simulations': len(batch.wds_names),
                                                                'num_nodes': num_real_nodes,
                                                                'num_pipes': num_real_edges,
                                                                'total_iterations': total_iterations,
                                                                'gpu_taken': memory_limit,
                                                                'total_time': total,
                                                                'flowrate_99_time': total_time[ds.name][
                                                                    memory_limit].get('flowrate_99', None),
                                                                'headloss_99_time': total_time[ds.name][
                                                                    memory_limit].get('headloss_99', None),
                                                                'flowrate_999_time': total_time[ds.name][
                                                                    memory_limit].get('flowrate_999', None),
                                                                'headloss_999_time': total_time[ds.name][
                                                                    memory_limit].get(
                                                                    'headloss_999', None),
                                                                'evaluation_time': total_checking_time}, index=[ds_i])])

                print('Time: ', time.time() - st_time)

                # compare it with mp scheme
                print('-----')

                # check if we need to stop because of memory limit
                old_num_simulations = num_simulations
                num_simulations = len(batch.wds_names)

                # releas memory
                # del sample, y, y_mask, h, total_checking_time, total_iterations, known_nodes, virtual_reservoir_nodes, \
                #     virtual_reservoir_pipes, \
                #     h_real, mask, loss_coefficient, virtual, real_edges, reservoir_connector, virtual_nodes

                torch.cuda.empty_cache()

            if old_num_simulations >= num_simulations:
                break

        # save stats
        sim_stats.to_csv(f'out/cellular_reconstruction_speed_optimal_experiment{suffix}.csv', index=False)
DEFAULT_COLORS
