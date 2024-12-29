import logging
import time
from os.path import join

import numpy as np
import torch

from src.model.torch_models.loader import BaseGNNDataLoader
from src.model.torch_models.model.convergence.convergence_checker import (
    StopPointsConvergenceChecker,
)
from src.model.torch_models.model.diffusion import Diffusion
from src.model.torch_models.model.metric import r2_score
from src.model.torch_models.parse_config import ConfigParser
from src.model.torch_models.runners import load_args, load_cli_options, load_datasets
from src.utils.torch.torch_utils import prepare_device
from src.utils.utils import PROJECT_ROOT
from src.utils.wds_utils import fetch_best

if __name__ == "__main__":

    # Configure logging
    logger = logging.getLogger("my_convergence_logger")
    logger.setLevel(logging.DEBUG)
    # Add a console handler with a simple format
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    reservoirs = torch.tensor(0)
    config_name = join(PROJECT_ROOT, "input", "config_simplex.yaml")

    args = load_args(
        config_name,
    )
    options = load_cli_options()
    config = ConfigParser.from_args(args, options)
    device, device_ids = prepare_device(config["n_gpu"], config["device"])

    dataset = load_datasets(config, device=device)
    data_loader = BaseGNNDataLoader(dataset, 10000, device=device)

    normalized = False
    # reload = 'out/cellular_reconstruction_speed_optimal_experiment.csv'
    down_iterations = 3
    get_best = True

    stop_points = np.linspace(100, 1000, 100).astype(int)
    convergence_checker = StopPointsConvergenceChecker(stop_points)

    fp = Diffusion().eval()
    # torch jittable
    # elif hasattr(torch_geometric, "compile"):
    #     fp = torch_geometric.compile(fp)

    torch.no_grad()

    for batch in data_loader:
        path_to_logs = join(
            PROJECT_ROOT,
            "input",
            "optimal_parameters",
            batch.wds_names[0],
            "logs_experiment.log.json",
        )

        param_dict, target = fetch_best(path_to_logs)

        print(f"Successfully loaded best config, in {target} iterations")
        print(param_dict)

        momentum_down = param_dict["momentum_down"]
        momentum_up = param_dict["momentum_up"]
        tau_down = param_dict["mu_down"]
        tau_up = param_dict["mu_up"]

        st_iteration = time.time()
        f, h = fp(batch, tau_down, tau_up, momentum_down, momentum_up)
        end_iteration = time.time()

        # comparison with the true values (if exist)
        if batch.edge_y is not None:
            h_true = (
                batch.edge_y.abs() ** 1.852 * batch.edge_y.sign() * batch.edge_attr[:, 1]
            )

            print("f:", r2_score(f, batch.edge_y.unsqueeze(-1)))
            print("h:", r2_score(h, h_true.unsqueeze(-1)))

