import flwr as fl
import torch, torch.nn as nn
import xgboost as xgb
import functools

from typing import Union, Dict
from xgboost import XGBClassifier, XGBRegressor
from matplotlib import pyplot as plt
from class_utils import do_fl_partitioning, serverside_eval, FL_Client, FL_Server
from flwr.common import Scalar
from flwr.server.app import ServerConfig
from torch.utils.data import DataLoader, Dataset, random_split
from flwr.server.history import History
from flwr.server.strategy import FedXgbNnAvg, Strategy
from flwr.server.client_manager import ClientManager, SimpleClientManager

def plot_xgbtree(tree: Union[XGBClassifier, XGBRegressor], n_tree: int) -> None:
    """Visualize the built xgboost tree."""
    xgb.plot_tree(tree, num_trees=n_tree)
    plt.rcParams["figure.figsize"] = [50, 10]
    plt.show()

def print_model_layers(model: nn.Module) -> None:
    print(model)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def start_experiment(
    task_type: str, trainset: Dataset,testset: Dataset,num_rounds: int = 5, 
    client_tree_num: int = 50, client_pool_size: int = 5, num_iterations: int = 100,
    fraction_fit: float = 1.0, min_fit_clients: int = 2,batch_size: int = 32,val_ratio: float = 0.1,
) -> History:
    
    client_resources = {"num_cpus": 0.5}  # 2 clients per CPU

    # Partition the dataset into subsets reserved for each client.
    # - 'val_ratio' controls the proportion of the (local) client reserved as a local test set
    # (good for testing how the final model performs on the client's local unseen data)
    trainloaders, valloaders, testloader = do_fl_partitioning(
        trainset, testset, batch_size="whole",
        pool_size=client_pool_size,val_ratio=val_ratio,
    )
    print(f"Data partitioned across {client_pool_size} clients"f" and {val_ratio} of local dataset reserved for validation.")

    # Configure the strategy
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        print(f"Configuring round {server_round}")
        return {"num_iterations": num_iterations,"batch_size": batch_size,}

    # FedXgbNnAvg
    strategy = FedXgbNnAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit if val_ratio > 0.0 else 0.0,
        min_fit_clients=min_fit_clients,min_evaluate_clients=min_fit_clients,
        min_available_clients=client_pool_size, on_fit_config_fn=fit_config,
        on_evaluate_config_fn=(lambda r: {"batch_size": batch_size}),
        evaluate_fn=functools.partial(
            serverside_eval,task_type=task_type,testloader=testloader,
            batch_size=batch_size,client_tree_num=client_tree_num,client_num=client_pool_size,
        ),
        accept_failures=False,
    )

    print(f"FL experiment configured for {num_rounds} rounds with {client_pool_size} client in the pool.")
    print(f"FL round will proceed with {fraction_fit * 100}% of clients sampled, at least {min_fit_clients}.")

    def client_fn(cid: str) -> fl.client.Client:
        """Creates a federated learning client"""
        if val_ratio > 0.0 and val_ratio <= 1.0:
            return FL_Client(
                task_type,trainloaders[int(cid)],valloaders[int(cid)],
                client_tree_num,client_pool_size,cid,log_progress=False,
            )
        else:
            return FL_Client(
                task_type,trainloaders[int(cid)],None,client_tree_num,client_pool_size,
                cid,log_progress=False,
            )

    # Start the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        server=FL_Server(client_manager=SimpleClientManager(), strategy=strategy),
        num_clients=client_pool_size,client_resources=client_resources,
        config=ServerConfig(num_rounds=num_rounds),strategy=strategy,
    )

    print(history)
    return history
