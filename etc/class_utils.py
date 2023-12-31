import tqdm
import flwr as fl
import numpy as np
import torch, torch.nn as nn
import timeit
import xgboost as xgb

from typing import Any, Dict, List, Optional, Tuple, Union, OrderedDict
from logging import DEBUG, INFO
from xgboost import XGBClassifier, XGBRegressor
from flwr.common import (
    NDArray, EvaluateIns, EvaluateRes, FitIns, FitRes,
    GetPropertiesIns, GetPropertiesRes, GetParametersIns, GetParametersRes,
    Status, Code, parameters_to_ndarrays, ndarrays_to_parameters, Parameters, Scalar
)
from torchmetrics import Accuracy, MeanSquaredError
from sklearn.metrics import mean_squared_error, accuracy_score
from torch.utils.data import DataLoader, Dataset, random_split
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.common.typing import Parameters
from flwr.server.server import (fit_clients,  evaluate_clients,)
from flwr.server.history import History
from flwr.server.strategy import FedXgbNnAvg, Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]

class TreeDataset(Dataset):
    def __init__(self, data: NDArray, labels: NDArray) -> None:
        self.labels = labels
        self.data = data

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[int, NDArray]:
        label = self.labels[idx]
        data = self.data[idx, :]
        sample = {0: data, 1: label}
        return sample

class CNN(nn.Module):
    def __init__(
        self, task_type : str = 'regression', client_num_tree : int = 100, 
        client_num : int = 2, n_channel: int = 64
    ) -> None:
        super(CNN, self).__init__()
        n_out = 1
        self.task_type = task_type
        self.conv1d = nn.Conv1d(1, n_channel, kernel_size=client_num_tree, 
                                stride=client_num_tree, padding=0)
        self.layer_direct = nn.Linear(n_channel * client_num, n_out)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.Identity = nn.Identity()

        # Add weight initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ReLU(self.conv1d(x))
        x = x.flatten(start_dim=1)
        x = self.ReLU(x)
        if self.task_type == "classification":
            x = self.Sigmoid(self.layer_direct(x))
        elif self.task_type == "regression":
            x = self.Identity(self.layer_direct(x))
        return x

    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        return [np.array(val.cpu().numpy(), copy=True) for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        layer_dict = {}
        for k, v in zip(self.state_dict().keys(), weights):
            if v.ndim != 0:
                layer_dict[k] = torch.Tensor(np.array(v, copy=True))
        state_dict = OrderedDict(layer_dict)
        self.load_state_dict(state_dict, strict=False)

def train(
    task_type: str, net: CNN, trainloader: DataLoader, device: torch.device,
    num_iterations: int, log_progress: bool = True,) -> Tuple[float, float, int]:
    # Define loss and optimizer
    if task_type == "classification": criterion = nn.BCELoss()
    elif task_type == "regression": criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))

    def cycle(iterable):
        """Repeats the contents of the train loader, in case it gets exhausted in 'num_iterations'."""
        while True:
            for x in iterable: yield x

    # Train the network
    net.train()
    total_loss, total_result, n_samples = 0.0, 0.0, 0
    pbar = (
        tqdm(iter(cycle(trainloader)), total=num_iterations, desc=f"TRAIN")
        if log_progress
        else iter(cycle(trainloader))
    )

    # Unusually, this training is formulated in terms of number of updates/iterations/batches processed
    # by the network. This will be helpful later on, when partitioning the data across clients: resulting
    # in differences between dataset sizes and hence inconsistent numbers of updates per 'epoch'.
    for i, data in zip(range(num_iterations), pbar):
        tree_outputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(tree_outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Collected training loss and accuracy statistics
        total_loss += loss.item()
        n_samples += labels.size(0)

        if task_type == "classification":
            acc = Accuracy(task="binary")(outputs, labels.type(torch.int))
            total_result += acc * labels.size(0)
        elif task_type == "regression":
            mse = MeanSquaredError()(outputs, labels.type(torch.int))
            total_result += mse * labels.size(0)

        if log_progress:
            if task_type == "classification":
                pbar.set_postfix({"train_loss": total_loss / n_samples, "train_acc": total_result / n_samples,})
            elif task_type == "regression":
                pbar.set_postfix({"train_loss": total_loss / n_samples, "train_mse": total_result / n_samples,})

    if log_progress: print("\n")
    return total_loss / n_samples, total_result / n_samples, n_samples


def test(
    task_type: str, net: CNN, testloader: DataLoader, device: torch.device, log_progress: bool = True,
) -> Tuple[float, float, int]:
    """Evaluates the network on test data."""
    if task_type == "classification": criterion = nn.BCELoss()
    elif task_type == "regression": criterion = nn.MSELoss()

    total_loss, total_result, n_samples = 0.0, 0.0, 0
    net.eval()
    with torch.no_grad():
        pbar = tqdm(testloader, desc="TEST") if log_progress else testloader
        for data in pbar:
            tree_outputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(tree_outputs)

            # Collected testing loss and accuracy statistics
            total_loss += criterion(outputs, labels).item()
            n_samples += labels.size(0)

            if task_type == "classification":
                acc = Accuracy(task="binary")(outputs.cpu(), labels.type(torch.int).cpu())
                total_result += acc * labels.size(0)
            elif task_type == "regression":
                mse = MeanSquaredError()(outputs.cpu(), labels.type(torch.int).cpu())
                total_result += mse * labels.size(0)

    if log_progress: print("\n")
    return total_loss / n_samples, total_result / n_samples, n_samples

class FL_Client(fl.client.Client):
    def __init__(self,task_type: str,trainloader: DataLoader,valloader: DataLoader,
                 client_tree_num: int,client_num: int,cid: str,log_progress: bool = False,):
        """ Creates a client for training `network.Net` on tabular dataset."""
        self.task_type = task_type
        print("TASK CLIENT: ", self.task_type)
        self.cid = cid
        self.tree = construct_tree_from_loader(trainloader, client_tree_num, task_type)
        self.trainloader_original = trainloader
        self.valloader_original = valloader
        self.trainloader, self.valloader = None, None
        self.client_tree_num = client_tree_num
        self.client_num = client_num
        self.properties = {"tensor_type": "numpy.ndarray"}
        self.log_progress = log_progress

        # instantiate model
        self.net = CNN(task_type = self.task_type, client_num_tree = self.client_tree_num,client_num = self.client_num)

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(properties=self.properties)

    def get_parameters(self, ins: GetParametersIns) -> Tuple[GetParametersRes, Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]]:
        return [GetParametersRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters(self.net.get_weights()),),(self.tree, int(self.cid)),]

    def set_parameters(
        self, parameters: Tuple[
            Parameters,
            Union[
                Tuple[XGBClassifier, int], Tuple[XGBRegressor, int],
                List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
            ],
        ],
    ) -> Union[
        Tuple[XGBClassifier, int], Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ]:
        self.net.set_weights(parameters_to_ndarrays(parameters[0]))
        return parameters[1]

    def fit(self, fit_params: FitIns) -> FitRes:
        # Process incoming request to train
        num_iterations, batch_size = fit_params.config["num_iterations"], fit_params.config["batch_size"]
        aggregated_trees = self.set_parameters(fit_params.parameters)

        if type(aggregated_trees) is list:
            print("Client " + self.cid + ": recieved", len(aggregated_trees), "trees")
        else:
            print("Client " + self.cid + ": only had its own tree")
        self.trainloader = tree_encoding_loader(
            self.trainloader_original, batch_size,aggregated_trees,self.client_tree_num,self.client_num,
        )
        self.valloader = tree_encoding_loader(
            self.valloader_original,batch_size,aggregated_trees,self.client_tree_num,self.client_num,
        )

        # num_iterations = None special behaviour: train(...) runs for a single epoch, however many updates it may be
        num_iterations = num_iterations or len(self.trainloader)

        # Train the model
        print(f"Client {self.cid}: training for {num_iterations} iterations/updates")
        self.net.to(self.device)
        train_loss, train_result, num_examples = train(
            self.task_type,self.net,self.trainloader,device=self.device,num_iterations=num_iterations,log_progress=self.log_progress,
        )
        print(f"Client {self.cid}: training round complete, {num_examples} examples processed")

        # Return training information: model, number of examples processed and metrics
        if self.task_type == "classification":
            return FitRes(
                status=Status(Code.OK, ""),parameters=self.get_parameters(fit_params.config),
                num_examples=num_examples,metrics={"loss": train_loss, "accuracy": train_result},
            )
        elif self.task_type == "regression":
            return FitRes(
                status=Status(Code.OK, ""),parameters=self.get_parameters(fit_params.config),
                num_examples=num_examples,metrics={"loss": train_loss, "mse": train_result},
            )

    def evaluate(self, eval_params: EvaluateIns) -> EvaluateRes:
        # Process incoming request to evaluate
        self.set_parameters(eval_params.parameters)

        # Evaluate the model
        self.net.to(self.device)
        loss, result, num_examples = test(
            self.task_type,self.net,self.valloader,device=self.device,log_progress=self.log_progress,
        )

        # Return evaluation information
        if self.task_type == "classification":
            print(f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, accuracy={result:.4f}")
            return EvaluateRes(
                status=Status(Code.OK, ""),loss=loss,num_examples=num_examples,metrics={"accuracy": result},
            )
        elif self.task_type == "regression":
            print(f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, mse={result:.4f}")
            return EvaluateRes(
                status=Status(Code.OK, ""),loss=loss,num_examples=num_examples,metrics={"mse": result},
            )

class FL_Server(fl.server.Server):
    """Flower server."""
    def __init__(self, *, client_manager: ClientManager, strategy: Optional[Strategy] = None) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(tensors=[], tensor_type="numpy.ndarray")
        self.strategy: Strategy = strategy
        self.max_workers: Optional[int] = None

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)

        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(INFO,"initial parameters (loss, other metrics): %s, %s",res[0],res[1],)
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime: self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(INFO,"fit progress: (%s, %s, %s, %s)", current_round,loss_cen,metrics_cen,timeit.default_timer() - start_time,)
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                    history.add_metrics_distributed(server_round=current_round, metrics=evaluate_metrics_fed)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate_round(
        self,server_round: int,timeout: Optional[float],) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,parameters=self.parameters,client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(DEBUG,"evaluate_round %s: strategy sampled %s clients (out of %s)",server_round,len(client_instructions),self._client_manager.num_available(),)

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(client_instructions,max_workers=self.max_workers,timeout=timeout,)
        log(DEBUG,"evaluate_round %s received %s results and %s failures",server_round,len(results),len(failures),)

        # Aggregate the evaluation results
        aggregated_result: Tuple[Optional[float],Dict[str, Scalar],] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,server_round: int,timeout: Optional[float],
    ) -> Optional[
        Tuple[
            Optional[
                Tuple[
                    Parameters,
                    Union[
                        Tuple[XGBClassifier, int], Tuple[XGBRegressor, int],
                        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
                    ],
                ]
            ],Dict[str, Scalar], FitResultsAndFailures,]]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,parameters=self.parameters,client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(DEBUG,"fit_round %s: strategy sampled %s clients (out of %s)",server_round,len(client_instructions),self._client_manager.num_available(),)

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,max_workers=self.max_workers,timeout=timeout,
        )

        log(DEBUG,"fit_round %s received %s results and %s failures",server_round,len(results),len(failures),)

        # Aggregate training results
        NN_aggregated: Parameters
        trees_aggregated: Union[
            Tuple[XGBClassifier, int], Tuple[XGBRegressor, int],
            List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
        ]
        metrics_aggregated: Dict[str, Scalar]
        aggregated, metrics_aggregated = self.strategy.aggregate_fit(server_round, results, failures)
        NN_aggregated, trees_aggregated = aggregated[0], aggregated[1]

        if type(trees_aggregated) is list:
            print("Server side aggregated", len(trees_aggregated), "trees.")
        else:
            print("Server side did not aggregate trees.")

        return ([NN_aggregated, trees_aggregated],metrics_aggregated,(results, failures),)

    def _get_initial_parameters(
        self, timeout: Optional[float]
    ) -> Tuple[Parameters, Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]]:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res_tree = random_client.get_parameters(ins=ins, timeout=timeout)
        parameters = [get_parameters_res_tree[0].parameters, get_parameters_res_tree[1]]
        log(INFO, "Received initial parameters from one random client")

        return parameters
    
def construct_tree_from_loader(
    dataset_loader: DataLoader, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset loader."""
    for dataset in dataset_loader:
        data, label = dataset[0], dataset[1]
    return construct_tree(data, label, n_estimators, tree_type)

def construct_tree(
    dataset: Dataset, label: NDArray, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset."""
    if tree_type == "classification":
        tree = xgb.XGBClassifier(
            objective="binary:logistic", learning_rate=0.1, max_depth=8,
            n_estimators=n_estimators,subsample=0.8,
            colsample_bylevel=1,colsample_bynode=1,colsample_bytree=1,
            alpha=5,gamma=5,num_parallel_tree=1,min_child_weight=1,
        )

    elif tree_type == "regression":
        tree = xgb.XGBRegressor(
            objective="reg:squarederror",learning_rate=0.1,max_depth=8,
            n_estimators=n_estimators,subsample=0.8,
            colsample_bylevel=1,colsample_bynode=1,colsample_bytree=1,
            alpha=5,gamma=5,num_parallel_tree=1,min_child_weight=1,)

    tree.fit(dataset, label)
    return tree

def central_xgboost(X_train, y_train, X_test, y_test, task, client_tree_num):
    global_tree = construct_tree(X_train, y_train, client_tree_num, task)
    preds_train = global_tree.predict(X_train)
    preds_test = global_tree.predict(X_test)

    if task == "classification":
        result_train = accuracy_score(y_train, preds_train)
        result_test = accuracy_score(y_test, preds_test)
        print("Global XGBoost Training Accuracy: %f" % (result_train))
        print("Global XGBoost Testing Accuracy: %f" % (result_test))
    elif task == "regression":
        result_train = mean_squared_error(y_train, preds_train)
        result_test = mean_squared_error(y_test, preds_test)
        print("Global XGBoost Training MSE: %f" % (result_train))
        print("Global XGBoost Testing MSE: %f" % (result_test))

    print(global_tree)

def simulate_client(trainset, testset, client_num, client_tree_num, task, X_test, y_test):
    client_trees_comparison = []
    trainloaders, _, testloader = do_fl_partitioning(
        trainset, testset, pool_size=client_num, batch_size="whole", val_ratio=0.0
    )

    for i, trainloader in enumerate(trainloaders):
        for local_dataset in trainloader:
            local_X_train, local_y_train = local_dataset[0], local_dataset[1]
            tree = construct_tree(local_X_train, local_y_train, client_tree_num, task)
            client_trees_comparison.append(tree)

            preds_train = client_trees_comparison[-1].predict(local_X_train)
            preds_test = client_trees_comparison[-1].predict(X_test)

            if task == "classification":
                result_train = accuracy_score(local_y_train, preds_train)
                result_test = accuracy_score(y_test, preds_test)
                print("Local Client %d XGBoost Training Accuracy: %f" % (i, result_train))
                print("Local Client %d XGBoost Testing Accuracy: %f" % (i, result_test))
            elif task == "regression":
                result_train = mean_squared_error(local_y_train, preds_train)
                result_test = mean_squared_error(y_test, preds_test)
                print("Local Client %d XGBoost Training MSE: %f" % (i, result_train))
                print("Local Client %d XGBoost Testing MSE: %f" % (i, result_test))

# https://github.com/adap/flower
def do_fl_partitioning(
    trainset: Dataset,testset: Dataset,pool_size: int,batch_size: Union[int, str],val_ratio: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // pool_size
    lengths = [partition_size] * pool_size
    if sum(lengths) != len(trainset):
        lengths[-1] = len(trainset) - sum(lengths[0:-1])
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(0))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = int(len(ds) * val_ratio)
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(0))
        trainloaders.append(get_dataloader(ds_train, "train", batch_size))
        if len_val != 0:
            valloaders.append(get_dataloader(ds_val, "val", batch_size))
        else:
            valloaders = None
    testloader = get_dataloader(testset, "test", batch_size)
    return trainloaders, valloaders, testloader

def get_dataloader(dataset: Dataset, partition: str, batch_size: Union[int, str]) -> DataLoader:
    if batch_size == "whole": batch_size = len(dataset)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train"))

def tree_encoding_loader(
    dataloader: DataLoader,
    batch_size: int,
    client_trees: Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],],
    client_tree_num: int, client_num: int, ) -> DataLoader:

    encoding = tree_encoding(dataloader, client_trees, client_tree_num, client_num)
    if encoding is None: return None
    data, labels = encoding
    tree_dataset = TreeDataset(data, labels)
    return get_dataloader(tree_dataset, "tree", batch_size)

def single_tree_prediction(tree: Union[XGBClassifier, XGBRegressor], n_tree: int, dataset: NDArray) -> Optional[NDArray]:
    """Extract the prediction result of a single tree in the xgboost tree ensemble."""
    # How to access a single tree
    # https://github.com/bmreiniger/datascience.stackexchange/blob/master/57905.ipynb
    num_t = len(tree.get_booster().get_dump())
    if n_tree > num_t:
        print("The tree index to be extracted is larger than the total number of trees.")
        return None

    return tree.predict(dataset, iteration_range=(n_tree, n_tree + 1), output_margin=True)

def tree_encoding(  # pylint: disable=R0914
    trainloader: DataLoader,
    client_trees: Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],],
    client_tree_num: int, client_num: int,) -> Optional[Tuple[NDArray, NDArray]]:

    """Transform the tabular dataset into prediction results using the
    aggregated xgboost tree ensembles from all clients."""
    if trainloader is None: return None

    for local_dataset in trainloader: x_train, y_train = local_dataset[0], local_dataset[1]

    x_train_enc = np.zeros((x_train.shape[0], client_num * client_tree_num))
    x_train_enc = np.array(x_train_enc, copy=True)

    temp_trees: Any = None
    if isinstance(client_trees, list) is False:
        temp_trees = [client_trees[0]] * client_num
    elif isinstance(client_trees, list) and len(client_trees) != client_num:
        temp_trees = [client_trees[0][0]] * client_num
    else:
        cids, temp_trees = [], []
        for i, _ in enumerate(client_trees):
            temp_trees.append(client_trees[i][0])  # type: ignore
            cids.append(client_trees[i][1])  # type: ignore
        sorted_index = np.argsort(np.asarray(cids))
        temp_trees = np.asarray(temp_trees)[sorted_index]

    for i, _ in enumerate(temp_trees):
        for j in range(client_tree_num):
            x_train_enc[:, i * client_tree_num + j] = single_tree_prediction(temp_trees[i], j, x_train)

    x_train_enc32: Any = np.float32(x_train_enc)
    y_train32: Any = np.float32(y_train)

    x_train_enc32, y_train32 = torch.from_numpy(np.expand_dims(x_train_enc32, axis=1)  # type: ignore
    ), torch.from_numpy(np.expand_dims(y_train32, axis=-1)  # type: ignore
)
    return x_train_enc32, y_train32

def serverside_eval(
    server_round: int,
    parameters: Tuple[Parameters,Union[
            Tuple[XGBClassifier, int],Tuple[XGBRegressor, int],
            List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],],],
    config: Dict[str, Scalar],task_type: str,testloader: DataLoader,batch_size: int,client_tree_num: int,client_num: int,
) -> Tuple[float, Dict[str, float]]:
    
    """An evaluation function for centralized/serverside evaluation over the entire test set."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN(client_num=client_num)

    model.set_weights(parameters_to_ndarrays(parameters[0]))
    model.to(device)

    trees_aggregated = parameters[1]
    testloader = tree_encoding_loader(testloader, batch_size, trees_aggregated, client_tree_num, client_num)
    loss, result, _ = test(task_type, model, testloader, device=device, log_progress=False)

    if task_type == "classification":
        print(f"Evaluation on the server: test_loss={loss:.4f}, test_accuracy={result:.4f}")
        return loss, {"accuracy": result}
    elif task_type == "regression":
        print(f"Evaluation on the server: test_loss={loss:.4f}, test_mse={result:.4f}")
        return loss, {"mse": result}