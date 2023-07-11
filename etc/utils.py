import os
import urllib.request
import bz2
import torch
import numpy as np
import shutil
import xgboost as xgb

from typing import Any, Dict, List, Optional, Tuple, Union
from xgboost import XGBClassifier, XGBRegressor
from matplotlib import pyplot as plt
from flwr.common import NDArray, NDArrays
from torchmetrics import Accuracy, MeanSquaredError
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_svmlight_file
from torch.utils.data import DataLoader, Dataset, random_split

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
    def __init__(self, n_channel: int = 64) -> None:
        super(CNN, self).__init__()
        n_out = 1
        self.task_type = task_type
        self.conv1d = nn.Conv1d(
            1, n_channel, kernel_size=client_tree_num, stride=client_tree_num, padding=0
        )
        self.layer_direct = nn.Linear(n_channel * client_num, n_out)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.Identity = nn.Identity()

        # Add weight initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

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
        return [
            np.array(val.cpu().numpy(), copy=True)
            for _, val in self.state_dict().items()
        ]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        layer_dict = {}
        for k, v in zip(self.state_dict().keys(), weights):
            if v.ndim != 0:
                layer_dict[k] = torch.Tensor(np.array(v, copy=True))
        state_dict = OrderedDict(layer_dict)
        self.load_state_dict(state_dict, strict=True)


def train(
    task_type: str,
    net: CNN,
    trainloader: DataLoader,
    device: torch.device,
    num_iterations: int,
    log_progress: bool = True,
) -> Tuple[float, float, int]:
    # Define loss and optimizer
    if task_type == "classification":
        criterion = nn.BCELoss()
    elif task_type == "regression":
        criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))

    def cycle(iterable):
        """Repeats the contents of the train loader, in case it gets exhausted in 'num_iterations'."""
        while True:
            for x in iterable:
                yield x

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
                pbar.set_postfix(
                    {
                        "train_loss": total_loss / n_samples,
                        "train_acc": total_result / n_samples,
                    }
                )
            elif task_type == "regression":
                pbar.set_postfix(
                    {
                        "train_loss": total_loss / n_samples,
                        "train_mse": total_result / n_samples,
                    }
                )
    if log_progress:
        print("\n")

    return total_loss / n_samples, total_result / n_samples, n_samples


def test(
    task_type: str,
    net: CNN,
    testloader: DataLoader,
    device: torch.device,
    log_progress: bool = True,
) -> Tuple[float, float, int]:
    """Evaluates the network on test data."""
    if task_type == "classification":
        criterion = nn.BCELoss()
    elif task_type == "regression":
        criterion = nn.MSELoss()

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
                acc = Accuracy(task="binary")(
                    outputs.cpu(), labels.type(torch.int).cpu()
                )
                total_result += acc * labels.size(0)
            elif task_type == "regression":
                mse = MeanSquaredError()(outputs.cpu(), labels.type(torch.int).cpu())
                total_result += mse * labels.size(0)

    if log_progress:
        print("\n")

    return total_loss / n_samples, total_result / n_samples, n_samples

def get_data(reg : bool = False): 
    # Select the downloaded training and test dataset
    if reg:
        train_path, test_path = ["eunite2001", "YearPredictionMSD"], ["eunite2001.t", "YearPredictionMSD.t"]
        dataset_path, task = "dataset/regression/", 'regression'
    else:
        train_path, test_path = ["cod-rna.t", "cod-rna", "ijcnn1.t"], ["cod-rna.r", "cod-rna.t", "ijcnn1.tr"]
        dataset_path, task = "dataset/binary_classification/", 'classification'

    train, test = train_path[0], test_path[0]
    data_train = load_svmlight_file(dataset_path + train, zero_based=False)
    data_test = load_svmlight_file(dataset_path + test, zero_based=False)

    print("Task type selected is: " + task)
    print("Training dataset is: " + train)
    print("Test dataset is: " + test)

    return data_train, data_test

def retreive_data(path : str, reg : bool = False) -> None:
    """ 
        Retreives data for regression or classification task.
        
        Parameters: 
            path: Path to data (str)
            reg: Whether task is regression or not (then classificaiton). 

        Returns:
            None
    """
    if reg: 
        print("Regression task. ")
        if not os.path.exists(path):
            os.makedirs(path)
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/eunite2001",
                f"{os.path.join(path, 'eunite2001')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/eunite2001.t",
                f"{os.path.join(path, 'eunite2001.t')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2",
                f"{os.path.join(path, 'YearPredictionMSD.bz2')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2",
                f"{os.path.join(path, 'YearPredictionMSD.t.bz2')}",
            )
            for filepath in os.listdir(path):
                if filepath[-3:] == "bz2":
                    abs_filepath = os.path.join(path, filepath)
                    with bz2.BZ2File(abs_filepath) as fr, open(abs_filepath[:-4], "wb") as fw:
                        shutil.copyfileobj(fr, fw)
        else:
            print("Data already downloaded. ")
    else:
        print("Classification task. ")
        if not os.path.exists(path):
            print("Downloading data ...")
            os.makedirs(path)
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna",
                f"{os.path.join(path, 'cod-rna')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t",
                f"{os.path.join(path, 'cod-rna.t')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.r",
                f"{os.path.join(path, 'cod-rna.r')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2",
                f"{os.path.join(path, 'ijcnn1.t.bz2')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2",
                f"{os.path.join(path, 'ijcnn1.tr.bz2')}",
            )
            for filepath in os.listdir(path):
                if filepath[-3:] == "bz2":
                    abs_filepath = os.path.join(path, filepath)
                    with bz2.BZ2File(abs_filepath) as fr, open(abs_filepath[:-4], "wb") as fw:
                        shutil.copyfileobj(fr, fw)
        else:
            print("Data already downloaded. ")


def plot_xgbtree(tree: Union[XGBClassifier, XGBRegressor], n_tree: int) -> None:
    """Visualize the built xgboost tree."""
    xgb.plot_tree(tree, num_trees=n_tree)
    plt.rcParams["figure.figsize"] = [50, 10]
    plt.show()


def construct_tree(
    dataset: Dataset, label: NDArray, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset."""
    if tree_type == "classification":
        tree = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.1,
            max_depth=8,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=1,
            alpha=5,
            gamma=5,
            num_parallel_tree=1,
            min_child_weight=1,
        )

    elif tree_type == "regression":
        tree = xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.1,
            max_depth=8,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=1,
            alpha=5,
            gamma=5,
            num_parallel_tree=1,
            min_child_weight=1,
        )

    tree.fit(dataset, label)
    return tree


def construct_tree_from_loader(
    dataset_loader: DataLoader, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset loader."""
    for dataset in dataset_loader:
        data, label = dataset[0], dataset[1]
    return construct_tree(data, label, n_estimators, tree_type)


def single_tree_prediction(
    tree: Union[XGBClassifier, XGBRegressor], n_tree: int, dataset: NDArray
) -> Optional[NDArray]:
    """Extract the prediction result of a single tree in the xgboost tree
    ensemble."""
    # How to access a single tree
    # https://github.com/bmreiniger/datascience.stackexchange/blob/master/57905.ipynb
    num_t = len(tree.get_booster().get_dump())
    if n_tree > num_t:
        print(
            "The tree index to be extracted is larger than the total number of trees."
        )
        return None

    return tree.predict(  # type: ignore
        dataset, iteration_range=(n_tree, n_tree + 1), output_margin=True
    )


def tree_encoding(  # pylint: disable=R0914
    trainloader: DataLoader,
    client_trees: Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ],
    client_tree_num: int,
    client_num: int,
) -> Optional[Tuple[NDArray, NDArray]]:
    """Transform the tabular dataset into prediction results using the
    aggregated xgboost tree ensembles from all clients."""
    if trainloader is None:
        return None

    for local_dataset in trainloader:
        x_train, y_train = local_dataset[0], local_dataset[1]

    x_train_enc = np.zeros((x_train.shape[0], client_num * client_tree_num))
    x_train_enc = np.array(x_train_enc, copy=True)

    temp_trees: Any = None
    if isinstance(client_trees, list) is False:
        temp_trees = [client_trees[0]] * client_num
    elif isinstance(client_trees, list) and len(client_trees) != client_num:
        temp_trees = [client_trees[0][0]] * client_num
    else:
        cids = []
        temp_trees = []
        for i, _ in enumerate(client_trees):
            temp_trees.append(client_trees[i][0])  # type: ignore
            cids.append(client_trees[i][1])  # type: ignore
        sorted_index = np.argsort(np.asarray(cids))
        temp_trees = np.asarray(temp_trees)[sorted_index]

    for i, _ in enumerate(temp_trees):
        for j in range(client_tree_num):
            x_train_enc[:, i * client_tree_num + j] = single_tree_prediction(
                temp_trees[i], j, x_train
            )

    x_train_enc32: Any = np.float32(x_train_enc)
    y_train32: Any = np.float32(y_train)

    x_train_enc32, y_train32 = torch.from_numpy(
        np.expand_dims(x_train_enc32, axis=1)  # type: ignore
    ), torch.from_numpy(
        np.expand_dims(y_train32, axis=-1)  # type: ignore
    )
    return x_train_enc32, y_train32

def get_dataloader(
    dataset: Dataset, partition: str, batch_size: Union[int, str]
) -> DataLoader:
    if batch_size == "whole":
        batch_size = len(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train")
    )

# https://github.com/adap/flower
def do_fl_partitioning(
    trainset: Dataset,
    testset: Dataset,
    pool_size: int,
    batch_size: Union[int, str],
    val_ratio: float = 0.0,
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