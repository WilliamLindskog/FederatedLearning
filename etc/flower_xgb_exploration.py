import os
import flwr as fl
import numpy as np
import xgboost as xgb
import torchvision
import torch, torch.nn as nn
import torch.nn.functional as F

from tqdm import trange, tqdm
from utils import (
    retreive_data, get_data, TreeDataset, central_xgboost, simulate_client,
    train, test, CNN
    )
from typing import Any, Dict, List, Optional, Tuple, Union
from xgboost import XGBClassifier, XGBRegressor
from flwr.common import NDArray, NDArrays
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    GetPropertiesRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Code,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from collections import OrderedDict
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, random_split
from flwr.common.typing import Parameters

print("Imported modules.")

CLASSIFICATION_PATH = os.path.join("dataset", "binary_classification")
REGRESSION_PATH = os.path.join("dataset", "regression")
TASK = 'regression'

assert TASK in ['classification', 'regression']

if TASK == 'regression':
    retreive_data(REGRESSION_PATH, True)
    train_data, test_data = get_data(True)
else:
    retreive_data(CLASSIFICATION_PATH, False)
    train_data, test_data = get_data(False)

X_train, y_train = train_data[0].toarray(), train_data[1]
X_test, y_test = test_data[0].toarray(), test_data[1]
X_train.flags.writeable, y_train.flags.writeable = True, True
X_test.flags.writeable, y_test.flags.writeable = True, True

assert X_train.shape[1] == X_test.shape[1]

if TASK == "classification":
    y_train[y_train == -1], y_test[y_test == -1] = 0, 0

trainset = TreeDataset(np.array(X_train, copy=True), np.array(y_train, copy=True))
testset = TreeDataset(np.array(X_test, copy=True), np.array(y_test, copy=True))

# The number of clients participated in the federated learning
client_num = 5

# The number of XGBoost trees in the tree ensemble that will be built for each client
client_tree_num = 500 // client_num

# Check central xgboost performance
central_xgboost(X_train, y_train, X_test, y_test, TASK, client_tree_num)

# Simulate performance on client side
simulate_client(trainset, testset, client_num, client_tree_num, TASK, X_test, y_test)