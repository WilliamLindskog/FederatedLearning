import os
import bz2
import shutil
import urllib

from sklearn.datasets import load_svmlight_file

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
    """ Retreives data for regression or classification task."""

    if os.path.exists(path): print("Data already downloaded. ")
    else: 
        print("Downloading data ...")
        os.makedirs(path)
        if reg: 
            print("Regression task. ")
            _retrieve_url(path, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/eunite2001")
            _retrieve_url(path, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/eunite2001.t")
            _retrieve_url(path, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2")
            _retrieve_url(path, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2")
        else:
            print("Classification task. ")
            _retrieve_url(path, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna")
            _retrieve_url(path, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t")
            _retrieve_url(path, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.r")
            _retrieve_url(path, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2")
            _retrieve_url(path, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2")
            
        # Copy bz2 files
        _copy_bz2(path=path)

def _retrieve_url(path: str, file_path : str) -> None:
    url_end = file_path.rsplit('/', 1)[-1]
    urllib.request.urlretrieve(file_path,f"{os.path.join(path, url_end)}",)

def _copy_bz2(path : str) -> None:
    for filepath in os.listdir(path):
        if filepath[-3:] == "bz2":
            abs_filepath = os.path.join(path, filepath)
            with bz2.BZ2File(abs_filepath) as fr, open(abs_filepath[:-4], "wb") as fw:
                shutil.copyfileobj(fr, fw)