#!/bin/bash

DATA_DIR=datasets
DATA_FILE_NAME=$1
DATA_PATH=./$DATA_DIR/$DATA_FILE_NAME

if [ ! -f $FILE_PATH ]; then
    echo "File $FILE_PATH not found!"
    exit 1
fi

echo "Running Main Script"
python3 src/main.py --data_path $DATA_PATH