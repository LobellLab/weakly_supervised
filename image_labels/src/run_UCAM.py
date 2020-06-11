import argparse
import logging
import os
import datetime
import uuid
from pathlib import Path

from tfrecords import tfrecord_iterator
from UNet import build_UNet
from utils import Params, set_logger
from train_UNet import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='../experiments',
    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='../data',
    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
        help="Optional directory or file containing weights to reload before training")

if __name__ == '__main__':
    # Uncomment below if want reproducible experiments
    # tf.set_random_seed(230)

    # Load parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No params.json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set up logger
    ts_uuid = f'{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S%z")}_{uuid.uuid4().hex[:6]}'
    log_dir = Path(args.model_dir) / 'logs'
    log_dir.mkdir(exist_ok=True)
    set_logger(str(log_dir / f'{ts_uuid}_train.log'))
    logging.info("Creating the datasets from TFRecords...")
    data_dir = args.data_dir
    data_folder = params.data_folder
    train_file = os.path.join(data_dir, params.train_file)
    val_file = os.path.join(data_dir, params.eval_file)
    test_file = os.path.join(data_dir, params.test_file)

    train_filenames = []
    val_filenames = []
    test_filenames = []
    with open(train_file) as f:
        for l in f:
            train_filenames.append(os.path.join(data_dir, data_folder, l[:-1]))
    with open(val_file) as f:
        for l in f:
            val_filenames.append(os.path.join(data_dir, data_folder, l[:-1]))
    with open(test_file) as f:
        for l in f:
            test_filenames.append(os.path.join(data_dir, data_folder, l[:-1]))
    params.train_size = len(train_filenames)
    params.eval_size = len(val_filenames)
    params.test_size = len(test_filenames)

    train_dataset = tfrecord_iterator(True, train_filenames, params)
    eval_dataset = tfrecord_iterator(False, val_filenames, params)
    test_dataset = tfrecord_iterator(False, test_filenames, params)

    logging.info("Creating the model...")
    train_model = build_UNet('train', train_dataset, params)
    eval_model = build_UNet('eval', eval_dataset, params, reuse=True)
    test_model = build_UNet('eval', test_dataset, params, reuse=True)

    logging.info("Starting training for {} epochs".format(params.num_epochs))
    train_and_evaluate(train_model, eval_model, test_model, args.model_dir, params, args.restore_from, ts_uuid)
    


