import importlib
import argparse
import os
import tempfile
import ntpath
import json

# Get current folder for default save_directory value
current_dir = os.path.dirname(os.path.realpath(__file__))

train_data_dir = str()
test_data_dir = str()
valid_data_dir = str()
checkpoint_path = str()

json_category_to_name = None

# Image pre-processing parameters
image_resize = 255
image_centercrop = 224
image_normalization_mean = [0.485, 0.456, 0.406]
image_normalization_stdev = [0.229, 0.224, 0.225]

def check_positive_int(x):
    try:
        value = int(x)
        if value <= 0:
            raise
    except:
        raise argparse.ArgumentTypeError("{} is not a positive integer".format(x))
    return value

def get_predefined_model_class(name):
    return getattr(importlib.import_module("torchvision.models"), name)

def check_predefined_model_exists(name):
    try:
        get_predefined_model_class(name)
    except:
        raise argparse.ArgumentTypeError("{} does not refer to an existing PyTorch pre-defined model".format(name))
    return name

def check_data_directory(name):
    try:
        global train_data_dir
        global test_data_dir
        global valid_data_dir
        absolute_path = os.path.abspath(name)
        train_data_dir = os.path.join(absolute_path, "train")
        test_data_dir = os.path.join(absolute_path, "test")
        valid_data_dir = os.path.join(absolute_path, "valid")
        dirs_exist = os.path.isdir(absolute_path)
        dirs_exist = dirs_exist and os.path.isdir(train_data_dir)
        dirs_exist = dirs_exist and os.path.isdir(test_data_dir)
        dirs_exist = dirs_exist and os.path.isdir(valid_data_dir)
        if not dirs_exist:
            raise
    except:
        raise argparse.ArgumentTypeError("'{}' directory does not exist or does not contain the train/test/valid directories".format(name))
    return absolute_path

def check_save_dir(name):
    try:
        # Generate a random filename respecting the 'checkpoint-XXXX.pth' scheme
        random_filename = ntpath.basename(tempfile.NamedTemporaryFile(prefix='checkpoint-', suffix='.pth').name)
        global checkpoint_path
        absolute_path = os.path.abspath(name)
        if not os.path.isdir(absolute_path):
            raise
    except:
        absolute_path = current_dir
    checkpoint_path = os.path.join(absolute_path, random_filename)
    return checkpoint_path

def check_file_exists(path):
    try:
        absolute_path = os.path.abspath(path)
        if not os.path.isfile(absolute_path):
            raise
    except:
        raise argparse.ArgumentTypeError("'{}' does not refer to an existing and valid file".format(path))
    return absolute_path

def check_json_file(path):
    absolute_path = check_file_exists(path)
    try:
        global json_category_to_name
        with open(absolute_path, mode='r') as json_file:
            json_category_to_name = json.load(json_file)
    except ValueError:
        raise argparse.ArgumentTypeError("'{}' does not contain a valid JSON".format(path))
    return absolute_path

def print_trainer_config(config):
    print("")
    print("Current configuration:")
    print("\t- data directory:", config.data_directory)
    print("\t- checkpoint save directory:", config.save_dir)
    print("\t- base architecture:", config.arch)
    print("\t- use GPU for training:", config.gpu)
    print("\t- hyperparameters:")
    print("\t\t* learning rate:", config.learning_rate)
    print("\t\t* hidden units:", config.hidden_units)
    print("\t\t* epochs:", config.epochs)
    print("")

def print_predictor_config(config):
    print("")
    print("Current configuration:")
    print("\t- path to image:", config.path_to_image)
    print("\t- path to checkpoint:", config.path_to_checkpoint)
    print("\t- top K class probabilities to output:", config.top_k)
    print("\t- use GPU for inference:", config.gpu)
    if config.category_names != None:
        print("\t- JSON file for class category-to-name mapping:", config.category_names)
    print("")

def get_trainer_config():
    parser = argparse.ArgumentParser(description='Train a neural network from a pre-trained one, to classify images.')
    parser.add_argument('data_directory', action='store', type=check_data_directory, help='The folder in which the dataset is stored')
    parser.add_argument('--save_dir', action='store', type=check_save_dir, default=current_dir, help='The folder to save the checkpoint to')
    parser.add_argument('--arch', action='store', type=check_predefined_model_exists, default='vgg11', help='The name of a pre-defined neural network to use as base architecture')
    parser.add_argument('--learning_rate', action='store', type=float, default=0.001, help='Learning rate used to train the model')
    parser.add_argument('--hidden_units', action='store', type=check_positive_int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', action='store', type=check_positive_int, default=20, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for training')
    params_dict = parser.parse_args()
    return params_dict

def get_predictor_config():
    parser = argparse.ArgumentParser(description='Infer category of an image, by computing the attached probability')
    parser.add_argument('path_to_image', action='store', type=check_file_exists, help='Path to the image which category will be inferred')
    parser.add_argument('path_to_checkpoint', action='store', type=check_file_exists, help='Checkpoint file to load the model from')
    parser.add_argument('--top_k', action='store', type=check_positive_int, default=1, help='Returns top K most likely classes')
    parser.add_argument('--category_names', action='store', type=check_json_file, default=None, help='File containing JSON object mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for inference')
    params_dict = parser.parse_args()
    return params_dict