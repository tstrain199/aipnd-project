import argparse
import torch
import model_helper
import data_helper


def main():
    parser = argparse.ArgumentParser(description="Train a Neural Network")
    parser.add_argument('data_directory', action="store",
                        help='Path to data', metavar='DIR',
                        default='flowers')

    parser.add_argument('--save_dir', action='store',
                        help='Directory to save checkpoints', metavar='DIR',
                        default='./', dest='save_dir')

    parser.add_argument('--arch', action='store', help='Learning model arch',
                        default='vgg16', dest='arch')

    parser.add_argument('--hidden_units', action='store', type=int,
                        help='Hyperparameter for number of hidden units',
                        default='512', dest='hidden_units')

    parser.add_argument('--learning_rate', action='store', type=float,
                        help='Hyperparameter for learning rate', default='.003',
                        dest='learning_rate')

    parser.add_argument('--epochs', action='store', type=int,
                        help='Hyperparameter for number of epochs', dest='epochs',
                        default='2')

    parser.add_argument('--gpu', action='store_true', help='Run trainig on gpu',
                        dest='gpu', default=False)


    # Set variables and conditions
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu :
        gpu = True
    else:
        gpu = False

    print(gpu)

    data_set, class_to_idx  = data_helper.load_train_data(args.data_directory)
    model_helper.trainer(data_set, class_to_idx, args.hidden_units, args.learning_rate,
                         args.epochs, args.arch, gpu, args.save_dir)


if __name__ == "__main__":
        main()
