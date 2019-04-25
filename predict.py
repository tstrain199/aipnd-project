import argparse
import model_helper
import data_helper


def main():
    parser = argparse.ArgumentParser(description="Predict a flower name")
    parser.add_argument('image_path', action="store",
                        help='Path to image,
                        default='flowers/test/21/image_06807.jpg')

    parser.add_argument('checkpoint', action='store',
                        help='Checkpoint to use',
                        default='flower.pth', dest='checkpoint')

    parser.add_argument('--top_k', action='store', help='Length of top results',
                        default='5', dest='top_k')

    parser.add_argument('--catagory_names', action='store', type=int,
                        help='Path of label mapping file',
                        default='cat_to_name.json', dest='catagory_names')

    parser.add_argument('--gpu', action='store_true', help='Run trainig on gpu',
                        dest='gpu')

    # Set variables and conditions
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu :
        gpu = True
    else:
        gpu = False

    data_set = data_helper.load_train_data(args.data_directory)

    model_helper.predict(args.image_path, args.checkpoint, args.top_k,
                         args.catagory_names, args.gpu)
