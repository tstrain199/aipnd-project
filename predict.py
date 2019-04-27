import argparse
import model_helper
import data_helper
import json
import torch

def main():
    parser = argparse.ArgumentParser(description="Predict a flower name")
    parser.add_argument('image_path', action="store",
                        help='Path to image',
                        default='flowers/test/21/image_06807.jpg')

    parser.add_argument('checkpoint', action='store',
                        help='Checkpoint to use',
                        default='flower.pth')

    parser.add_argument('--top_k', action='store', help='Length of top results',
                        default='5', dest='top_k', type=int)

    parser.add_argument('--category_names', action='store',
                        help='Path of label mapping file',
                        default='cat_to_name.json', dest='category_names')

    parser.add_argument('--gpu', action='store_true', help='Run trainig on gpu',
                        dest='gpu')

    # Set variables and conditions
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu :
        gpu = True
    else:
        gpu = False

    top_p, top_class, class_to_idx = model_helper.foward(args.image_path, args.checkpoint,
                                           args.top_k, args.gpu)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    inverted_class_to_idx = {class_to_idx[k]: k for k in class_to_idx}

    mapped_classes = []
    names = []

    for label in top_class:
        mapped_classes.append(inverted_class_to_idx[label])

    for c in mapped_classes:
        names.append(cat_to_name[str(c)])

    for i in range(0, len(top_p) -1):
        print('Flower : {}  Probabliity {}'.format(names[i], top_p[i]))

if __name__ == "__main__":
        main()
