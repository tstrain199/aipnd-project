from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
import os

def arch_to_model(arch):
    switcher = {
        'vgg11': models.vgg11(pretrained=True),
        'vgg13': models.vgg13(pretrained=True),
        'vgg16' : models.vgg16(pretrained=True),
        'alexnet' : models.alexnet(pretrained=True)
    }

    selected = switcher.get(arch, 0)
    if selected == 0:
        print('The selected model {} is not available. Defaulting to VGG16.'.format(arch))
        return models.vgg16(pretrained=True)
    else:
        return selected

def trainer(data_set, class_to_idx, hidden_units, learning_rate, epochs, arch, gpu, save_dir):

    #Configure all our model, classifier stuff here-------------------------->
    model = arch_to_model(arch)
    if arch == 'alexnet':
        features = model.classifier[1].in_features
    else:
        features = model.classifier[0].in_features

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(features, hidden_units)),
                                            ('dropout', nn.Dropout(.20)),
                                            ('relu1', nn.ReLU()),
                                            ('fc2', nn.Linear(hidden_units, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)

    model.class_to_idx = class_to_idx

    criterion = nn.NLLLoss()

    # Start training here ----------------------------------------->
    if gpu == True:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = model.to(device)

    training_loss = 0
    validation_loss = 0

    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in data_set[0]:

            if gpu == True:
                images = images.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            torch.set_grad_enabled(True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        with torch.no_grad():
            valid_loss = 0
            accuracy = 0
            model.eval()

            for images, labels in data_set[1]:
                if gpu == True:
                    images = images.to(device)
                    labels = labels.to(device)

                logps = model(images)
                batch_loss = criterion(logps, labels)

                valid_loss += batch_loss.item()

                ps = torch.exp(logps)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                #equals = (labels.data == ps.max(1)[1])
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                #print('Top Class {}  Labels {}'.format(top_class, labels.view(*top_class.shape)))
                training_loss = running_loss / len(data_set[0])
                validation_loss = valid_loss / len(data_set[1])
                test_accuracy = accuracy / len(data_set[1])

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(training_loss),
                  "Test Loss: {:.3f}.. ".format(validation_loss),
                  "Test Accuracy: {:.3f}".format(test_accuracy))
    # End of model config ----------------------------------------->

    # Start of Save ----------------------------------------------->
    model.cpu()
    write_dir = os.path.join(save_dir, 'flowers.pth')
    torch.save({'class_to_idx' : model.class_to_idx,
                'arch' : arch,
                'model_state_dict' : model.state_dict(),
                'classifier' : classifier,
                'learning_rate' : learning_rate,
                'optimizer' : optimizer,
                'optimizer_dict' : optimizer.state_dict()},
               write_dir)

    print('Successfully saved model to {}'.format(write_dir))

def load_model(ckp_path):
    ckp = torch.load(ckp_path)
    arch = ckp['arch']
    print('Arch is {}'.format(arch))
    model = arch_to_model(arch)
    model.class_to_idx = ckp['class_to_idx']
    classifier = ckp['classifier']
    learning_rate = ckp['learning_rate']
    model.classifier = classifier
    optimizer = ckp['optimizer']
    optimizer.load_state_dict(ckp['optimizer_dict'])
    return model, model.class_to_idx

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
    img = Image.open(image)
    if img.size[0] < img.size[1]:
        img.thumbnail((256, img.size[1]))
    else:
        img.thumbnail((img.size[0], 256))

    w = img.width
    h = img.height

    dim = 224
    left = (w - dim)/2
    top = (h - dim)/2
    right = (left + dim)
    bot = (top + dim)

    img = img.crop((left, top, right, bot))
    np_img = np.array(img)/255

    mean = np.array([0.485, 0.456, 0.406])
    standard = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) / standard

    np_img = np_img.transpose((2, 0, 1))

    return np_img

def foward(image_path, checkpoint, top_k, gpu):

    model, class_to_idx = load_model(checkpoint)
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)

    if gpu == True:
        device = torch.device('cuda:0')
        model = model.to(device)
        image = image.to(device)

    logps = model(image)
    prob = torch.exp(logps)
    top_p, top_class = prob.topk(top_k, dim=1)


    classes = top_class[0].tolist()
    probs = top_p[0].tolist()
    return probs, classes, class_to_idx
