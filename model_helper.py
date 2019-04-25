import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from torchvision import datasets, transforms, models

def trainer(data_set, hidden_units, learning_rate, epochs, arch, gpu, save_dir):

    def arch_to_model(arch):
        switcher = {
            'vgg16': models.vgg16(pretrained=True),
            #'vgg19': models.vgg19(pretrained=True),
            #'resnet18': models.resnet18(pretrained=True),
            #'resnet33': models.resnet33(pretrained=True)
        }
        return switcher.get(arch, 0)

    #Configure all our model stuff here-------------------------->
    model = arch_to_model(arch)
    for param in model.parameters():
        param.requires_grad = False

    features = model.classifier[0].in_features

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(features, hidden_units)),
                              ('dropout', nn.Dropout(.20)),
                              ('relu1',nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Start training here ----------------------------------------->
    if gpu == True:
        device = torch.device('cuda:0')
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

        else:

            with torch.no_grad():
                valid_loss = 0
                accuracy = 0
                model.eval()

                for images, labels in data_set[1]:
                    if gpu == True:
                        images = images.to(device)
                        labels = labels.to(device)

                    logps = model(images)
                    batch_loss =  criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    training_loss = running_loss / len(data_set[0])
                    validation_loss = valid_loss / len(data_set[1])
                    test_accuracy = accuracy / len(data_set[1])

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(training_loss),
                  "Test Loss: {:.3f}.. ".format(validation_loss),
                  "Test Accuracy: {:.3f}".format(test_accuracy))
    # End of model config ----------------------------------------->
    # Start of Save ----------------------------------------------->

    write_dir = save_dir + 'flowers.pth'
    model.class_to_idx = data_set[0].class_to_idx
    torch.save({'epoch' : e,
        'class_to_idx' : model.class_to_idx,
        'model_state_dict' : model.state_dict(),
        'classifier' : classifier,
        'optimizer_dict' : optimizer.state_dict()},
        'icp.pth')

    print('Successfully saved model to {}'.format(write_dir))

def load_model(ckp_path):
    ckp = torch.load(ckp_path)
    model = models.vgg16(pretrained=True)
    model.class_to_idx = ckp['class_to_idx']
    model.classifier = ckp['classifier']
    optimizer = optim.SGD(model.parameters(), lr=0.009)
    optimizer.load_state_dict(ckp['optimizer_dict'])
    model.load_state_dict(ckp['model_state_dict'])
    model.epochs = ckp['epoch']

    return model, optimizer

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
