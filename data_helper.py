import torch
from torchvision import datasets, transforms


def load_train_data(data_directory):
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Keep batch size small as my gpu only has 6GB (which was a lot when I bought it)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=16, shuffle=True)

    dataset = []
    dataset.append(trainloader)
    dataset.append(validloader)

    return dataset
