import torch
from torch import nn
from torch import optim
from torchvision import transforms, datasets, models
from workspace_utils import active_session
import os
import time
import numpy as np

data_transforms = {
    'train':transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(255),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

def make_data_loaders(data_dir):
    train_dir = data_dir + '/train'
    if not os.path.isdir(train_dir):
        print('No training directory found')
    valid_dir = data_dir + '/valid'
    if not os.path.isdir(valid_dir):
        print('No validation directory found')
    
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['test'])
    }
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)
    }
    
    return dataloaders, image_datasets['train'].class_to_idx


def make_model(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        fm = 25088

    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        fm = 2208

    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(
        nn.Linear(fm, hidden_units[0]),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(hidden_units[0], hidden_units[1]),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(hidden_units[1], hidden_units[2]),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(hidden_units[2], 102),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    
    return model


def train_model(model, dataloaders, lr, epochs, gpu):
    if gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)
    
    model.to(device)
    
    with active_session():
        times = []
        i = 0
        for e in range(epochs):
            try:
                running_loss = 0
                start = time.time()
                btimes = []
                j = 0
                i += 1
                for inputs, labels in dataloaders['train']:
                    j += 1
                    bstart = time.time()
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    out = model.forward(inputs)
                    loss = criterion(out, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    if j%10 == 0:
                        perc, bleft, running_test_loss, accuracy = validate(j, btimes, model, dataloaders, device, criterion)
                        print(f'{perc}% done with batch, {np.round(bleft, 2)} minutes left')
                        print(f'Train loss: {running_loss/10},',
                              f'Test loss: {running_test_loss/len(dataloaders["valid"])}',
                              f'Test accuracy: {accuracy/len(dataloaders["valid"])}')
                        running_loss = 0
                        model.train()

                    btimes.append(time.time() - bstart)

                times.append(time.time() - start)
                left = (epochs - e - 1) * (sum(times)/len(times))/60

                print(f'\nDone with epoch {e+1}, time left: {left}')
            except KeyboardInterrupt:
                break
    
    return model, optimizer, e+1, running_loss


def validate(j, btimes, model, dataloaders, device, criterion):
    perc = np.round(j/len(dataloaders['train'])*100, 2)
    bleft = (len(dataloaders['train']) - j) * (sum(btimes)/len(btimes))/60

    accuracy = 0
    running_test_loss = 0

    with torch.no_grad():
        model.eval()
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            test_loss = criterion(logps, labels)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            running_test_loss += test_loss.item()
    return perc, bleft, running_test_loss, accuracy