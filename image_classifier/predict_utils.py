from train_utils import make_model
from PIL import Image
import numpy as np
import torch

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    
    image = resize(image)
    image = center_crop(image)
    
    np_image = np.array(image)[:,:,:]
    np_image = np_image/255
    
    np_image -= [0.485, 0.456, 0.406]
    np_image /= [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2,0,1))
    
    return torch.from_numpy(np_image)
    
def resize(image):
    width, height = image.size
    rat = height/width
    if height <= width:
        nheight = 256
        nwidth = int(256/rat)
    else:
        nwidth = 256
        nheight = int(rat*256)
    image = image.resize((nwidth, nheight))
    return image


def center_crop(image):
    width, height = image.size
    
    left = width//2 - 112
    if left < 0:
        left = 0
    right = left + 224
    upper = height//2 - 112
    if upper < 0:
        upper = 0
    lower = upper + 224
    image = image.crop((left, upper, right, lower))
    return image


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    model = make_model(checkpoint['arch'], checkpoint['hidden_units'])
    
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def predict(model, im, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    im = im.to(device)
    im = im.resize(1, 3, 224, 224).float()
    model.to(device)
    
    with torch.no_grad():
        model.eval()
        logps = model.forward(im)
        model.train()
    
    ps = torch.exp(logps)
    probs, indeces = ps.topk(topk)
    probs, indeces = probs.to('cpu'), indeces.to('cpu')
    
    idx_to_class = {value:key for key, value in model.class_to_idx.items()}
    classes = [idx_to_class[key] for key in indeces.numpy().flatten()]
    
    return list(probs.numpy().flatten()), classes