#imports
import predict_utils as put

import argparse
import os
import json


#parse the input
parser = argparse.ArgumentParser()

parser.add_argument('image_path', help="Path to image to predict")
parser.add_argument('checkpoint_path', help="Path to model checkpoint to use")
parser.add_argument('--topk', help="Number of classes and probabilities to display", default=1, type=int)
parser.add_argument('--category_names', help="File with name mappings for classes")
parser.add_argument('--gpu', help="Flag to use GPU for training data, recommended if GPU available.",
                    action='store_true')

A = parser.parse_args()

#process the image
im = put.process_image(A.image_path)

#load the checkpoint into the model
model = put.load_model(A.checkpoint_path)

#predict the flower class
probabilities, classes = put.predict(model, im, A.topk, A.gpu)

#output something
print(f"Probabilities: {probabilities}")
if A.category_names is not None:
    with open(A.category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[key] for key in classes]
    print(f"Names: {names}")
else:
    print(f"Classes: {classes}")