import utils
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn

print("Reading configuration...")
config = utils.get_predictor_config()
utils.print_predictor_config(config)

# Open image
print("Opening image...")
image = Image.open(config.path_to_image)

# Apply preprocessing
print("Applying image preprocessing...")
image.thumbnail((utils.image_resize, utils.image_resize))
width, height = image.size
crop_left = (width - utils.image_centercrop) / 2
crop_top = (height - utils.image_centercrop) / 2
crop_right = (width + utils.image_centercrop) / 2
crop_bottom = (height + utils.image_centercrop) / 2
image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
arr = np.array(image)
arr = arr.astype('float')
arr /= float(255)
arr = (arr - np.array(utils.image_normalization_mean)) / np.array(utils.image_normalization_stdev)
arr = arr.transpose(2, 0, 1)
tensor_image = torch.Tensor(arr)
# Add batch dimension to tensor so that image can be passed to model for inference
tensor_image.unsqueeze_(0)
print("Done!")
print("")

# Load checkpoint
print("Loading checkpoint...")
checkpoint = torch.load(config.path_to_checkpoint)
model_class = utils.get_predefined_model_class(checkpoint['base_architecture_name'])

# Do not initialize the weights as, they will be loaded through the state_dict
model = model_class(pretrained=False)
model._modules[checkpoint['last_module_name']] = nn.Sequential(*checkpoint['model_classifier_layers'])
model.load_state_dict(checkpoint['model_state_dict'])
model.class_to_idx = checkpoint['class_to_idx']
print("Done!")
print("")

if config.gpu and torch.cuda.is_available():
    print("Loading data to GPU...")
    device = 'cuda'
    model.to(device)
    tensor_image.to(device)
    print("Done!")

# Compute probabilities for all categories
print("Computing predictions...")
print("")
with torch.no_grad():
    model.eval()
    logits = model(tensor_image)
    model.train()

# Filter top K probabilities
probs = torch.exp(logits)
top_probs, classes_idx = probs.topk(config.top_k, dim=1)

# Map class indices on class identifiers
reversed_class_to_idx_dict = {model.class_to_idx[key]: key for key in model.class_to_idx}
if config.gpu:
    classes_idx = classes_idx.cpu()
classes = list()
for i in classes_idx.view(-1):
    classes.append(reversed_class_to_idx_dict[int(i)])

# Map class identifiers on class names
str_classes = list()
j = 0
for i in classes:
    str_classes.append(i)
    if utils.json_category_to_name:
        str_classes[j] += " - " + utils.json_category_to_name[i]
        j += 1

# Write probabilities and attached class
for prob, str_class in zip(top_probs.view(-1), str_classes):
    print("class {}: {:.4f} probability".format(str_class, prob))
print("")
