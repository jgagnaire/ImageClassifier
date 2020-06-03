import utils
from torchvision import transforms, datasets, models
from torch.utils import data
from torch.optim import SGD
from torch import nn
import torch

print("Reading configuration...")
config = utils.get_trainer_config()
utils.print_trainer_config(config)

# Set image transformations
train_transforms = transforms.Compose([
    transforms.Resize(utils.image_resize),
    transforms.CenterCrop(utils.image_centercrop),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(utils.image_normalization_mean, utils.image_normalization_stdev)
])
test_valid_transforms = transforms.Compose([
    transforms.Resize(utils.image_resize),
    transforms.CenterCrop(utils.image_centercrop),
    transforms.ToTensor(),
    transforms.Normalize(utils.image_normalization_mean, utils.image_normalization_stdev)
])

print("Loading datasets...")
BATCH_SIZE = 32
train_set = datasets.ImageFolder(utils.train_data_dir, transform=train_transforms)
test_set = datasets.ImageFolder(utils.test_data_dir, transform=test_valid_transforms)
valid_set = datasets.ImageFolder(utils.valid_data_dir, transform=test_valid_transforms)
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE)
valid_loader = data.DataLoader(valid_set, batch_size=BATCH_SIZE)
print("Done!")
print("")

print("Creating pre-trained model...")
model_class = utils.get_predefined_model_class(config.arch)
model = model_class(pretrained=True)
print("Done!")
print("")

print("Preparing for training:")
print("\t-> Freezing pre-trained model's weights...")
for param in model.parameters():
    param.requires_grad = False

print("\t-> Replacing classifier layers of pre-trained model with our own...")

# Find last module's name
for module in model.named_children():
    pass
last_module_name = module[0]

# Get number of input units of original classifier
for layer in module[1]:
    input_units = layer.in_features
    break

# Replace it
model._modules[last_module_name] = nn.Sequential(
    nn.Linear(input_units, config.hidden_units),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(config.hidden_units, len(train_set.class_to_idx)),
    nn.LogSoftmax(dim=1)
)

print("\t-> Initializing optimizer...")
optimizer = SGD(model._modules[last_module_name].parameters(), lr=config.learning_rate)
print("\t-> Initializing criterion...")
criterion = nn.NLLLoss()
print("\t-> Initializing computing device...")
device = torch.device('cuda' if config.gpu and torch.cuda.is_available() else 'cpu')
model.to(device)
print("Done!")
print("")

print("Beginning training...")
print("Training can be interrupted at any time by pressing ctrl C, the global process will then continue")
try:
    for epoch in range(0, config.epochs):
        batch_training_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_training_loss += loss.item()

        else:
            # deactivate dropout
            model.eval()
            with torch.no_grad():
                number_of_correct_preds = 0
                validation_loss = 0.0
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    validation_loss += criterion(logits, labels).item()
                    probas = torch.exp(logits)
                    _, preds = probas.topk(1, dim=1)
                    equals = preds == labels.view(-1, 1)
                    number_of_correct_preds += torch.sum(equals)
            # reactivate dropout
            model.train()
        print("Epoch #{}:".format(epoch))
        print("\ttraining loss: {:.4f}".format(float(batch_training_loss) / float(len(train_loader))))
        print("\tvalidation loss: {:.4f}".format(float(validation_loss) / float(len(valid_loader))))
        print("\taccuracy: {:.4f}%".format(float(number_of_correct_preds)/float(len(valid_loader) * valid_loader.batch_size) * 100))
        print("")
    print("Training done!")
except KeyboardInterrupt:
    print("Training interrupted by user!")
print("")

print("Measuring accuracy on testing set...")
with torch.no_grad():
    model.eval()
    test_loss = 0.0
    correct_preds = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        test_loss += criterion(logits, labels).item()
        probas = torch.exp(logits)
        _, preds = probas.topk(1, dim=1)
        equals = preds == labels.view(-1, 1)
        correct_preds += torch.sum(equals)
    model.train()
    print("\ttest loss: {:.4f}".format(float(test_loss) / float(len(test_loader))))
    print("\taccuracy: {:.4f}%".format(float(correct_preds) / float(len(test_loader) * test_loader.batch_size) * 100))
print("Done!")
print("")

print("Saving checkpoint to {}...".format(utils.checkpoint_path))
if config.gpu:
    model.to('cpu')
model.class_to_idx = train_set.class_to_idx
torch.save({
    'base_architecture_name': config.arch,
    'last_module_name': last_module_name,
    'model_classifier_layers': [layer for layer in model._modules[last_module_name]],
    'class_to_idx': model.class_to_idx,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, utils.checkpoint_path)
print("Done!")
print("")
