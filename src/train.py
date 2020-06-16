# Imports here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
from graph import Plotter
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import sys
from pathlib import Path


save_name = sys.argv[1]
dataset_type = sys.argv[2]


print(save_name, dataset_type)


project_root_dir = Path(__file__).resolve().parents[1]
print(str(project_root_dir))


if dataset_type == "simple":
    data_dir = str(project_root_dir) + "/dataset-beamng-simple"

elif dataset_type == "full":
    data_dir = str(project_root_dir) + "/dataset-beamng"
elif dataset_type == "cropped":
    data_dir = str(project_root_dir) + "/dataset-beamng-cropped"
elif dataset_type == "cropped-extra":
    data_dir = str(project_root_dir) + "/dataset-beamng-cropped-extra"
else:
    sys.exit("Invalid Argument")


print(str(data_dir))
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# tranforms are being devined
training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])

#dataset is being loaded
training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

#defining loaders from dataset
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=30, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=14)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=14)
###################################################################################
#import pretrained models for transfer learning

model = models.vgg16(pretrained=True)
#model = models.alexnet(pretrained=True)
#model = models.densenet121(pretrained=True)



#freese the parameters of the pretrained model
for parameter in model.parameters():
    parameter.requires_grad = True



#Create custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(5000, 2000)),
                                        ('relu', nn.ReLU()),
                                        ('drop',nn.Dropout(p=0.5)),
                                        ('fc3', nn.Linear(2000, 2)),
                                        ('output', nn.LogSoftmax(dim=1))]))

#Create custom classifier

# classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(9216, 4096)),
#                                         ('relu', nn.ReLU()),
#                                         ('drop', nn.Dropout(p=0.5)),
#                                         ('fc2', nn.Linear(4096, 500)),
#                                         ('relu', nn.ReLU()),
#                                         ('drop',nn.Dropout(p=0.5)),
#                                         ('fc3', nn.Linear(500, 2)),
#                                         ('output', nn.LogSoftmax(dim=1))]))


model.classifier = classifier



print(model)

#####################################################################################

def test_accuracy(model, test_loader):
    
  # Do validation on the test set
  model.eval()
  model.to('cuda')

  with torch.no_grad():
  
      accuracy = 0
  
      for images, labels in iter(test_loader):
  
          images, labels = images.to('cuda'), labels.to('cuda')
  
          output = model.forward(images)

          probabilities = torch.exp(output)
      
          equality = (labels.data == probabilities.max(dim=1)[1])
      
          accuracy += equality.type(torch.FloatTensor).mean()
      
      print("Test Accuracy: {}".format(accuracy/len(test_loader)))


# Function for the validation pass
def validation(model, validateloader, criterion):
    
    val_loss = 0
    accuracy = 0
    
    for images, labels in iter(validateloader):

        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)
        
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy


def save_model(model, name=None):

    model.class_to_idx = training_dataset.class_to_idx

    checkpoint = {'arch': save_name,
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict()
                 }


    if name is not None:
        torch.save(checkpoint, str(project_root_dir)+"/saved-models/"+save_name+name+'.pth')
        print("model saved with name at {}".format(name))
    else:
        torch.save(checkpoint, str(project_root_dir)+"/saved-models/"+save_name+'.pth')
        print("model saved")


# Loss function and gradient descent
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

def train_classifier():


  epochs = 35
  print_every = 40

  higherst_acc = 0
  highest_acc_time = 0

  plot = Plotter(save_name)
  model.to('cuda')

  for e in range(epochs):
  
      model.train()

      running_loss = 0
      total_im = len(train_loader)

      index = 1
      
      for images, labels in iter(train_loader):
          
          images, labels = images.to('cuda'), labels.to('cuda')
  
          optimizer.zero_grad()
  
          output = model.forward(images)
          loss = criterion(output, labels)
          loss.backward()
          optimizer.step()
  
          running_loss += loss.item()
  

          model.eval()

          with torch.no_grad():
              validation_loss, accuracy = validation(model, validate_loader, criterion)
          
          train_loss = running_loss/print_every
          val_loss = validation_loss/len(validate_loader)
          val_acc = accuracy/len(validate_loader)


          process_in_time = index/total_im
          process_in_time = e + process_in_time    
          
          
          
          #only save the model when it reaches an higher accuracy
          if val_acc > higherst_acc:
              save_model(model)
              higherst_acc = val_acc
              highest_acc_time = process_in_time

              list_of_highest = [highest_acc_time, higherst_acc]


          plot.add_data(train_loss, val_loss, val_acc, process_in_time, list_of_highest)
          plot.plot_graph()
          
          print("Epoch: {}/{}: ".format(e+1, epochs),
                "Training Loss: {:.5f}".format(train_loss),
                  "Validation Loss: {:.5f}.. ".format(val_loss),
                  "Validation Accuracy: {:.5f}".format(val_acc),
                  "Process {:.2f}".format(process_in_time))
        
          running_loss = 0
          model.train()

          index  = index + 1


          

          torch.cuda.empty_cache()
  

  
train_classifier()
save_model(model, "-end")          
test_accuracy(model, test_loader)


