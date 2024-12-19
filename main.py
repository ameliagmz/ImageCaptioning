import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import transforms, models
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import pandas as pd
import evaluate
from models import ResNetEncoder, GRUDecoder, ImageCaptioningModel, CNN_Encoder
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os
import random
from train_test_loop import train_image_captioning_model, test_image_captioning_model

class Config:
    maxlen = 0 # max sequence length (we will change it according to the dataset)
    image_size = 256 # image size for resizing
    batch_size = 64
    epochs = 200 # num epochs
    embedding_dim = 512
    hidden_dim = 512
    attn_dim = 512
    learning_rate = 1e-4 # learning rate for optimizer
    images_path = r"food_dataset/Food Images/Food Images/"  # path to images file for preprocessing step.
    vocab_size = None # this will be changed later

config = Config

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the data
dataset = torch.load('dataset.pt')
train_dataset = torch.load('train_dataset.pt')
val_dataset = torch.load('val_dataset.pt')
test_dataset = torch.load('test_dataset.pt')

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

config.vocab_size = len(dataset.word2idx)
config.maxlen = dataset.maxlen

# Initialize model, criterion, optimizer
encoder = ResNetEncoder(model=models.resnet50, embed_size=256)
decoder = GRUDecoder(hidden_dim=256, vocab_size=config.vocab_size, embedding_dim=256, teacher_forcing_ratio=1)
model = ImageCaptioningModel(encoder, decoder)

PAD_IDX = 0

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # Ignore <PAD> tokens
optimizer = optim.Adam(model.parameters(), lr=1e-4)

folder_path = "model1"

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

train_image_captioning_model(model, train_loader, val_loader, criterion, optimizer, device='cuda', num_epochs=200, folder_path=folder_path, dataset=dataset)
test_image_captioning_model(model, test_loader, criterion, device, folder_path, dataset)

model_file = os.path.join(folder_path, "image_captioning_model1.pth")
torch.save(model, model_file)