import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from transformers import ResNetModel, AutoTokenizer
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torch import optim
import re
from tqdm import tqdm, trange
import torchvision
import torchvision.transforms.functional as TF
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import evaluate

csv_path = "food_dataset/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
image_path = "food_dataset/Food Images/Food Images/"

df = pd.read_csv(csv_path, index_col=0)
df.head()

image_files = os.listdir(image_path)

image_formats = set([os.path.splitext(img)[1].lower() for img in image_files])

# Check if all images have the same format
if len(image_formats) == 1:
    print(f"All images are in the same format: {image_formats.pop()}")
else:
    print(f"Different formats found: {image_formats}")

# Check if all image names in the CSV are present in the image folder
missing_images = []
for img_name in df['Image_Name']:
    img_file = img_name + '.jpg'
    if img_file not in image_files:
        missing_images.append(img_name)

if missing_images:
    print(f"Missing images: {missing_images}")
else:
    print("All images in the CSV are present in the folder.")

# Remove rows with invalid 'Image_Name' entries (e.g., '#NAME?')
df_cleaned = df[df['Image_Name'] != '#NAME?']

# Check the remaining missing images
missing_images_cleaned = []
for img_name in df_cleaned['Image_Name']:
    img_file = img_name + '.jpg'
    if img_file not in image_files:
        missing_images_cleaned.append(img_name)

if missing_images_cleaned:
    print(f"Missing images after cleaning: {missing_images_cleaned}")
else:
    print("No missing images after cleaning.")

df_cleaned = df_cleaned.drop(columns=['Ingredients', 'Instructions', 'Cleaned_Ingredients'])

class Config:
    maxlen = 12 # max sequence length
    image_size = 128 # image size for resizing
    batch_size = 128 # batch size per dataloader
    epochs = 20 # num epochs
    learning_rate = 1e-4 # learning rate for optimizer
    d_model = 512   # model's dimension
    n_heads_encoder = 1 # encoders num_heads for multihead attention
    n_heads_decoder = 2 # decoders num_heads for multihead attention
    txt = df_cleaned
    images_path = r"food_dataset/Food Images/Food Images/"  # path to images file for preprocessing step.
    vocab_size = None # this will be changed later

config = Config

device = "cuda" if torch.cuda.is_available() else "cpu"


def format_recipe_name(input_string):
    """
    Generate titles from the image name for the images without title attribute
    """
    # Split the string by '-'
    parts = input_string.split('-')
    
    # Remove numeric parts (if any)
    parts = [part for part in parts if not part.isdigit()]
    
    # Join the remaining parts with spaces and capitalize the first letter
    formatted_string = ' '.join(parts).capitalize()
    
    return formatted_string

for index, row in df_cleaned.iterrows():
    if pd.isnull(row['Title']):  # Check if Title is NaN
        if pd.isnull(row['Image_Name']):  # If Image_Name is also NaN
            df_cleaned.drop(index, inplace=True)  # Drop the row
        else:
            # Use format_recipe_name to generate a Title
            df_cleaned.at[index, 'Title'] = format_recipe_name(row['Image_Name'])

# Reset the index after dropping rows
df_cleaned.reset_index(drop=True, inplace=True)

nan_rows = df_cleaned[df_cleaned['Title'].isnull()]
print(nan_rows)

def load_captions_data(config:Config):
    """
    Loads and prepares captioning data.

    Arguments:
        config: Configuration file.

    Returns:
        caption_mapping: Dictionary that contains image paths paired with captions.
        word2idx: dictionary that contains words corresponding to their numerical representation.
    """

    data = config.txt

    caption_mapping = {}
    text_data = set()
    images_to_skip = set()
    word2idx = {"<PAD>":0,"<SOS>":1,"<EOS>":2}

    for caption, image in zip(data['Title'], data['Image_Name']):
        img_name = os.path.join(config.images_path,image)+'.jpg'
        caption = re.sub(r'[^\w\s]', '', caption)
        tokens = caption.strip().split()
        for token in tokens:
            text_data.add(token.lower())
        
        if len(tokens) < 5 and len(tokens) > config.maxlen:
            images_to_skip.add(img_name)

        if img_name.endswith(".jpg") and img_name not in images_to_skip:
            caption = caption.strip() 
        
        if img_name in caption_mapping:
                caption_mapping[img_name].append(caption)
        else:
            caption_mapping[img_name] = [caption]
        
    for img_name in images_to_skip:
        if img_name in caption_mapping:
            del caption_mapping[img_name]
    
    len_word2idx = len(word2idx)
    for i,token in enumerate(text_data):
        word2idx[token] = len_word2idx+i

    return caption_mapping, word2idx

def train_val_split(caption_data:dict, train_size = 0.8, shuffle = True):
    """
    Splits data to train and validation.

    Arguments:
        caption_data (dict): caption data that contains image-caption pairs.
        train_size (float): indicates size of the train. Must be fraction.
        shuffle (boolean): shuffles data before split.

    Returns:
        train_data (dict): splitted train data.
        val_data (dict): splitted validation data.
    """

    all_images = list(caption_data.keys())

    if shuffle:
        np.random.shuffle(all_images)

    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    return training_data,validation_data

captions_mapping , word2idx = load_captions_data(config)

train_data, valid_data = train_val_split(captions_mapping)
print("number of training samples: ",len(train_data))
print("number of validation samples: ",len(valid_data))

# setting config.vocab_size with length of word2idx
config.vocab_size = len(word2idx)

idx2word = {idx: word for word, idx in word2idx.items()}


class CaptionDataset(Dataset):
    def __init__(self, caption_data: dict, config: Config, model_name: str):
        """
        Inializes classes construction method.

        Arguments:
            caption_data (dict): Caption data prepared and processed before.
            config: Configuration class.
            model_name (str): Pretrained model name for AutoTokenizer (e.g., "distilbert-base-uncased").

        Returns:
            None.
        """
        self.caption_data = caption_data
        self.config = config

        # Initialize AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((config.image_size, config.image_size)),
                torchvision.transforms.ToTensor()
            ]
        )
        self.images = list(caption_data.keys())
        self.captions = list(caption_data.values())

    # loads and prepares images using torchvision.transforms
    def load_and_prepare_images(self, image_path):
        """
        Loads and prepares given image path.

        Arguments:
            image_path (str): Path of the image.

        Returns:
            image (torch.Tensor): Preprocessed tensor type of the image.
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        return image

    # loads and prepares captions and returns list of preprocessed captions.
    def load_and_prepare_captions(self, captions: list):
        """
        Loads and prepares given list of captions using AutoTokenizer.

        Arguments:
            captions (list): List of captions. Contains 5 elements per image pair.

        Returns:
            preprocessed_captions (list): List of preprocessed captions that contains: input_tokens, target_tokens, tgt_padding_mask respectively.
        """
        preprocessed_captions = []

        for caption in captions:
            preprocessed_captions.append(caption)

        return preprocessed_captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.load_and_prepare_images(self.images[index])
        y = self.load_and_prepare_captions(self.captions[index])
        return image, y


train_dataset = CaptionDataset(train_data, config, model_name="distilbert-base-uncased")
valid_dataset = CaptionDataset(valid_data, config, model_name="distilbert-base-uncased")

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)


def remove_special_tokens(input_tokens, tokenizer):
    # Get the token IDs for special tokens (assuming tokenizer is already initialized)
    special_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    
    # Remove special tokens from input_tokens
    filtered_tokens = [token for token in input_tokens if token not in special_tokens]
    
    return filtered_tokens

def visualize_samples(dataset, num_samples=5):
    """
    Visualizes samples from the CaptionDataset.

    Args:
        dataset: CaptionDataset instance.
        num_samples: Number of samples to visualize.
    """

    # Randomly select `num_samples` indices
    indices = torch.randint(0, len(dataset), (num_samples,))
    
    for i, idx in enumerate(indices):
        # Get the image and captions
        image, captions = dataset[idx]
        
        # Decode image tensor to a displayable format
        image = TF.to_pil_image(image)  # Convert tensor back to PIL image
        
        # Decode the captions back to text
        decoded_captions = []
        for cap in captions:
            decoded_captions.append(cap)

        # Plot the image
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.axis('off')
        plt.title("\n".join(decoded_captions), fontsize=8)
        plt.show()

# Visualize 5 samples from the training dataset
visualize_samples(valid_dataset, num_samples=5)


# Sinusoidal Positional Embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(-torch.arange(half_dim, device=device) * emb_scale)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # Combine sine and cosine
        return emb


# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, masking=True, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.masking = masking
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, 
                                                    batch_first=True, 
                                                    dropout=dropout)

    def forward(self, x_in, kv_in, key_mask=None):
        # Create causal mask if masking is enabled
        if self.masking:
            l = x_in.size(1)  # Sequence length
            causal_mask = torch.triu(torch.ones(l, l, device=x_in.device), diagonal=1).bool()
        else:
            causal_mask = None

        # Multi-head attention with optional masks
        attn_output, _ = self.multihead_attn(
            query=x_in, key=kv_in, value=kv_in,
            attn_mask=causal_mask,
            key_padding_mask=key_mask
        )
        return attn_output


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, decoder=False, masking=True, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.decoder = decoder
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn1 = AttentionBlock(hidden_size, num_heads, masking, dropout)

        if self.decoder:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.attn2 = AttentionBlock(hidden_size, num_heads, masking=False, dropout=dropout)

        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x, input_key_mask=None, cross_key_mask=None, kv_cross=None):
        # Self-attention
        x = x + self.attn1(x, x, key_mask=input_key_mask)
        x = self.norm1(x)

        # Cross-attention (if decoder)
        if self.decoder:
            x = x + self.attn2(x, kv_cross, key_mask=cross_key_mask)
            x = self.norm2(x)

        # Feedforward network
        x = x + self.mlp(x)
        return self.norm_mlp(x)


# Decoder
class Decoder(nn.Module):
    def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_emb, hidden_size)
        self.pos_emb = SinusoidalPosEmb(hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, decoder=True, masking=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_size, num_emb)

    def forward(self, input_seq, encoder_output, input_padding_mask=None, encoder_padding_mask=None):
        input_embs = self.embedding(input_seq)
        bs, seq_len, h = input_embs.shape

        # Positional embeddings
        seq_idx = torch.arange(seq_len, device=input_seq.device)
        pos_emb = self.pos_emb(seq_idx).unsqueeze(0).expand(bs, -1, -1)
        embs = input_embs + pos_emb

        # Transformer blocks
        for block in self.blocks:
            embs = block(embs, input_key_mask=input_padding_mask, cross_key_mask=encoder_padding_mask, kv_cross=encoder_output)

        # Global caption embedding for contrastive loss
        global_text_emb = embs.mean(dim=1)  # Shape: (bs, hidden_size)

        return self.fc_out(embs), global_text_emb  # Return both token outputs and global embedding

# Image Encoder with ResNet and Transformers
class ImageEncoderWithResNet(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads, num_layers, resnet_variant="resnet50", finetune=False):
        super(ImageEncoderWithResNet, self).__init__()
        resnet = getattr(models, resnet_variant)(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        if finetune:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
        
        resnet_out_channels = resnet.fc.in_features
        self.fc_in = nn.Linear(resnet_out_channels, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, decoder=False, masking=False)
            for _ in range(num_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # For creating global image embedding

    def forward(self, image):
        feature_map = self.feature_extractor(image)  # Shape: (bs, c, h, w)
        bs, c, h, w = feature_map.size()
        grid_size = h * w

        if self.pos_embedding.size(1) != grid_size:
            self.pos_embedding = nn.Parameter(torch.randn(1, grid_size, self.fc_in.out_features, device=image.device))

        feature_seq = feature_map.view(bs, c, -1).permute(0, 2, 1)
        patch_emb = self.fc_in(feature_seq)
        embs = patch_emb + self.pos_embedding
        
        for block in self.blocks:
            embs = block(embs)
        
        # Global image embedding for contrastive loss
        global_image_emb = self.global_pool(embs.permute(0, 2, 1)).squeeze(-1)  # Shape: (bs, embed_dim)
        return embs, global_image_emb  # Return both sequence and global embedding

# Vision Encoder-Decoder
class VisionEncoderDecoder(nn.Module):
    def __init__(self, image_size, channels_in, num_emb, patch_size=16, hidden_size=128, num_layers=(3, 3), num_heads=4, dropout=0.1, finetune=False):
        super(VisionEncoderDecoder, self).__init__()
        self.encoder = ImageEncoderWithResNet(embed_dim=hidden_size, hidden_size=hidden_size, 
                                              num_heads=num_heads, num_layers=num_layers[0], finetune=finetune)
        self.decoder = Decoder(num_emb=num_emb, hidden_size=hidden_size, 
                               num_layers=num_layers[1], num_heads=num_heads, dropout=dropout)

    def forward(self, input_image, target_seq, padding_mask):
        bool_padding_mask = padding_mask == 0
        encoded_seq, global_image_emb = self.encoder(image=input_image)  # Get sequence and global embeddings
        decoded_seq, global_text_emb = self.decoder(input_seq=target_seq, encoder_output=encoded_seq, 
                                                    input_padding_mask=bool_padding_mask)
        return decoded_seq, global_image_emb, global_text_emb


# Create a dataloader itterable object
dataiter = next(iter(train_dataloader))
# Sample from the itterable object
train_images, train_captions = dataiter

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("alexdseo/RecipeBERT")

# Define the learning rate for the optimizer
learning_rate = 1e-4

# Image size
image_size = 128

# Define the number of epochs for training
nepochs = 200

# Define the batch size for mini-batch gradient descent
batch_size = 128

# Embedding Size
hidden_size = 192

# Number of Transformer blocks for the (Encoder, Decoder)
num_layers = (6, 6)

# MultiheadAttention Heads
num_heads = 8

# Size of the patches
patch_size = 16

# Create model
caption_model = VisionEncoderDecoder(image_size=image_size, channels_in=train_images.shape[1], 
                                     num_emb=tokenizer.vocab_size, patch_size=patch_size, 
                                     num_layers=num_layers,hidden_size=hidden_size, 
                                     num_heads=num_heads, finetune=True).to(device)

# Initialize the optimizer with above parameters
optimizer = optim.Adam(caption_model.parameters(), lr=learning_rate)

scaler = torch.amp.GradScaler('cuda')

# Define the loss function
loss_fn = nn.CrossEntropyLoss(reduction="none")

temperature = 0.1
contrastive_weight = 1.3

# Initialize the training loss logger
training_loss_logger = []
contrastive_loss_logger = []

from torch.optim import AdamW

# Define optimizer with different learning rates for encoder and decoder
optimizer = AdamW([
    {"params": caption_model.encoder.feature_extractor.parameters(), "lr": 1e-5},  # Small LR for ResNet
    {"params": caption_model.encoder.fc_in.parameters(), "lr": 5e-5},             # Slightly larger LR
    {"params": caption_model.encoder.blocks.parameters(), "lr": 5e-5},           # Transformer blocks in encoder
    {"params": caption_model.decoder.parameters(), "lr": 1e-4},                  # Larger LR for decoder
])


# Iterate over epochs
for epoch in trange(0, nepochs, leave=False, desc="Epoch"):
    # Set the model in training mode
    caption_model.train()
    steps = 0
    # Iterate over the training data loader
    for images, captions in tqdm(train_dataloader, desc="Training", leave=False):
        
        images = images.to(device)

        captions = [item for sublist in captions for item in sublist]
        
        # Tokenize and pre-process the captions
        tokens = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        token_ids = tokens['input_ids'].to(device)
        padding_mask = tokens['attention_mask'].to(device)
        bs = token_ids.shape[0]
        
        # Shift the input sequence to create the target sequence
        target_ids = torch.cat((token_ids[:, 1:], 
                                torch.zeros(bs, 1, device=device).long()), 1)
        
        tokens_in = token_ids
        
        with torch.amp.autocast('cuda'):
            # Forward pass
            pred, image_embeds, text_embeds = caption_model(images, tokens_in, padding_mask=padding_mask)

        # Compute the loss
        image_embeds = F.normalize(image_embeds, dim=-1)  # (B, D)
        text_embeds = F.normalize(text_embeds, dim=-1)    # (B, D)

        # Compute similarity scores
        similarity_matrix = torch.matmul(image_embeds, text_embeds.T)  # (B, B)

        # Create targets: diagonal should represent correct pairs
        targets = torch.arange(similarity_matrix.size(0)).to(device)

        # Compute contrastive loss
        contrastive_loss = F.cross_entropy(similarity_matrix / temperature, targets)

        crossentropy_loss = (loss_fn(pred.transpose(1, 2), target_ids) * padding_mask).mean()

        loss = crossentropy_loss + contrastive_weight * contrastive_loss

        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        
        # Log the training loss
        training_loss_logger.append(crossentropy_loss.item())
        contrastive_loss_logger.append(contrastive_loss.item())

_ = plt.figure(figsize=(10, 5))
_ = plt.plot(training_loss_logger[100:])
_ = plt.title("Training Loss")


# TESTING
# Create a dataloader itterable object
dataiter = next(iter(valid_dataloader))
# Sample from the itterable object
test_images, test_captions = dataiter

# Prepare lists to store predictions, ground truth captions, and F1 scores
all_generated_captions = []
all_ground_truth_captions = []
all_f1_scores = []

# Set the model to evaluation mode
caption_model.eval()

# Iterate through the validation dataloader
for batch in tqdm(valid_dataloader, desc="Evaluating"):
    test_images, test_captions = batch  # Get the batch of images and their ground truth captions
    
    for index in range(test_images.size(0)):
        test_image = test_images[index].unsqueeze(0)
        
        sos_token = 101 * torch.ones(1, 1).long()
        log_tokens = [sos_token]
        temp = 0.5  # Sampling temperature
        
        with torch.no_grad():
            # Encode the input image
            with torch.cuda.amp.autocast():
                image_embedding, _ = caption_model.encoder(test_image.to(device))
            
            # Generate tokens for the caption
            for i in range(50):  # Limit to 50 tokens
                input_tokens = torch.cat(log_tokens, 1)
                data_pred, _ = caption_model.decoder(input_tokens.to(device), image_embedding)
                dist = Categorical(logits=data_pred[:, -1] / temp)
                next_tokens = dist.sample().reshape(1, 1)
                log_tokens.append(next_tokens.cpu())
                if next_tokens.item() == 102:  # End-of-caption token
                    break
        
        # Convert generated tokens to text
        pred_text = torch.cat(log_tokens, 1)
        pred_text_strings = tokenizer.decode(pred_text[0], skip_special_tokens=True)
        generated_caption = "".join(pred_text_strings)
        print("PRED:",generated_caption)
        all_generated_captions.append(generated_caption)

        # Append all ground truth captions for this image (could be multiple references)
        ground_truth_caption = "".join(test_captions[0][index]).lower()
        all_ground_truth_captions.append(ground_truth_caption)
        print("GT:",ground_truth_caption)
        print()

        # Tokenize the generated caption into words for F1 calculation
        generated_caption_tokens = generated_caption.split()


# METRICS
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

ref = [[r] for r in all_ground_truth_captions]
cand = all_generated_captions

bleu1 = bleu.compute(predictions=cand, references=ref, max_order=1)
bleu2 = bleu.compute(predictions=cand, references=ref, max_order=2)
rouge_l = rouge.compute(predictions=cand, references=ref)
meteor_metric = meteor.compute(predictions=cand, references=ref)

print(f"\nBLEU-1: {bleu1['bleu']*100:.1f}%, BLEU-2: {bleu2['bleu']*100:.1f}%, ROUGE-L: {rouge_l['rougeL']*100:.1f}%, METEOR: {meteor_metric['meteor']*100:.1f}%")

def plot_metrics_and_save(metrics, plot_path, title):
    metric_names = ['BLEU-1', 'BLEU-2', 'ROUGE-L', 'METEOR']
    plt.figure(figsize=(10, 6))
    plt.bar(metric_names, metrics, color=['blue', 'orange', 'green', 'red'])
    plt.title(title)
    plt.ylabel('Metric Value')
    plt.savefig(plot_path)
    plt.close()

folder_path = "transformers_results"
# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)
metrics = [bleu1['bleu'], bleu2['bleu'], rouge_l['rougeL'], meteor_metric['meteor']]
plot_path = os.path.join(folder_path, "metrics_transf_contrloss.png")
plot_metrics_and_save(metrics, plot_path, title="Test Metrics")