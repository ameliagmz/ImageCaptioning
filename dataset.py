import os
import json
import numpy as np
import pandas as pd
import torch
import re
import torchvision
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def format_recipe_name(input_string):
    # Split the string by '-'
    parts = input_string.split('-')

    # Remove numeric parts (if any)
    parts = [part for part in parts if not part.isdigit()]

    # Join the remaining parts with spaces and capitalize the first letter
    formatted_string = ' '.join(parts).capitalize()

    return formatted_string

def preprocess_df(csv_path, image_path):
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

    return df_cleaned

def load_captions_data(data, images_path):
    """
    Loads and prepares captioning data.

    Arguments:
        config: Configuration file.

    Returns:
        caption_mapping: Dictionary that contains image paths paired with captions.
        word2idx: dictionary that contains words corresponding to their numerical representation.
    """

    caption_mapping = {}
    text_data = set()
    images_to_skip = set()

    for caption, image in zip(data['Title'], data['Image_Name']):
        img_name = os.path.join(images_path,image)+'.jpg'

        if img_name.endswith(".jpg"):
            caption_mapping[img_name] = caption

    return caption_mapping

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

class CaptionDataset(Dataset):
    def __init__(self, caption_data, image_size):
        self.caption_data = caption_data
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor()
            ]
        )
        self.images = list(caption_data.keys())
        self.captions = list(caption_data.values())
        self.maxlen = max(len(value.split()) for value in caption_data.values())
        self.word2idx = {}
        self.idx2word = {}
        self.lexicon = set()

        for cap in self.captions:
            self.lexicon.update(self.preprocess_captions(cap))
        
        self.lexicon = list(self.lexicon)
        self.build_vocabulary()

    def preprocess_captions(self, sentence, padding=False):
        sentence = sentence.lower()
        words = nltk.word_tokenize(sentence)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        words = [word for word in words if not re.match(r'^[^\w]+$', word)]
        words = [subword for word in words for subword in re.split(r'\W+', word) if subword]
        if padding:
            words.insert(0, "<SOS>")
            words.append("<EOS>")
            n_padding = self.maxlen + 2 - len(words)
            words = words + ["<PAD>" for _ in range(max(0, n_padding))]
        return words
    
    def build_vocabulary(self):
        special_tokens = ["<PAD>","<SOS>", "<EOS>", "<UNK>"]
        vocabulary = special_tokens + sorted(self.lexicon)
        self.word2idx = {word: idx for idx, word in enumerate(vocabulary)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def load_and_prepare_images(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        return image

    def load_and_prepare_captions(self, caption):
        caption = self.preprocess_captions(caption, padding=True)
        caption_indices = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in caption] # If the word is not in the vocab then use "UNK" 
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)
        return caption_tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.load_and_prepare_images(self.images[index])
        length = len(self.captions[index])
        caption = self.load_and_prepare_captions(self.captions[index])
        return image, caption, length
    

csv_path = "food_dataset/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
image_path = "food_dataset/Food Images/Food Images/"

df_cleaned = preprocess_df(csv_path, image_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the dataloaders
captions_mapping = load_captions_data(df_cleaned, image_path)

dataset = CaptionDataset(captions_mapping, image_size=256)

total_size = len(dataset)
test_size = int(0.1 * total_size)
val_size = int(0.1 * total_size)
train_size = total_size - test_size - val_size

# Random split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

torch.save(train_dataset, 'train_dataset.pt')
torch.save(val_dataset, 'val_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')
torch.save(dataset, "dataset.pt")

word2idx = dataset.word2idx
idx2word = dataset.idx2word

with open('word2idx.json', 'w') as f:
    json.dump(word2idx, f)

with open('idx2word.json', 'w') as f:
    json.dump(idx2word, f)