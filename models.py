import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet201_Weights, ResNet50_Weights, VGG16_Weights
import torch.nn.functional as F
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

class ResNetEncoder(nn.Module):
    def __init__(self, model, embed_size):
        super(ResNetEncoder, self).__init__()
        weights = weights=ResNet50_Weights.DEFAULT
        self.model = model(weights=weights)
        self.embed_size = embed_size
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove last layer
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.hidden_state_layer = nn.Linear(in_features, embed_size)

    def forward(self, images, attention=True):
        features = self.features(images)  # Extract features, shape: (batch_size, channels, H, W)

        # Generate hidden_state by pooling over the spatial dimensions
        pooled_features = torch.mean(features, dim=[2, 3])  # Global average pooling, shape: (batch_size, channels)
        hidden_state = self.hidden_state_layer(pooled_features)

        if attention:
            features = features.permute(0, 2, 3, 1)
            # Flatten the tensor along height and width dimensions to be used in a fully connected
            features = features.view(features.size(0), -1, features.size(-1))
            embed_layer = nn.Linear(features.size(2), self.embed_size) # Linear embedding to get equal dim for all backbones
            embed_layer = embed_layer.to(device)
            features = embed_layer(features)
        
            return features, hidden_state
    
class Attention(nn.Module):
    def __init__(self,in_features=256,decom_space=256,ATTENTION_BRANCHES=1):
        super(Attention, self).__init__()
        self.M = in_features #Input dimension of the Values NV vectors 
        self.L = decom_space # Dimension of Q(uery),K(eys) decomposition space
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES


        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

    def forward(self, x):

        # H feature vector matrix  # NV vectors x M dimensions
        H = x.squeeze(0)
        # Attention weights
        A = self.attention(H)  # NVxATTENTION_BRANCHES
        A = A.permute(0,2,1)  # ATTENTION_BRANCHESxNV
        A = F.softmax(A, dim=1)  # softmax over NV
        
        # Context Vector (Attention Aggregation)
        Z = torch.matmul(A, H)  # ATTENTION_BRANCHESxM 
        
        return Z, A


class DotProdAttention(nn.Module):
    def __init__(self, units, hidden_size=512):
        super(DotProdAttention, self).__init__()
        self.W1 = nn.Linear(units, hidden_size)  # Project features to hidden_size
        self.W2 = nn.Linear(hidden_size, hidden_size)  # Project hidden state to hidden_size
        self.V = nn.Linear(hidden_size, 1)  # Compute attention scores
        self.hidden_size = hidden_size

    def forward(self, features, hidden_state):
        # Step 1: Align features to hidden_size
        features = self.W1(features)  # Shape: (batch_size, num_features, hidden_size)
        
        # Step 2: Query and key alignment
        query = hidden_state.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        key = features
        value = features  # For scaled dot-product, key and value are the same

        # Step 3: Compute scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_size)  # Shape: (batch_size, 1, num_features)

        # Step 4: Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, 1, num_features)

        # Step 5: Compute context vector
        context_vector = torch.matmul(attention_weights, value)  # Shape: (batch_size, 1, hidden_size)

        return context_vector.squeeze(1)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, features, hidden_state):
        """
        Args:
            features: Tensor (batch_size, num_features, embed_dim)
            hidden_state: Tensor (batch_size, embed_dim)
        
        Returns:
            context_vector: Tensor (batch_size, embed_dim), aggregated context.
            attention_weights: Tensor (batch_size, num_features), attention weights for visualization.
        """
        query = hidden_state.unsqueeze(1)  # Shape: (batch_size, 1, embed_dim)
        key = value = features  # Shape: (batch_size, num_features, embed_dim)

        # Multihead attention
        context_vector, attention_weights = self.multihead_attn(query, key, value)
        
        return context_vector.squeeze(1)

class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embedding_dim, teacher_forcing_ratio=0, num_layers=1):
        super(GRUDecoder, self).__init__()
        
        # Embedding layer for text input
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU Decoder
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, 
                          num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for generating word predictions
        self.fc = nn.Linear(hidden_dim, vocab_size)


    def decode_step(self, input_word, hidden_state):
        """
        Decodes one step in the sequence and predicts the next word.
        """
        # Convert input_word to embedding (batch_size, 1, embed_size)
        input_word_embedding = self.embedding(input_word)
        
        # Pass through the GRU to get the next hidden state
        gru_out, hidden_state = self.gru(input_word_embedding, hidden_state)
        
        # Get logits for the next word
        output = self.fc(gru_out.squeeze(1))  # Output shape: (batch_size, vocab_size)
        
        return output, hidden_state
        

    def forward(self, captions, h0, is_training=True, seq_len=None):
        """
        Args:
            features: Image features from the encoder (batch_size, feature_dim)
            captions: Tokenized captions (batch_size, seq_len)
            h0: Initial hidden state for the GRU (num_layers, batch_size, hidden_dim)
            is_training: Indicates whether use gt captions for teacher forcing
            seq_len: Maximum length of the generated sequence
        
        Returns:
            outputs: Predicted word probabilities (batch_size, seq_len, vocab_size)
        """
        
        # return outputs
        if captions is not None:
            # Training mode
            embeddings = self.embedding(captions)  # (batch_size, seq_len, embedding_dim)
            outputs, _ = self.gru(embeddings, h0)  # outputs: (batch_size, seq_len, hidden_dim)
            outputs = self.fc(outputs)  # (batch_size, seq_len, vocab_size)
            return outputs
        else:
            # Inference mode 
            batch_size = h0.size(1)
            device = h0.device
            
            # Start with the <SOS> token
            inputs = torch.full((batch_size, 1), 1, dtype=torch.long).to(device)  # (batch_size, 1)
            generated_captions = []
            outputs_list = []

            # Generate tokens step-by-step
            for _ in range(seq_len):
                # Embed the input token
                embeddings = self.embedding(inputs)  # (batch_size, 1, embedding_dim)
                
                # Decode the embedding
                outputs, h0 = self.gru(embeddings, h0)  # outputs: (batch_size, 1, hidden_dim)
                
                # Predict the next token
                outputs = self.fc(outputs)  # (batch_size, 1, vocab_size)
                outputs_list.append(outputs)

                # Get the token with the highest probability
                predicted_token = outputs.argmax(dim=-1)  # (batch_size, 1)
                generated_captions.append(predicted_token)

                # Use the predicted token as the next input
                inputs = predicted_token

            # Combine outputs and generated tokens
            outputs = torch.cat(outputs_list, dim=1)  # (batch_size, seq_len, vocab_size)
            generated_captions = torch.cat(generated_captions, dim=1)  # (batch_size, seq_len)
            
            return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = Attention()

    def forward(self, images, captions, is_training=True, max_seq_length=None):
        # Encode the image to extract features
        features, hidden_state = self.encoder(images)  # (batch_size, num_features, attn_size) (16,64,512)
        h0, _ = self.attention(features)
        h0 = h0.permute(1,0,2)

        # Decode the features to generate captions
        outputs = self.decoder(captions, h0, is_training, max_seq_length)
        
        return outputs