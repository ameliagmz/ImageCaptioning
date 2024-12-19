import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import pandas as pd
import os
import csv
import evaluate
from dataset import preprocess_df, load_captions_data, train_val_split, CaptionDataset
from models import ResNetEncoder, GRUDecoder, ImageCaptioningModel
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import random
import json

bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')


def train_image_captioning_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, folder_path, dataset):
    model.to(device)

    os.makedirs(folder_path, exist_ok=True)

    output_file = os.path.join(folder_path, "training_output.txt")
    train_metrics_plot_path = os.path.join(folder_path, "train_metrics_plot.png")
    val_metrics_plot_path = os.path.join(folder_path, "val_metrics_plot.png")
    train_losses_plot_path = os.path.join(folder_path, "train_val_losses_plot.png")

    cumulative_train_bleu1 = 0
    cumulative_train_bleu2 = 0
    cumulative_train_rouge_l = 0
    cumulative_train_meteor = 0
    cumulative_val_bleu1 = 0
    cumulative_val_bleu2 = 0
    cumulative_val_rouge_l = 0
    cumulative_val_meteor = 0
    train_losses = []
    val_losses = []

    patience=50
    min_delta=0.001
    best_val_loss = float('inf')  # Initialize best validation loss as infinity
    epochs_without_improvement = 0  # Counter for epochs without improvement

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total_words = 0
        train_references = []
        train_hypotheses = []

        train_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
        for i, (images, captions, lengths) in train_pbar:
            images, captions = images.to(device), captions.to(device)

            # Here we prepare the targets by shifting the captions.
            # The input to the model will be the caption (including <SOS>), 
            # and the target will be the same caption shifted by one position (removing <SOS> and adding <EOS>)
            input_captions = captions
            targets = captions
            
            lengths = [l - 1 for l in lengths]  # Adjust lengths for targets
            
            # Forward pass
            optimizer.zero_grad()

            outputs = model(images, input_captions, is_training=True, max_seq_length=targets.size(1)) # (batch_size, seq_length, vocab_size)
        
            vocab_size =  len(dataset.word2idx)
            outputs = outputs[:, 1:, :] 
            outputs = outputs.reshape(-1, vocab_size) # (batch_size * seq_length, vocab_size)
            
            # Reshape outputs and targets for loss calculation
            targets = targets[:,1:].contiguous().view(-1)  # Skip the first token (which is <SOS>)
            
            loss = criterion(outputs, targets)  # Calculate loss
            train_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Accuracy Calculation
            predictions = outputs.argmax(dim=1)
            mask = (targets != 0)  # Mask padding tokens
            train_correct += ((predictions == targets) & mask).sum().item()
            train_total_words += mask.sum().item()

            # BLEU Data Preparation
            predicted_sequences = predictions.view(len(lengths), -1)
            for j in range(len(lengths)):
                ref = [captions[j, 1:lengths[j] + 1].tolist()]  # Ground truth (skip <SOS>)
                hyp = predicted_sequences[j, :lengths[j]].tolist()
                train_references.append(ref)
                train_hypotheses.append(hyp)

            train_pbar.set_postfix(loss=train_loss / (i + 1), acc=train_correct / train_total_words)

        # Process references and hypotheses for BLEU calculation
        processed_train_references, processed_train_hypotheses = process_references_hypotheses(
            train_references, train_hypotheses, dataset)

        train_bleu1 = bleu.compute(predictions=processed_train_hypotheses, references=processed_train_references, max_order=1)
        train_bleu1 = train_bleu1['bleu']

        train_bleu2 = bleu.compute(predictions=processed_train_hypotheses, references=processed_train_references, max_order=2)
        train_bleu2 = train_bleu2['bleu']

        train_rouge_l = rouge.compute(predictions=processed_train_hypotheses, references=processed_train_references)
        train_rouge_l = train_rouge_l['rougeL']

        train_meteor_metric = meteor.compute(predictions=processed_train_hypotheses, references=processed_train_references)
        train_meteor_metric = train_meteor_metric['meteor']

        cumulative_train_bleu1 += train_bleu1
        cumulative_train_bleu2 += train_bleu2
        cumulative_train_rouge_l += train_rouge_l
        cumulative_train_meteor += train_meteor_metric
        
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total_words = 0
        val_references = []
        val_hypotheses = []

        with torch.no_grad():
            val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")
            for i, (images, captions, lengths) in val_pbar:
                images, captions = images.to(device), captions.to(device)

                # Prepare validation targets in the same way
                lengths = [l - 1 for l in lengths]
                input_captions = captions
                targets = captions

                # Forward pass
                outputs = model(images, captions=None, is_training=False, max_seq_length=targets.size(1))  # No teacher forcing during validation
                
                outputs = outputs.view(-1, vocab_size)

                targets = targets.contiguous().view(-1)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Accuracy Calculation
                predictions = outputs.argmax(dim=1)
                mask = (targets != 0)  # Mask padding tokens
                val_correct += ((predictions == targets) & mask).sum().item()
                val_total_words += mask.sum().item()

                # BLEU Data Preparation
                predicted_sequences = predictions.view(len(lengths), -1)
                for j in range(len(lengths)):
                    ref = [captions[j, 1:lengths[j] + 1].tolist()]
                    hyp = predicted_sequences[j, :lengths[j]].tolist()
                    val_references.append(ref)
                    val_hypotheses.append(hyp)

                val_pbar.set_postfix(loss=val_loss / (i + 1), acc=val_correct / val_total_words)

        # Process references and hypotheses for BLEU calculation
        processed_val_references, processed_val_hypotheses = process_references_hypotheses(val_references, val_hypotheses, dataset)

        val_bleu1 = bleu.compute(predictions=processed_val_hypotheses, references=processed_val_references, max_order=1)
        val_bleu1 = val_bleu1['bleu']

        val_bleu2 = bleu.compute(predictions=processed_val_hypotheses, references=processed_val_references, max_order=2)
        val_bleu2 = val_bleu2['bleu']

        val_rouge_l = rouge.compute(predictions=processed_val_hypotheses, references=processed_val_references)
        val_rouge_l = val_rouge_l['rougeL']

        val_meteor_metric = meteor.compute(predictions=processed_val_hypotheses, references=processed_val_references)
        val_meteor_metric = val_meteor_metric['meteor']

        cumulative_val_bleu1 += val_bleu1
        cumulative_val_bleu2 += val_bleu2
        cumulative_val_rouge_l += val_rouge_l
        cumulative_val_meteor += val_meteor_metric

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss / len(train_dataloader):.4f}, "
            f"Train Accuracy: {train_correct / train_total_words:.4f}, Train BLEU: {train_bleu1:.4f}, "
            f"Val Loss: {val_loss / len(val_dataloader):.4f}, Val Accuracy: {val_correct / val_total_words:.4f}, "
            f"Val BLEU: {val_bleu1:.4f}")


        # Early Stopping Check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(folder_path, 'best_model.pth'))  # Save the best model
            print(f"Validation loss improved. Model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break


    avg_train_bleu1 = cumulative_train_bleu1 / num_epochs
    avg_train_bleu2 = cumulative_train_bleu2 / num_epochs
    avg_train_rouge_l = cumulative_train_rouge_l / num_epochs
    avg_train_meteor = cumulative_train_meteor / num_epochs

    avg_val_bleu1 = cumulative_val_bleu1 / num_epochs
    avg_val_bleu2 = cumulative_val_bleu2 / num_epochs
    avg_val_rouge_l = cumulative_val_rouge_l / num_epochs
    avg_val_meteor = cumulative_val_meteor / num_epochs  

    plot_losses(train_losses, val_losses, train_losses_plot_path)
    plot_metrics_and_save([avg_train_bleu1, avg_train_bleu2, avg_train_rouge_l, avg_train_meteor],
                            train_metrics_plot_path, "Average Training Metrics")
    plot_metrics_and_save(
        [avg_val_bleu1, avg_val_bleu2, avg_val_rouge_l, avg_val_meteor],
        val_metrics_plot_path, "Average Validation Metrics")


def test_image_captioning_model(model, test_dataloader, criterion, device, folder_path, dataset):
    model.to(device)
    model.eval()  # Set the model to evaluation mode (disables dropout, batchnorm, etc.)

    output_file = os.path.join(folder_path, "test_output.txt")
    test_metrics_plot_path = os.path.join(folder_path, "test_metrics_plot.png")

    test_loss = 0
    test_correct = 0
    test_total_words = 0
    test_references = []
    test_hypotheses = []

    with torch.no_grad():  # No need to compute gradients during testing
        test_pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing")
        for i, (images, captions, lengths) in test_pbar:
            images, captions = images.to(device), captions.to(device)

            # Prepare the targets (captions)
            lengths = [l - 1 for l in lengths]  # Adjust lengths for targets

            input_captions = captions
            targets = captions

            # Forward pass (no backpropagation)
            outputs = model(images, input_captions, is_training=False)
            vocab_size = len(dataset.idx2word)
            outputs = outputs.view(-1, vocab_size)

            # Reshape outputs and targets for loss calculation
            targets = targets.contiguous().view(-1)  # Flatten the target sequence
            loss = criterion(outputs, targets)  # Calculate loss
            test_loss += loss.item()

            # Accuracy Calculation
            predictions = outputs.argmax(dim=1)
            mask = (targets != 0)  # Mask padding tokens
            test_correct += ((predictions == targets) & mask).sum().item()
            test_total_words += mask.sum().item()

            # BLEU Data Preparation
            predicted_sequences = predictions.view(len(lengths), -1)
            for j in range(len(lengths)):
                ref = [captions[j, 1:lengths[j] + 1].tolist()]  # Ground truth (skip <SOS>)
                hyp = predicted_sequences[j, :lengths[j]].tolist()
                test_references.append(ref)
                test_hypotheses.append(hyp)

            test_pbar.set_postfix(loss=test_loss / (i + 1), acc=test_correct / test_total_words)

        # Process references and hypotheses for BLEU calculation
        processed_test_references, processed_test_hypotheses = process_references_hypotheses(test_references, test_hypotheses, dataset)

        test_bleu1 = bleu.compute(predictions=processed_test_hypotheses, references=processed_test_references, max_order=1)
        test_bleu1 = test_bleu1['bleu']

        test_bleu2 = bleu.compute(predictions=processed_test_hypotheses, references=processed_test_references, max_order=2)
        test_bleu2 = test_bleu2['bleu']

        test_rouge_l = rouge.compute(predictions=processed_test_hypotheses, references=processed_test_references)
        test_rouge_l = test_rouge_l['rougeL']

        test_meteor_metric = meteor.compute(predictions=processed_test_hypotheses, references=processed_test_references)
        test_meteor_metric = test_meteor_metric['meteor']

        avg_test_loss = test_loss / len(test_dataloader)

        # Write results to the output file
        with open(output_file, "a") as output_f:  # Append to the output file
            output_f.write(f"TESTING RESULTS\n")
            for ref, hyp in zip(processed_test_references, processed_test_hypotheses):
                output_f.write(f"TEST REFERENCE: {ref}\n")
                output_f.write(f"TEST HYPOTHESIS: {hyp}\n\n")

            output_f.write("-" * 40 + "\n")  # Separator for testing results

        print(f"Test Loss: {avg_test_loss:.4f}, "
            f"Test Accuracy: {test_correct / test_total_words:.4f}, "
            f"Test BLEU 1: {test_bleu1:.4f}",
            f"Test BLEU 2: {test_bleu2:.4f}",
            f"Test ROUGE L: {test_rouge_l:.4f}",
            f"Test METEOR: {test_meteor_metric:.4f}")
        
        plot_metrics_and_save([test_bleu1, test_bleu2, test_rouge_l, test_meteor_metric], 
                              test_metrics_plot_path, "Test Metrics")


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color='blue', marker='o')
    plt.plot(val_losses, label="Validation Loss", color='orange', marker='o')
    plt.title("Training and Validation Loss Over Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(save_path)  # Save plot as image
    plt.show()


def plot_metrics_and_save(metrics, plot_path, title):
    metric_names = ['BLEU-1', 'BLEU-2', 'ROUGE-L', 'METEOR']
    plt.figure(figsize=(10, 6))
    plt.bar(metric_names, metrics, color=['blue', 'orange', 'green', 'red'])
    plt.title(title)
    plt.ylabel('Metric Value')
    plt.savefig(plot_path)
    plt.close()


def process_references_hypotheses(references, hypotheses, dataset):
    processed_references = []
    processed_hypotheses = []
    pad_token = 0
    sos_token = 1
    eos_token = 2  # Assuming <EOS> has index 2; update based on your dataset

    for ref, hyp in zip(references, hypotheses):
        ref_caption = ref[0]
        processed_ref_caption = []
        processed_hyp_caption = []

        for idx, token in enumerate(ref_caption):
            if token != pad_token and token != eos_token and token != sos_token:
                processed_ref_caption.append(dataset.idx2word.get(token, "<UNK>"))
                processed_hyp_caption.append(dataset.idx2word.get(hyp[idx], "<UNK>"))

        processed_references.append(" ".join(processed_ref_caption))
        processed_hypotheses.append(" ".join(processed_hyp_caption))

    return processed_references, processed_hypotheses