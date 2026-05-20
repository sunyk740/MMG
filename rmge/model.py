#!/usr/bin/env python3

# Robust MetaGene Extractor (RMGE)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

class RMGE:
    def __init__(self, sn_data, st_data, class_columns,device='cuda',normalize=True, dropout_rate=0.8, lambda_l2=0.01, 
                 kl_weight=1, adv_weight=0.1, re_weight=1, entropy_reg_weight=1):
        """
        Initialize the model for cross-modal learning between single-nucleus and spatial transcriptomics data.

        Parameters:
        - sn_data: Single-nucleus RNA-seq data (AnnData object)
        - st_data: Spatial transcriptomics data (AnnData object)
        - device: The device to run the model on ('cuda' or 'cpu')
        - dropout_rate: Dropout rate for regularization
        - lambda_l2: L2 regularization weight
        - kl_weight: Weight for KL divergence loss
        - adv_weight: Weight for adversarial loss
        - re_weight: Weight for reconstruction loss
        - entropy_reg_weight: Weight for entropy regularization loss
        """
        self.sn_data = sn_data
        self.st_data = st_data
        self.device = device
        self.class_columns = class_columns
        # Set parameters
        self.dropout_rate = dropout_rate
        self.lambda_l2 = lambda_l2
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight
        self.re_weight = re_weight
        self.entropy_reg_weight = entropy_reg_weight
        self.normalize = normalize
        
        # Prepare data
        self.prepare_data()

        # Initialize model and discriminator
        self.model = AE(num_gene=self.sn_data.X.shape[1], num_classes=self.num_classes,dropout_rate=self.dropout_rate).to(self.device)
        self.discriminator = Discriminator(input_dim=self.num_classes).to(self.device)

        # Loss functions and optimizers
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.disc_criterion = nn.BCELoss()
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.reconstruction_criterion = nn.MSELoss()

    def prepare_data(self):
        """Prepare data by handling zero columns, normalization, and splitting"""
        class_columns = self.class_columns
        if 'noise' in self.sn_data.layers:
            X_np = self.sn_data.layers['noise']
        else:
            X_np = self.sn_data.X

        # Encode labels
        y_np = self.sn_data.obs[class_columns].astype('category')
        self.class_names = y_np.cat.categories  # Get class names
        class_counts = y_np.value_counts()
        weights = 1.0 / class_counts
        weights = weights.sort_index()
        self.class_weights = torch.tensor(weights.values, dtype=torch.float)
        y_np = y_np.cat.codes.values  # Encode as integers
        self.num_classes = len(np.unique(y_np))

  
        cell_ratio = self.sn_data.obs[class_columns].value_counts(normalize=True)

        self.prior_probs = torch.tensor(cell_ratio.loc[self.class_names], dtype=torch.float32).to(self.device)

        # Split dataset into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
        self.X_st_tensor = torch.tensor(self.st_data.X.astype(np.float32)).to(self.device)

        # Calculate weight for samples
        self.num_cell = len(X_np)
        self.num_st_cell = len(self.X_st_tensor)
        total_samples = self.num_cell + self.num_st_cell
        self.weight_sn = total_samples / (2.0 * self.num_cell)
        self.weight_st = total_samples / (2.0 * self.num_st_cell)

        self.weight_sn = torch.tensor([self.weight_sn]).to(self.device)
        self.weight_st = torch.tensor([self.weight_st]).to(self.device)

    def train(self, epochs=1000):
        """Train the cross-modal model"""
        classification_losses = []
        reconstruction_losses = []
        kl_div_losses = []
        adv_losses = []
        entropy_losses = []
        l2_losses = []
        train_losses = []
        train_accuracies = []

        for epoch in range(epochs):
            self.model.train()

            # 1. Train discriminator
            self.optimizer_disc.zero_grad()

            features_sn, _ = self.model(self.X_train_tensor)
            features_st, _ = self.model(self.X_st_tensor)

            labels_sn = torch.ones(features_sn.size(0), 1).to(self.device)  # sub_cell labels
            labels_st = torch.zeros(features_st.size(0), 1).to(self.device)  # st_sub_cell labels

            outputs_sn = self.discriminator(features_sn, set_T=False)
            outputs_st = self.discriminator(features_st, set_T=True)

            disc_loss = self.disc_criterion(outputs_sn, labels_sn) * self.weight_sn + self.disc_criterion(outputs_st, labels_st) * self.weight_st
            disc_loss.backward()
            self.optimizer_disc.step()

            # 2. Train feature extractor (DNN), making it hard for the discriminator to distinguish modalities
            self.optimizer.zero_grad()
            z, y = self.model(self.X_train_tensor)

            # Classification loss
            classification_loss = self.criterion(z, self.y_train_tensor)
            reconstruction_loss = self.reconstruction_criterion(y, self.X_train_tensor)

            # L2 regularization
            l2_loss = 0
            for param in self.model.parameters():
                l2_loss += torch.sum(param ** 2)

            output_probs = torch.softmax(z, dim=1)
            expected_probs = output_probs.mean(dim=0)
            kl_div = self.kl_divergence_loss(expected_probs, self.prior_probs)

            features_st, _ = self.model(self.X_st_tensor)
            st_output_probs = torch.softmax(features_st, dim=1)
            st_expected_probs = st_output_probs.mean(dim=0)
            st_kl_div = self.kl_divergence_loss(st_expected_probs, self.prior_probs)

            outputs_sn = self.discriminator(z, set_T=False)
            outputs_st = self.discriminator(features_st, set_T=True)

            adv_loss = self.disc_criterion(outputs_sn, 0.5 * torch.ones_like(outputs_sn)) * self.weight_sn + \
                       self.disc_criterion(outputs_st, 0.5 * torch.ones_like(outputs_st)) * self.weight_st

            # Entropy regularization
            entropy_reg = self.entropy_loss(output_probs)

            # Total loss
            total_loss = classification_loss + self.lambda_l2 * l2_loss + \
                         self.kl_weight * st_kl_div + self.adv_weight * adv_loss + self.re_weight * reconstruction_loss + \
                         self.entropy_reg_weight * entropy_reg
            total_loss.backward()
            self.optimizer.step()

            train_losses.append(total_loss.item())

            # Calculate training accuracy
            self.model.eval()
            with torch.no_grad():
                train_outputs, _ = self.model(self.X_train_tensor)
                _, train_preds = torch.max(train_outputs, 1)
                train_corrects = (train_preds == self.y_train_tensor).sum().item()
                train_acc = train_corrects / self.y_train_tensor.size(0)
                train_accuracies.append(train_acc)

            # Record losses
            classification_losses.append(classification_loss.item())
            reconstruction_losses.append(reconstruction_loss.item())
            kl_div_losses.append(st_kl_div.item())
            adv_losses.append(adv_loss.item())
            entropy_losses.append(entropy_reg.item())
            l2_losses.append(l2_loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {total_loss.item()}, Accuracy: {train_acc}")

        return train_losses, train_accuracies

    def evaluate(self):
        """Evaluate the model on validation set and spatial transcriptomics"""
        self.model.eval()
        with torch.no_grad():
            # Validation
            val_outputs, _ = self.model(self.X_val_tensor)
            val_loss = self.criterion(val_outputs, self.y_val_tensor).item()
            _, val_preds = torch.max(val_outputs, 1)
            corrects = (val_preds == self.y_val_tensor).sum().item()
            val_acc = corrects / self.y_val_tensor.size(0)

            # Spatial transcriptomics prediction
            st_projection, _ = self.model(self.X_st_tensor)
            st_predicted_classes = torch.argmax(st_projection, dim=1).cpu().numpy()
            predict_class_names = self.class_names[st_predicted_classes]

            # true_classes = self.st_data.obs[self.class_columns].values
            # predict_acc = sum(predict_class_names == true_classes) / len(true_classes)

            print(f"Validation Accuracy: {val_acc:.4f}")
            # print(f"Prediction Accuracy: {predict_acc:.4f}")
        return predict_class_names

    def kl_divergence_loss(self, output_probs, prior_probs):
        """Compute the KL divergence loss"""
        log_output_probs = torch.log(output_probs + 1e-8)
        kl_div = torch.sum(output_probs * (log_output_probs - torch.log(prior_probs + 1e-8)))
        return kl_div

    def entropy_loss(self, probs):
        """Compute the entropy loss"""
        return -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))


class AE(nn.Module):
    """Autoencoder model for feature extraction"""
    def __init__(self,num_gene, num_classes,dropout_rate):
        super(AE, self).__init__()
        self.fc = nn.Linear(num_gene, num_classes, bias=False)
        self.decoder = nn.Linear(num_classes, num_gene, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        z = self.fc(x)
        y = self.decoder(z)
        return z, y


class Discriminator(nn.Module):
    """Discriminator for distinguishing between sub-cell and spatial transcriptomics data"""
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, x, set_T=False):
        if set_T:
            x = x / self.T
        x = torch.softmax(x, dim=1)
        return torch.sigmoid(self.fc(x))