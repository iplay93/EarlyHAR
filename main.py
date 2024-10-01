import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import seaborn as sns

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score,f1_score,recall_score

import argparse, os, utils
from data_preprocessing.dataloader import loading_data, count_label_labellist
import numpy as np
from numpy import arange
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib.pylab import plt

from tsaug import *

global_entropy_values = []
global_test_acc_values = []
global_high_values = []
global_diff_values = []

# Contrastive Learning을 위한 NT-Xent Loss 정의

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        positive_sim = torch.exp(torch.sum(z_i * z_j, dim=1) / self.temperature)
        
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.exp(torch.mm(z, z.t()) / self.temperature)
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, 0)
        
        positive_sum = positive_sim / (sim_matrix.sum(dim=1)[:batch_size] + sim_matrix.sum(dim=1)[batch_size:])
        loss = -torch.log(positive_sum).mean()
        
        return loss

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        contrast_count = features.shape[0]
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - torch.eye(contrast_count).to(features.device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive samples
        mask = mask / mask.sum(1, keepdim=True)
        loss = - (mask * log_prob).sum(1)
        loss = loss.mean()

        return loss

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, src):
        # Self-attention with attention weights returned
        attn_output, attn_weights = self.self_attn(src, src, src)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        
        # Feedforward network
        output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(output)
        src = self.norm2(src)

        return src, attn_weights  # Return both output and attention weights

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, label_list):
        super(Load_Dataset, self).__init__()
        X_train = data_list
        y_train = label_list

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        
        #X_train = X_train.permute(0, 2, 1)
        # (N, C, T)
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # print(np.array(pe[:, 0, 0::2]).shape, torch.sin(position * div_term).shape)
        # print(np.array(pe[:, 0, 1::2]).shape, torch.cos(position * div_term).shape)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model %2 == 1:
            pe[:, 0, 1::2] = torch.cos(position * div_term[0:len(div_term)-1])
        else:   
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        #print(x.shape)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Save the plots
def save_attention_plots(seq_attn, var_attn):
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    # Plot seq-seq attention
    sns.heatmap(torch.mean(seq_attn, dim=0).detach().cpu().numpy(), ax=ax[0],  xticklabels=range(1, seq_attn.shape[1]+1), yticklabels=range(1, seq_attn.shape[1]+1), cbar=True, square=True)
    ax[0].set_title('[seq-seq] Attention Map')

    # Plot var-var attention
    sns.heatmap(torch.mean(var_attn, dim=0).detach().cpu().numpy(), ax=ax[1],  xticklabels=range(1, var_attn.shape[1]+1), yticklabels=range(1, var_attn.shape[1]+1), cbar=True, square=True)
    ax[1].set_title('[var-var] Attention Map')
    
    print(seq_attn.shape, seq_attn.T.shape)
    is_symmetric = torch.allclose(torch.mean(seq_attn, dim=0), torch.mean(seq_attn, dim=0).T, atol=1e-6) 
    is_symmetric2 = torch.allclose(torch.mean(var_attn, dim=0), torch.mean(var_attn, dim=0).T, atol=1e-6)
    print("symmetric", is_symmetric, is_symmetric2)
    
    plt.tight_layout()
    # Save the figure
    file_path = 'attention_maps.png'
    plt.savefig(file_path)
    plt.close()
    return file_path


class Encoder(nn.Module):
    def __init__(self, seq, sensor_n, class_n):
        super(Encoder, self).__init__()
        d_model = 64
        self.value_embedding = nn.Linear(seq, d_model)
        self.temp_embedding = nn.Linear(sensor_n, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=seq)

        # Use the custom encoder layer
        encoder_layers = CustomTransformerEncoderLayer(d_model=d_model, nhead=1)
        self.transformer_encoder = nn.ModuleList([encoder_layers for _ in range(2)])

        self.projector = nn.Sequential(
            nn.Linear(seq * d_model, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.pos_encoder_inter = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=sensor_n)

        encoder_layers_inter = CustomTransformerEncoderLayer(d_model=d_model, nhead=1)
        self.transformer_encoder_inter = nn.ModuleList([encoder_layers_inter for _ in range(2)])

        self.projector_inter = nn.Sequential(
            nn.Linear(d_model * sensor_n, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )    

        self.linear = nn.Linear(64*2, 32)
        self.logits_simple = nn.Linear(32, class_n)        
        self.softmax = nn.Softmax(dim=1)

        # for encoder weighting
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5], requires_grad=True))

    def forward(self, x):
        # Apply value and temporal embedding
        x_val = self.value_embedding(x.permute(0,2,1))
        x_temp = self.temp_embedding(x)
        
        # Get embeddings
        #x_val = self.pos_encoder(x_val)
        #x_temp = self.pos_encoder(x_temp)

        # Transformer Encoder: Get attention weights from each layer
        attn_maps = []
        for layer in self.transformer_encoder:
            x_temp, attn_weights = layer(x_temp)
            attn_maps.append(attn_weights)
        
        # Projection layer
        z_time = self.projector(x_temp.view(x_temp.size(0), -1))
        
        # Inter Transformer
        for layer_inter in self.transformer_encoder_inter:
            x_val, attn_weights_inter = layer_inter(x_val)
            attn_maps.append(attn_weights_inter)

        # Projector inter
        z_inter = self.projector_inter(x_val.view(x_val.size(0), -1))

        weights_norm = F.softmax(self.weights, dim=0)
        
        z_time_weighted = weights_norm[0] * z_time
        z_inter_weighted = weights_norm[1] * z_inter

        # 텐서를 두 번째 차원에서 병합
        fused_output = torch.cat((z_time_weighted, z_inter_weighted), dim=1)

        transformed_output = self.linear(fused_output)
        #emb = torch.sigmoid(transformed_output)
        pred = self.logits_simple(transformed_output)

        s_t = self.softmax(pred) 

        return fused_output, fused_output, pred, s_t, attn_maps

        # # Concatenate and pass through final linear layer
        # combined = torch.cat([x_temp, x_val], dim=-1)
        # output = self.linear(combined)
        
        # return output, attn_maps  # Return both output and attention maps

# class Encoder(nn.Module):
#     def __init__(self, seq, sensor_n, class_n):
#         super(Encoder, self).__init__()
#         d_model = 64
#         self.value_embedding = nn.Linear(sensor_n, d_model)
#         self.temp_embedding = nn.Linear(seq, d_model)
        
#         self.pos_encoder = PositionalEncoding(d_model = d_model, dropout = 0.1, max_len = seq)

#         encoder_layers = TransformerEncoderLayer(d_model= d_model, nhead=1, batch_first=True)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers= 2)

#         self.projector = nn.Sequential(
#             nn.Linear(seq * d_model, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 64)
#         )

#         self.pos_encoder_inter = PositionalEncoding(d_model = d_model, dropout = 0.1, max_len = sensor_n)

#         encoder_layers_inter = TransformerEncoderLayer(d_model= d_model, nhead=1, batch_first=True)
#         self.transformer_encoder_inter = TransformerEncoder(encoder_layers_inter, num_layers= 2)

#         self.projector_inter = nn.Sequential(
#             nn.Linear(d_model * sensor_n, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 64)
#         )    

#         self.linear = nn.Linear(64*2, 32)
#         #self.linear = nn.Linear(64, 32)
#         self.logits_simple = nn.Linear(32, class_n)        
#         self.softmax = nn.Softmax(dim=1)

#         # for encoder weighting
#         self.weights = nn.Parameter(torch.tensor([0.5, 0.5], requires_grad=True))


#     def forward(self, x):        

#         x_inter = x.permute(0,2,1)

#         #x = self.pos_encoder(x.permute(1,0,2)).permute(1,0,2)
#         x = self.value_embedding(x)
#         x = self.transformer_encoder(x)
#         h_time = x.reshape(x.shape[0], -1)
#         z_time = self.projector(h_time)

#         #x_inter = self.pos_encoder_inter(x_inter.permute(1,0,2)).permute(1,0,2)
#         x_inter = self.temp_embedding(x_inter)
#         x_inter = self.transformer_encoder_inter(x_inter)
#         h_inter = x_inter.reshape(x_inter.shape[0], -1)
#         z_inter = self.projector_inter(h_inter)

#         weights_norm = F.softmax(self.weights, dim=0)
#             # 가중치를 적용한 텐서
#         z_time_weighted = weights_norm[0] * z_time
#         z_inter_weighted = weights_norm[1] * z_inter

#         # 텐서를 두 번째 차원에서 병합
#         fused_output = torch.cat((z_time_weighted, z_inter_weighted), dim=1)
#         #fused_output = weights_norm[0] * z_time + weights_norm[1] * z_inter
        

#         z_concatenated = torch.cat((z_time, z_inter), dim=-1)
#         h_concatenated = torch.cat((h_time, h_inter), dim=-1)
        
#         transformed_output = self.linear(fused_output)
#         #emb = torch.sigmoid(transformed_output)
#         pred = self.logits_simple(transformed_output)

#         s_t = self.softmax(pred) 

#         return h_concatenated, fused_output, pred, s_t

    
def arg_segment():
    parser = argparse.ArgumentParser()
    ######################## Model parameters ########################

    home_dir = os.getcwd()
    parser.add_argument('--experiment_description', default='exp1', type=str,
                        help='Experiment Description')
    parser.add_argument('--run_description', default='run1', type=str,
                        help='Experiment Description')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed value')
    parser.add_argument('--dataset', default='doore', type=str,
                        help='Dataset of choice: doore, casas, opportunity, aras_a')
    parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                        help='saving directory')
    parser.add_argument('--device', default='cuda', type=str,
                        help='cpu or cuda')
    parser.add_argument("--model_save_path", type=str, default="./saved_models/", help="Where to save the model once it is trained.")
    parser.add_argument('--home_path', default=home_dir, type=str,
                        help='Project home directory')

    parser.add_argument('--padding', type=str, 
                        default='mean', help='choose one of them : no, max, mean')
    parser.add_argument('--timespan', type=int, 
                        default=10000, help='choose of the number of timespan between data points (1000 = 1sec, 60000 = 1min)')
    parser.add_argument('--min_seq', type=int, 
                        default=10, help='choose of the minimum number of data points in a example')
    parser.add_argument('--min_samples', type=int, default=20, 
                        help='choose of the minimum number of samples in each label')
    parser.add_argument('--one_class_idx', type=int, default=0, 
                        help='choose of one class label number that wants to deal with. -1 is for multi-classification')
        
    parser.add_argument('--aug_method', type=str, default='AddNoise', help='choose the data augmentation method')
    parser.add_argument('--aug_wise', type=str, default='Temporal', 
                            help='choose the data augmentation wise : "None,  Temporal, Sensor" ')

    parser.add_argument('--test_ratio', type=float, default=0.2, help='choose the number of test ratio')
    parser.add_argument('--overlapped_ratio', type=int, default= 50, help='choose the number of windows''overlapped ratio')
    parser.add_argument('--lam_a', type=float, default= 1, help='choose lam_a ratio')
    parser.add_argument('--train_num_ratio', type=float, default = 1, help='choose the number of test ratio')
    parser.add_argument('--lam_score', type=float, default = 1, help='choose the number of test ratio')
    parser.add_argument('--training_ver', type = str, default = 'Diverse', help='choose one of them: One, Diverse, Random')
   # parser.add_argument('--batch', type=int, default= 32, help='batch_size')
    parser.add_argument('--neg_ths', type=float, default= 0.9, help='choose neg_thrshold ratio')

    # for training   
    parser.add_argument('--loss', type=str, default='SupCon', help='choose one of them: crossentropy loss, contrastive loss')
    parser.add_argument('--optimizer', type=str, default='', help='choose one of them: adam')
    #parser.add_argument('--lr', type=float, default=3e-5, help='choose the number of learning rate')
    parser.add_argument('--temp', type=float, default=0.7, help='temperature for loss function')
    parser.add_argument('--warm', action='store_true',help='warm-up for large batch training')
    parser.add_argument("--nepochs", type=int, default=40, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    
    args = parser.parse_args()

    return args

def plot_loss(train_loss, train_acc, val_loss, val_acc, args):

    train_values = train_loss.values()
    train_accur = train_acc.values()

    val_values = val_loss.values()
    val_accur = val_acc.values()

    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, args.nepochs+1)
    
    # Plot and label the training and validation loss values
    plt.plot(epochs, train_values, label='Training Loss')
    plt.plot(epochs, val_values, label='Validation Loss')
    
    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Set the tick locations
    plt.xticks(arange(0, args.nepochs+1, 2))
    
    # Display the plot
    plt.legend(loc='best')
    #plt.show()
    plt.savefig('loss.jpg')
    plt.close()  

    #accuracy
    plt.plot(epochs, train_accur, label='Training Acc')
    plt.plot(epochs, val_accur, label='Validation Acc')
    # Add in a title and axes labels
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set the tick locations
    plt.xticks(arange(0, args.nepochs+1, 2))    
    # Display the plot
    plt.legend(loc='best')
    #plt.show()
    plt.savefig('acc.jpg')
    plt.close()  

def Trainer(model, model_optimizer, train_dl, val_dl, device, args):
    criterion = nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    train_loss_dict, train_acc_dict = {},{}
    val_loss_dict, val_acc_dict = {},{}

    for epoch in range(1, args.nepochs+1):        
        ave_loss, ave_acc, ave_val_loss, ave_val_acc, entropy = model_train(model, model_optimizer, criterion, train_dl, val_dl, args, device)
        
        train_loss_dict[epoch] = ave_loss
        train_acc_dict[epoch] = ave_acc
        val_loss_dict[epoch] = ave_val_loss
        val_acc_dict[epoch] = ave_val_acc
    
    # Plot losses
    plot_loss(train_loss_dict, train_acc_dict, val_loss_dict, val_acc_dict, args)

    # Save the model
    os.makedirs(os.path.join(args.experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict()}
    torch.save(chkpoint, os.path.join(args.experiment_log_dir, "saved_models", f'ckp_last.pt'))

    print("\n################## Training is Done! #########################")

    return entropy


def model_train(model, model_optimizer, criterion, train_dl, val_dl, args, device):
    model.train()
    
    total_loss = []
    total_acc = []
    total_auc = []  # it should be outside of the loop
    total_f1 = []

    outs = np.array([])
    trgs = np.array([])
    contrastive_loss_fn = SupervisedContrastiveLoss()
    contrastive_loss = NTXentLoss()
    my_aug = (AddNoise(scale=0.1))

    penalty_factor = 1.0
    for (data, labels) in train_dl:
        data, labels = data.float().to(device), labels.long().to(device)

        data_a = torch.from_numpy(my_aug.augment(data.cpu().numpy())).to(device)
        #x1, x2 = x, x_a  # 실제로는 augmentation 적용

        model_optimizer.zero_grad()

        h, z, pred, s, _ = model(data)
        h_a, z_a, pred_a, s_a, _ = model(data_a)

        pred, pred_a = F.normalize(pred, dim=1), F.normalize(pred_a, dim=1)

        loss1 = criterion(pred, labels)
        loss2 = criterion(pred_a, labels)
        loss3 = contrastive_loss_fn(torch.cat([pred, pred_a], dim=0), torch.cat([labels, labels], dim=0))
        #loss3 = contrastive_loss(z, z_a)
        lambda_loss = 0.1
        loss = (1-lambda_loss)*(loss1 + loss2) + lambda_loss*(loss3/2)
        #print("loss: {:.2f}, loss1: {:.2f}, loss2: {:.2f}, loss3: {:.2f}".format(loss.item(), loss1.item(), loss2.item(), loss3.item()))
        # _, predictions = torch.max(pred, 1)
        # incorrect_predictions_mask = predictions != labels

        # if incorrect_predictions_mask.any():
        #     incorrect_probs = F.softmax(pred, dim=1)[incorrect_predictions_mask, :]
        #     incorrect_labels = labels[incorrect_predictions_mask]
        #     # 잘못된 예측에 대한 로그 확률 계산
        #     incorrect_log_probs = incorrect_probs.gather(1, incorrect_labels.unsqueeze(1))
        #     # 잘못된 예측에 대한 크로스 엔트로피 손실 계산
        #     incorrect_ce_loss = -incorrect_log_probs.mean()
        #     # 패널티 적용
        #     penalty = incorrect_ce_loss * penalty_factor
        # else:
        #     penalty = 0

        # loss = loss + penalty

        loss.backward()
        model_optimizer.step()

        acc_bs = labels.eq(pred.detach().argmax(dim=1)).float().mean()
        onehot_label = F.one_hot(labels)
        pred_numpy = pred.detach().cpu().numpy()

        try:
            auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr" )
        except:
            auc_bs = float(0)
        f1_bs = f1_score(labels.detach().cpu().numpy(), np.argmax(pred_numpy, axis=1), average='macro',)

        total_acc.append(acc_bs)
        total_auc.append(auc_bs)
        total_f1.append(f1_bs)
        total_loss.append(loss.item())

        pred = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
        outs = np.append(outs, pred.cpu().numpy())
        trgs = np.append(trgs, labels.data.cpu().numpy())

    precision = precision_score(trgs, outs, average='macro', )
    recall = recall_score(trgs, outs, average='macro', )
    F1 = f1_score(trgs, outs, average='macro', )
    
    ave_loss = torch.tensor(total_loss).mean()
    ave_acc = torch.tensor(total_acc).mean()
    ave_auc = torch.tensor(total_auc).mean()
    ave_f1 = torch.tensor(total_f1).mean()

    print(' Train: loss = %.4f| Acc=%.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f| AUROC=%.4f | F1 = %.4f'
          % (ave_loss, ave_acc*100, precision * 100, recall * 100, F1 * 100, ave_auc * 100, ave_f1 *100))
    
    # all_features, all_labels = [], []
    # for data, label in train_dl:
    #     all_features.append(data)
    #     all_labels.append(label)

    # all_features = torch.cat(all_features, dim=0)

    # h, z, pred, s = model(all_features.float().to(device))

    # entropy = -(s * torch.log(s + 1e-9)).sum(dim=1).mean()
    
    # #average_entropy = entropy.mean()

    # print(f"Batch average entropy: {entropy.item(),  math.log(args.n_class)}")
    

    val_loss = []
    val_acc = []

    for (data, labels) in val_dl:
        data, labels = data.float().to(device), labels.long().to(device) 

        h, z, pred, s, _ = model(data)

        loss = criterion(pred, labels)
        acc_bs = labels.eq(pred.detach().argmax(dim=1)).float().mean()

        val_loss.append(loss.item())
        val_acc.append(acc_bs)
    
    ave_val_loss = torch.tensor(val_loss).mean()
    ave_val_acc = torch.tensor(val_acc).mean()

    return ave_loss, ave_acc, ave_val_loss, ave_val_acc, 0 #entropy.item()


def plot_entropy_over_time(entropies):
    plt.plot(range(1, len(entropies) + 1), entropies)
    plt.xlabel('Timestep')
    plt.ylabel('Average Entropy')
    plt.title('Entropy Change over Time')
    plt.show()
    file_path = 'plot_entropy_over_time.png'
    plt.savefig(file_path)

def model_entropy_change(model, test_dl, device):
    model.eval()

    entropies = []  # 각 타임스텝별 평균 엔트로피 저장용

    for (data, labels) in test_dl:
        data, labels = data.float().to(device), labels.long().to(device) 
        batch_size, seq_len, num_features = data.size()  # 들어오는 데이터는 (batch_size, num_features, t) 형태
        print('seq_len', seq_len)

        # 배치마다 각 샘플별 엔트로피 저장용 리스트 초기화
        batch_entropies = [[] for _ in range(batch_size)]  # 각 샘플에 대해 엔트로피를 저장


        # 각 타임스텝마다 entropy 계산
        for t in range(1, seq_len + 1):
            # 현재 t까지의 데이터 슬라이스 (batch_size, t, num_features)
            input_at_t = data[:, :t, :]

            # 나머지 부분을 0으로 패딩 (seq_len - t만큼)
            padded_input = F.pad(input_at_t, (0, 0, 0, seq_len - t))  # (batch_size, seq_len, num_features)

            # 모델에 입력하고 출력 계산
            h, z, pred, s, _ = model(padded_input)  # Transformer encoder의 출력

            # 확률 계산 (softmax)
            probabilities = F.softmax(pred, dim=-1)

            # 엔트로피 계산
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)  # 엔트로피 계산

            # 배치 내 평균 엔트로피 계산
            avg_entropy = torch.mean(entropy).item()
            batch_entropies.append(avg_entropy)

        entropies.append(batch_entropies)

    # Entropy 변화 시각화 (각 타임스텝에 대한 평균 엔트로피)
    entropies = torch.tensor(entropies).mean(dim=0).tolist()  # 타임스텝별 평균

    plot_entropy_over_time(entropies)

def model_evaluate(model, test_dl, device):
    model.eval()

    total_loss = []
    total_acc = []
    total_auc = []  # it should be outside of the loop
    total_f1 = []
    total_entropy =[]
    total_high = []
    total_diff = []

    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for (data, labels) in test_dl:
            data, labels = data.float().to(device), labels.long().to(device) 

            h, z, pred, s, _ = model(data)

            acc_bs = labels.eq(pred.detach().argmax(dim=1)).float().mean()
            onehot_label = F.one_hot(labels)
            pred_numpy = pred.detach().cpu().numpy()

            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr" )
            except:
                auc_bs = np.float(0)
            f1_bs = f1_score(labels.detach().cpu().numpy(), np.argmax(pred_numpy, axis=1), average='macro',)
        

            entropy = -(s * torch.log(s + 1e-9)).sum(dim=1).mean()
            
            max_values, _ = torch.max(s, dim=1)
            mean_of_max_values = max_values.mean()

            # 각 행에서 상위 2개의 최대값 추출
            top2_values, _ = torch.topk(s, 2, dim=1)

            # 최대값과 두 번째 최대값의 차이 계산
            differences = top2_values[:, 0] - top2_values[:, 1]

            # 차이들의 평균 계산
            mean_difference = differences.mean()


            total_acc.append(acc_bs)
            total_auc.append(auc_bs)
            total_f1.append(f1_bs)
            total_entropy.append(entropy)
            total_high.append(mean_of_max_values)
            total_diff.append(mean_difference)

            pred = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())            

        print(s[0], max(s[0]), top2_values[:, 0], top2_values[:, 1])
        precision = precision_score(trgs, outs, average='macro', )
        recall = recall_score(trgs, outs, average='macro', )
        F1 = f1_score(trgs, outs, average='macro', )
        
        ave_acc = torch.tensor(total_acc).mean()
        ave_auc = torch.tensor(total_auc).mean()
        ave_f1 = torch.tensor(total_f1).mean()
        print(len(total_entropy))
        if len(total_entropy) != 1:
            average_entropy =  torch.tensor(total_entropy).mean()
        else:
            average_entropy =  torch.tensor(total_entropy)

        high = torch.tensor(total_high).mean()
        diff = torch.tensor(total_diff).mean()
        print(' Test: Acc=%.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f| AUROC=%.4f | F1 = %.4f'
          % (ave_acc*100, precision * 100, recall * 100, F1 * 100, ave_auc * 100, ave_f1 *100))

        return [ave_acc.item(), precision, recall, F1, ave_auc.item(), ave_f1.item(), average_entropy.item(), high.item(), diff.item()]

if __name__ == "__main__":

    args = arg_segment()

    data_type = args.dataset
    device = torch.device(args.device)
    store_path = 'results/AttnStop_'+args.dataset+'_entropy_test.xlsx'

    print("Data loaded ...")
    # setting for each dataset       
    if args.dataset == 'doore': 
        args.padding, args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise = 'mean', 10000, 10, 20, 'AddNoise', 'Temporal'
        
    elif args.dataset == 'casas': 
        args.padding, args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise = 'mean', 10000, 10, 20, 'AddNoise', 'Temporal2'

    elif args.dataset == 'usc-had': 
        args.padding, args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise = 'mean', 10, 0, 0, 'AddNoise', 'None'

    elif args.dataset == 'openpack': 
        args.padding, args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise = 'mean', 100, 10, 20, 'AddNoise', 'None'
    
    elif args.dataset == 'opportunity': 
        args.padding, args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise = 'mean', 1000, 10, 20, 'AddNoise', 'Temporal'
    
    elif args.dataset == 'aras': 
        args.padding, args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise = 'mean', 1000, 10, 20, 'AddNoise', 'Temporal'
        
    num_classes, datalist, labellist, length_list = loading_data(args.dataset, args)

    
    args.n_class = len(num_classes)
    data_size, timestamps, feature_num = datalist.shape

    # mask 초기화 (batch_size, max_len) 형태로 패딩 마스크 생성, 모두 1로 초기화 (참조할 수 있음)
    padding_mask = torch.ones(data_size, max(length_list))

    # 각 시퀀스의 실제 길이 이후 부분은 패딩되므로 마스킹 (0으로 설정)
    for i, length in enumerate(length_list):
        padding_mask[i, length:] = 0  # 해당 시퀀스의 실제 길이 이후는 마스킹 처리
    padding_mask = padding_mask.bool()

    final_rs = []

    for SEED in [20]:

        ###### fix random seeds for reproducibility ########
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        #####################################################
        
        for earliness in [1]:        
            timestamp = int(timestamps * earliness)

            args.experiment_log_dir = os.path.join(args.logs_save_dir, data_type, f"timestamp_{timestamp}_seed_{SEED}")
            os.makedirs(args.experiment_log_dir, exist_ok=True)

            datalist_split = datalist[:, :(timestamp+1), :]
            print(timestamp, "dataslist shape", datalist_split.shape)

            # train/val/test data extraction
            train_list, test_list, train_label_list, test_label_list = train_test_split(datalist_split, 
                labellist, test_size = args.test_ratio, stratify=labellist, random_state=SEED) 
            
            train_list, val_list, train_label_list, val_label_list = train_test_split(train_list, 
                train_label_list, test_size = args.test_ratio, stratify=train_label_list, random_state=SEED)

            train_list = torch.tensor(train_list).cuda().cpu()
            train_label_list = torch.tensor(train_label_list).cuda().cpu()
            val_list = torch.tensor(val_list).cuda().cpu()
            val_label_list = torch.tensor(val_label_list).cuda().cpu()
            test_list = torch.tensor(test_list).cuda().cpu()
            test_label_list = torch.tensor(test_label_list).cuda().cpu()

            print(f"Train Data: {len(train_list)} --------------")
            exist_labels, _ = count_label_labellist(train_label_list)

            print(f"Validation Data: {len(val_list)} --------------")
            count_label_labellist(val_label_list)

            print(f"Test Data: {len(test_list)} --------------")
            count_label_labellist(test_label_list) 

            # Build data loader
            dataset = Load_Dataset(train_list, train_label_list)
            train_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            dataset = Load_Dataset(val_list, val_label_list)
            val_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            dataset = Load_Dataset(test_list, test_label_list)
            test_dl = DataLoader(dataset, batch_size= args.batch_size, shuffle=True)       

            model = Encoder(seq = train_list.shape[1], sensor_n = train_list.shape[2], class_n = len(num_classes)).to(device)
            model_optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=3e-4)
            
            # Trainer
            training_en = Trainer(model, model_optimizer, train_dl, val_dl, device, args)

            outputs = model_evaluate(model, test_dl, device)
            model_entropy_change(model, test_dl, device)

            global_test_acc_values.append(outputs[0])
            global_entropy_values.append(outputs[6])
            global_high_values.append(outputs[7])
            global_diff_values.append(outputs[8])

            outputs.append(training_en)

            final_rs.append(outputs)
            print(outputs)

            _, _, _, _, attn_maps = model(train_list.float().to(device))
            # Save the plot and provide the file path
            
            file_path = save_attention_plots(attn_maps[1], attn_maps[3])
    
    df = pd.DataFrame(final_rs, columns=['ave_acc', 'precision', 'recall', 'F1', 'ave_auc', 'ave_f1', 'ave_entropy', 'high_entropy', 'diff_entropy', 'training_enp'], index= [int(timestamp * x)  for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]])
    #df = pd.DataFrame(outputs)
    df.to_excel(store_path, sheet_name='the results')


    # --- save model ---
    #torch.save(model.state_dict(), model_save_path+"_model.pt")
    print("entropy", global_entropy_values)
    print("acc", global_test_acc_values)
    print("high", global_high_values)
    print("diff", global_diff_values)

    torch.cuda.empty_cache()
