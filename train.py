import re
import math
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchtext
import os
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel

from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from torchmetrics import Precision
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import wikipediaapi


def seed_all(seed=42):
    import torch, random, os, numpy
    
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()

def normalizeAdjacency(W):
    assert W.size(0) == W.size(1)
    d = torch.sum(W, dim = 1)
    d = 1/torch.sqrt(d)
    D = torch.diag(d)
    return D @ W @ D

def get_embedding_from_ggnews(text, ggnews):
    text = text.lower()
    text = text.replace('/', ',')
    try:
        if len(text.split(',')) == 1:
            return torch.tensor(ggnews[text]).detach().cpu()
        embed = []
        for t in text.split(','):
            embed.append(ggnews[t])
        embed = torch.stack(embed)
        return torch.mean(embed, dim=0).detach().cpu()
    except:
        return torch.rand((1, 300)).detach().cpu()

def get_embedding_from_glove(text):
    text = text.lower()
    text = text.replace('/', ',')
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    if len(text.split(',')) == 1:
        return glove[text]
    embed = []
    for t in text.split(','):
        embed.append(glove[t])
    embed = torch.stack(embed)
    return torch.mean(embed, dim=0)

def get_embedding_from_fasttext(text):
    text = text.lower()
    text = text.replace('/', ',')
    fasttext = torchtext.vocab.FastText('simple')
    try:
        if len(text.split(',')) == 1:
            return torch.Tensor(fasttext[text])
        embed = []
        for t in text.split(','):
            embed.append(fasttext[t])
        embed = torch.stack(embed)
        return torch.mean(embed, dim=0)
    except:
        return torch.rand((1, 300))
def get_paragraph_from_wiki(text, n_sent=2):
    text = text.lower()
    wiki_wiki = wikipediaapi.Wikipedia('en')
    text = text.replace('/', ',')
    text = text.split(',')
    paragr = []
    for t in text:
        page = wiki_wiki.page(t)
        paragraph = sent_tokenize(page.summary)
        if len(paragraph) == 0:
            paragr.extend(t)
        elif len(paragraph) <= n_sent:
            paragr.extend(paragraph)
        else:
            paragr.extend(paragraph[:n_sent])
    return ' '.join(paragr)

def get_embedding_from_wiki(text, n_sent=2):
    sbert = SentenceTransformer('paraphrase-distilroberta-base-v1', device='cpu')
    text = get_paragraph_from_wiki(text, n_sent)
    embedding = sbert.encode(text, convert_to_tensor=True)
    return embedding
    
def PMI_create_edges_and_features(train, mlb, embedding_type, threshold):
    label2id = {v: k for k, v in enumerate(mlb.classes_)}

    edges = torch.zeros((len(label2id), len(label2id)))
    for label in train["label"]:
        if len(label) >= 2:
            for i in range(len(label) - 1):
                for j in range(i + 1, len(label)):
                    src, tgt = label2id[label[i]], label2id[label[j]]
                    edges[src][tgt] += 1
                    edges[tgt][src] += 1
    
    marginal_edges = torch.zeros((len(label2id)))
    for label in train["label"]:
        for i in range(len(label)):
            marginal_edges[label2id[label[i]]] += 1
    
    for i in range(edges.size(0)):
        for j in range(edges.size(1)):
            if edges[i][j] != 0:
                edges[i][j] = (edges[i][j] * len(train["label"]))/(marginal_edges[i] * marginal_edges[j])

    edges = normalizeAdjacency(edges + torch.diag(torch.ones(len(label2id))))

    if embedding_type=='random':
        features = torch.rand(101, 768)
    elif embedding_type=='wiki':
        features = torch.zeros(len((label2id)), 768)
        for label, id in tqdm(label2id.items()):
            features[id] = get_embedding_from_wiki(label, n_sent=2)
    else:
        features = torch.zeros(len((label2id)), 300)
        if embedding_type=='fasttext':
            for label, id in tqdm(label2id.items()):
                features[id] = get_embedding_from_fasttext(label)
        elif embedding_type=='ggnews':
            import gensim
            ggnews = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
            for label, id in tqdm(label2id.items()):
                features[id] = get_embedding_from_ggnews(label, ggnews)
        elif embedding_type=='glove':
            for label, id in tqdm(label2id.items()):
                features[id] = get_embedding_from_glove(label)
    return edges, features

def read_dataset(train_path, val_path, embedding_type, threshold):
    train = json.load(open(train_path))
    val = json.load(open(val_path))

    train_sents = [clean_string(text) for text in train['text']]
    val_sents = [clean_string(text) for text in val['text']]

    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train['label'])
    val_labels = mlb.transform(val['label'])

    edges, label_features = PMI_create_edges_and_features(train, mlb, embedding_type, threshold)

    return train_sents, train_labels, val_sents, val_labels, edges, label_features

def load_data(train_path, val_path, max_length, batch_size, device, embedding_type, threshold):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    train_sents, train_labels, val_sents, val_labels, edges, label_features = read_dataset(train_path, val_path, embedding_type, threshold)
    X_train = tokenizer.batch_encode_plus(train_sents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    y_train = torch.tensor(train_labels)
    X_val = tokenizer.batch_encode_plus(val_sents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    y_val = torch.tensor(val_labels)

    train_tensor = TensorDataset(X_train['input_ids'].to(device), X_train['attention_mask'].to(device), y_train.to(device))
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    val_tensor = TensorDataset(X_val['input_ids'].to(device), X_val['attention_mask'].to(device), y_val.to(device))
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, edges, label_features

def plot(x_label, y_label, x_axis, figsize=(10,5), **kwargs):
    plt.figure(figsize=figsize)
    y_axis, color, legend = kwargs.get('y_axis'), kwargs.get('color'), kwargs.get('legend')
    if isinstance(y_axis, list):
        for y, c, lb in zip(y_axis, color, legend):
            plt.plot(x_axis, y, color = c, marker = '.', label = lb)
    else:
        plt.plot(x_axis, y_axis, color = color, marker = '.', label = legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(np.arange(0, len(x_axis), 5))
    plt.legend()
    plt.show()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.init_weights()
    
    def init_weights(self):
        stdv = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj_matrix):
        sp = torch.matmul(x, self.weights)
        output = torch.matmul(adj_matrix, sp)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
            
class BertGCN(nn.Module):
    def __init__(self, edges, features):
        super(BertGCN, self).__init__()
        self.bert = AutoModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.2)
        
        self.label_features = features
        self.edges = edges
        self.gc1 = GCNLayer(features.size(1), self.bert.config.hidden_size)
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)['last_hidden_state'][:, 0]
        bert_output = self.dropout(bert_output)

        label_embed = self.gc1(self.label_features, self.edges)
        label_embed = F.relu(label_embed)
        
        output = torch.matmul(bert_output, label_embed.T)
        return output

# Init model and optimizer & schedule
def initialize_model(edges, label_features, device, len_trainloader, epochs, lr=3e-5):
    model = BertGCN(edges.to(device), label_features.to(device))
    model.to(device)
    
    no_decay = ['bias', 'LayerNorm.weight']
    param_optimizer = [[name, para] for name, para in model.named_parameters() if para.requires_grad]
    optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
                'weight_decay': 0.01},
                {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
            'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    n_steps = len_trainloader * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=n_steps, num_warmup_steps=100)
    criterion = nn.BCEWithLogitsLoss()

    return model, optimizer, scheduler, criterion
    
def step(model, optimizer, scheduler, criterion, batch):
    input_ids, attention_mask, label = batch
    optimizer.zero_grad()
    y_pred = model.forward(input_ids, attention_mask)

    loss = criterion(y_pred, label.float())
    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss.item()
    
def validate(model, criterion, val_loader):
    print("Evaluating...")
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        pred_labels, targets = list(), list()

        for _, batch in enumerate(val_loader):
            input_ids, attention_mask, y_true = batch
            output = model(input_ids, attention_mask)
            loss = criterion(output, y_true.float())

            running_loss += loss.item()

            pred_labels.extend(torch.sigmoid(output).detach().cpu().numpy())
            targets.extend(y_true.detach().cpu().numpy())

        val_loss = running_loss/len(val_loader)

    pred_labels, targets = np.array(pred_labels), np.array(targets)
    accuracy = metrics.accuracy_score(targets, pred_labels.round())
    micro_f1 = metrics.f1_score(targets, pred_labels.round(), average='micro')
    macro_f1 = metrics.f1_score(targets, pred_labels.round(), average='macro')
    
    ndcg1 = metrics.ndcg_score(targets, pred_labels, k=1)
    ndcg3 = metrics.ndcg_score(targets, pred_labels, k=3)
    ndcg5 = metrics.ndcg_score(targets, pred_labels, k=5)

    p1 = Precision(num_classes=101, top_k=1)(torch.tensor(pred_labels), torch.tensor(targets))
    p3 = Precision(num_classes=101, top_k=3)(torch.tensor(pred_labels), torch.tensor(targets))
    p5 = Precision(num_classes=101, top_k=5)(torch.tensor(pred_labels), torch.tensor(targets))

    return val_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5

def train(model, optimizer, scheduler, criterion, train_loader, val_loader, checkpoint, epochs=20):

    early_stopping = EarlyStopping(delta=1e-5, patience=10)
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, batch in enumerate(train_loader):
            loss = step(model, optimizer, scheduler, criterion, batch)
            running_loss += loss 
            if (i + 1) % 100 == 0 or i == 0:
                print("Epoch: {} - iter: {}/{} - train_loss: {}".format(epoch+1, i + 1, len(train_loader), running_loss/(i + 1)))
        else:
            print("Epoch: {} - iter: {}/{} - train_loss: {}".format(epoch+1, i + 1, len(train_loader), running_loss/len(train_loader)))
            val_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = validate(model, criterion, val_loader)

            train_losses.append(running_loss/(i+1))
            val_losses.append(val_loss), val_accs.append(accuracy)
            print("Val_loss: {} - Accuracy: {} - Micro-F1: {} - Macro-F1: {}".format(val_loss, accuracy, micro_f1, macro_f1))
            print("nDCG1: {} - nDCG@3: {} - nDCG@5: {} - P@1: {} - P@3: {} - P@5: {}".format(ndcg1, ndcg3, ndcg5, p1, p3, p5))

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print('Early stoppping. Previous model saved in: ', checkpoint)
                train_losses, val_losses, val_accs = np.array(train_losses).reshape(-1, 1), np.array(val_losses).reshape(-1, 1), np.array(val_accs).reshape(-1, 1)
                np.savetxt(os.path.join(checkpoint, 'log.txt'), np.hstack((train_losses, val_losses, val_accs)), delimiter='#')
                break
            torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'val_loss': val_loss
            }, os.path.join(checkpoint, 'cp'+str(epoch+1)+'.pt'))

    train_losses, val_losses, val_accs = np.array(train_losses).reshape(-1, 1), np.array(val_losses).reshape(-1, 1), np.array(val_accs).reshape(-1, 1)
    np.savetxt(os.path.join(checkpoint, 'log.txt'), np.hstack((train_losses, val_losses, val_accs)), delimiter='#')


def main():
    parser = argparse.ArgumentParser(description='RCV1 Classification')

    parser.add_argument('--train_data', type=str, default='data/rcv1_train_data.json', help='The train dataset directory.')
    parser.add_argument('--val_data', type=str, default='data/rcv1_val_data.json', help='The val dataset directory')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default= 16, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number epochs')
    parser.add_argument('--max_length', type=int, default= 384, help='max sequence length')
    parser.add_argument('--embedding_type', type=str, default='wiki', help='type of the word embeding: wiki, random, glove, fasttext, ggnews')
    parser.add_argument('--checkpoint', type=str, default='checkpoint', help='check point')
    # parser.add_argument('--resume', type=int, default=0, help='resume train model from checkpoint')
    parser.add_argument('--graph_feature', type=str, default='./data/graph_feature.pth', help='path to feature of graph: adjacency, node feature')
    parser.add_argument('--threshold', type=float, default=0.0)
    args = parser.parse_args()

    TRAIN_DATA = args.train_data
    VAL_DATA = args.val_data
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length
    CHECK_POINT = args.checkpoint
    EMBEDDING_TYPE = args.embedding_type
    # RESUME = args.resume
    EPOCHS = args.epochs
    GRAPH_FEATURE = args.graph_feature
    LR = args.lr
    THRESHOLD = args.threshold
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed_all()

    # read dataset
    print('reading dataset...')
    train_loader, val_loader, edges, label_features = load_data(train_path=TRAIN_DATA, val_path=VAL_DATA, max_length=MAX_LENGTH, batch_size=BATCH_SIZE, device=DEVICE, embedding_type=EMBEDDING_TYPE, threshold=THRESHOLD)
    torch.save({'edges': edges,
                'label_features' :label_features,
                'len_trainloader': len(train_loader),
                'epochs': EPOCHS,
                'batch_size':BATCH_SIZE,
                'max_length': MAX_LENGTH}, GRAPH_FEATURE)
    print('initialize model')
    model, optimizer, scheduler, criterion = initialize_model(edges=edges, label_features=label_features, device=DEVICE, len_trainloader=len(train_loader), epochs = EPOCHS, lr=LR )
    # if RESUME:
    #     checkpoint = torch.load(CHECK_POINT)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    
    print('training model...')
    train(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, train_loader=train_loader, val_loader=val_loader, checkpoint=CHECK_POINT, epochs=EPOCHS)


if __name__ == '__main__':
    main()
