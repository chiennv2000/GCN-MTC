import re
import json
import argparse
from tqdm import tqdm
import torch
import numpy as np
from torch._C import device
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from train import initialize_model
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from torchmetrics import Precision
import math


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()

def read_dataset(train_path, test_path):
    train = json.load(open(train_path))
    test = json.load(open(test_path))

    test_sents = [clean_string(text) for text in test['text']]

    mlb = MultiLabelBinarizer()
    mlb.fit(train['label'])
    test_labels = mlb.transform(test['label'])

    return test_sents, test_labels

class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self,  sents, labels, tokenizer, max_length):
        super(IterableDataset).__init__()
        self.sents = sents
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length
        self.length = len(sents)

    def __iter__(self):
        for i in range(len(self.sents)):
            tokens = self.tokenizer.encode(self.sents[i], padding=True, truncation=True, pad_to_max_length = True, max_length=self.max_length, return_attention_mask = True)

            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            mask = torch.zeros(self.max_length, dtype=torch.long)
                
            current_len =  len(tokens)
            if current_len > self.max_length:
                    input_ids = torch.LongTensor(tokens[:self.max_length])
                    mask = torch.LongTensor([1]*self.max_length)
            else:
                    input_ids[:current_len] = torch.LongTensor(tokens)
                    mask[:current_len] = torch.LongTensor([1]*current_len)

            label = self.labels[i]
                
            yield input_ids, mask, label

def load_data(train_path, test_path, max_length, batch_size):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    test_sents, test_labels = read_dataset(train_path, test_path)

    # X_test = tokenizer.batch_encode_plus(test_sents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    # y_test = torch.tensor(test_labels)

    test_tensor = IterableDataset(test_sents, test_labels, tokenizer, max_length)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    return test_tensor, test_loader

def validate(model, loss_fn, test_tensor, test_loader, batch_size, num_classes, device):
    print("Evaluating")
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        predicted_labels, target_labels = list(), list()

        with tqdm(total = math.ceil(test_tensor.length/batch_size)) as pbar:
            for _, batch in enumerate(test_loader):
                input_ids, attention_mask, y_true = tuple(t.to(device) for t in batch)
                output = model.forward(input_ids, attention_mask)
                loss = loss_fn(output, y_true.float())

                # print(loss.item())
                running_loss += loss.item()

                target_labels.extend(y_true.detach().cpu().numpy())
                predicted_labels.extend(torch.sigmoid(output).detach().cpu().numpy())
                pbar.update(1)
        val_loss = running_loss/math.ceil(test_tensor.length/batch_size)

    predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
    accuracy = metrics.accuracy_score(target_labels, predicted_labels.round())
    micro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='micro')
    macro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='macro')

    ndcg1 = metrics.ndcg_score(target_labels, predicted_labels, k=1)
    ndcg3 = metrics.ndcg_score(target_labels, predicted_labels, k=3)
    ndcg5 = metrics.ndcg_score(target_labels, predicted_labels, k=5)

    p1 = Precision(num_classes=num_classes, top_k=1)(torch.tensor(predicted_labels), torch.tensor(target_labels))
    p3 = Precision(num_classes=num_classes, top_k=3)(torch.tensor(predicted_labels), torch.tensor(target_labels))
    p5 = Precision(num_classes=num_classes, top_k=5)(torch.tensor(predicted_labels), torch.tensor(target_labels))

    return val_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5

# def validate(model, criterion, test_loader, device):
#     print("Evaluating...")
#     model.eval()
#     with torch.no_grad():
#         running_loss = 0.0
#         pred_labels, targets = list(), list()

#         for _, batch in enumerate(test_loader):
#             input_ids, attention_mask, y_true = tuple(t.to(device) for t in batch)

#             output = model(input_ids, attention_mask)
#             loss = criterion(output, y_true.float())

#             running_loss += loss.item()

#             pred_labels.extend(torch.sigmoid(output).detach().cpu().numpy())
#             targets.extend(y_true.detach().cpu().numpy())

#         val_loss = running_loss/len(test_loader)

#     pred_labels, targets = np.array(pred_labels), np.array(targets)
#     accuracy = metrics.accuracy_score(targets, pred_labels.round())
#     micro_f1 = metrics.f1_score(targets, pred_labels.round(), average='micro')
#     macro_f1 = metrics.f1_score(targets, pred_labels.round(), average='macro')
    
#     ndcg1 = metrics.ndcg_score(targets, pred_labels, k=1)
#     ndcg3 = metrics.ndcg_score(targets, pred_labels, k=3)
#     ndcg5 = metrics.ndcg_score(targets, pred_labels, k=5)

#     p1 = Precision(num_classes=101, top_k=1)(torch.tensor(pred_labels), torch.tensor(targets))
#     p3 = Precision(num_classes=101, top_k=3)(torch.tensor(pred_labels), torch.tensor(targets))
#     p5 = Precision(num_classes=101, top_k=5)(torch.tensor(pred_labels), torch.tensor(targets))

#     return val_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5

def main():
    parser = argparse.ArgumentParser('pharse test RCV1 dataset')

    parser.add_argument('--train_path', type=str,default='./data/rcv1_train_data.json',help='path to train data')
    parser.add_argument('--test_path',type=str,default='./data/rcv1_train_test.json',help='path to test data')
    parser.add_argument('--checkpoint',type=str,default='.',help='path to checkpoint file')
    parser.add_argument('--graph_feature', type=str, default='./data/graph_feature.pth', help='path to feature of graph: adjacency, node feature')

    args = parser.parse_args()

    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path
    CHECK_POINT = args.checkpoint
    GRAPH_FEATURE = args.graph_feature
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('load graph feature, batch size, max_length...')
    g_feature = torch.load(GRAPH_FEATURE)
    # print(g_feature)
    edges, label_features, len_trainloader, epochs = g_feature['edges'], g_feature['label_features'], g_feature['len_trainloader'], g_feature['epochs']
    batch_size, max_length = g_feature['batch_size'], g_feature['max_length']
    print('use batch size: {}, max_length: {}, epochs: {}'.format(batch_size, max_length, epochs))

    print('reading test data and creating test loader')
    test_tensor, test_loader = load_data(train_path=TRAIN_PATH, test_path=TEST_PATH, max_length=max_length, batch_size=batch_size)

    print('initialize model...')
    model, _, _, criterion = initialize_model(edges=edges, label_features=label_features, device=DEVICE, len_trainloader=len_trainloader, epochs = epochs, lr=5e-5)

    #load weigth to model
    print('load checpoint from: ', CHECK_POINT)
    checkpoint = torch.load(CHECK_POINT)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Val loss: {}'.format(checkpoint['val_loss']))
    # validate model
    print('test model...')
    test_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = validate(model, criterion, test_tensor=test_tensor, test_loader=test_loader, batch_size=batch_size, num_classes=101, device=DEVICE)
    print("Test_loss: {} - Accuracy: {} - Micro-F1: {} - Macro-F1: {}".format(test_loss, accuracy, micro_f1, macro_f1))
    print("nDCG1: {} - nDCG@3: {} - nDCG@5: {} - P@1: {} - P@3: {} - P@5: {}".format(ndcg1, ndcg3, ndcg5, p1, p3, p5))


if __name__ == '__main__':
    main()
        
