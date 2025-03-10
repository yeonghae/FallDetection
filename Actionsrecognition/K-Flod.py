import os
import sys
import time
import torch
import pickle
import numpy as np
import torch.nn.functional as F
import glob
import natsort
import re
import numpy as np
from shutil import copyfile
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Actionsrecognition.Models import *
from Visualizer import plot_graphs, plot_confusion_metrix

device = 'cuda'

epochs = 50
batch_size = 256

data_files = glob.glob('C:/Users/rty33/Documents/FallDetection-original/Data/for_kfold/*.pkl')
# data_files = glob.glob('C:/Users/rty33/Documents/FallDetection-original/__SyntheticData/*.pkl')
data_files = natsort.natsorted(data_files)

class_names = ['Normal', 'Fall Down']
num_class = len(class_names)

def load_dataset(data_files, batch_size, split_size=0):
    """Load data files into torch DataLoader with/without spliting train-test.
    """
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    if split_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=split_size,
                                                              random_state=9)
        train_set = data.TensorDataset(torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_train, dtype=torch.float32))
        valid_set = data.TensorDataset(torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_valid, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_set, batch_size)
    else:
        train_set = data.TensorDataset(torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(labels, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = None
    return train_loader, valid_loader

def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()

def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model

# milk add in 2023/12/10
def load_all_data(data_files):
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels
# milk end add in 2023/12/10

if __name__ == '__main__':
    n_splits = 5
    test_name = "prison_test_data"
    for i in range(1):
        # load all data
        features,labels = load_all_data(data_files[i:i+1])

        # vaild split 
        split_size = 0.2
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=split_size,
                                                              random_state=9)
        
        
        features = x_train
        labels = y_train
        
        # 5-fold
        kf = KFold(n_splits=n_splits, shuffle=True)

        temp = data_files[0]

        save_folder = 'saved/kfold_' + str(n_splits) + '_' + test_name
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Model
        graph_args = {'strategy': 'spatial'}
        model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)

        # set model in evaluation mode
        model = set_training(model, False)

        # milk modify in 2023/12/10
        accuracies, precisions, recalls, f1_scores = [], [], [], []

        round = 0
        for train_index, test_index in kf.split(features):
            
            round += 1
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            
            train_set = data.TensorDataset(torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_train, dtype=torch.float32))
            test_set = data.TensorDataset(torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_test, dtype=torch.float32))
            val_set = data.TensorDataset(torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_valid, dtype=torch.float32))
            train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
            test_loader = data.DataLoader(test_set, batch_size, shuffle=True)
            val_loader = data.DataLoader(val_set, batch_size, shuffle=True)
            
            data_loader = {'train': train_loader, 'valid': val_loader}
            
            # train
            optimizer = Adadelta(model.parameters())
            losser = torch.nn.BCELoss()

            loss_list = {'train': [], 'valid': []}
            accu_list = {'train': [], 'valid': []}
            for e in range(epochs):
                print('Epoch {}/{}'.format(e, epochs - 1))
                for phase in ['train','valid']:
                    if phase == 'train':
                        model = set_training(model, True)

                    run_loss = 0.0
                    run_accu = 0.0
                    with tqdm(data_loader[phase], desc=phase) as iterator:
                        for pts, lbs in iterator:
                            # Create motion input by distance of points (x, y) of the same node
                            # in two frames.
                            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]

                            mot = mot.to(device)
                            pts = pts.to(device)
                            lbs = lbs.to(device)

                            # Forward.
                            out = model((pts, mot))
                            loss = losser(out, lbs)

                            if phase == 'train':
                                # Backward.
                                model.zero_grad()
                                loss.backward()
                                optimizer.step()

                            run_loss += loss.item()
                            accu = accuracy_batch(out.detach().cpu().numpy(),
                                                lbs.detach().cpu().numpy())
                            run_accu += accu

                            iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                                loss.item(), accu))
                            iterator.update()
                            #break
                    loss_list[phase].append(run_loss / len(iterator))
                    accu_list[phase].append(run_accu / len(iterator))
                    #break

                print('Summary epoch:\n - Train loss: {:.4f}, accu: {:.4f}\n - Valid loss:'
                ' {:.4f}, accu: {:.4f}'.format(loss_list['train'][-1], accu_list['train'][-1],
                                                loss_list['valid'][-1], accu_list['valid'][-1]))

                # SAVE.
                torch.save(model.state_dict(), os.path.join(save_folder, f'tsstg-model_{round}.pth'))       
                
                plot_graphs(list(loss_list.values()), list(loss_list.keys()),
                        'Last Train: {:.2f}, Valid: {:.2f}'.format(
                            loss_list['train'][-1], loss_list['valid'][-1]
                        ), 'Loss', xlim=[0, epochs],
                        save=os.path.join(save_folder, f'loss_graph_{round}.png'))
                plot_graphs(list(accu_list.values()), list(accu_list.keys()),
                        'Last Train: {:.2f}, Valid: {:.2f}'.format(
                            accu_list['train'][-1], accu_list['valid'][-1]
                        ), 'Accu', xlim=[0, epochs],
                        save=os.path.join(save_folder, f'accu_graph_{round}.png'))

                #break

            model.load_state_dict(torch.load(os.path.join(save_folder, f'tsstg-model_{round}.pth')))


            model = set_training(model, False)
            data_file = data_files[2] # Prison or trestbed
            # eval_loader, _ = load_dataset([data_file], batch_size)

            print('Evaluation.')
            run_loss = 0.0
            run_accu = 0.0
            y_preds = []
            y_trues = []
            with tqdm(test_loader, desc='eval') as iterator:
                for pts, lbs in iterator:
                    mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
                    mot = mot.to(device)
                    pts = pts.to(device)
                    lbs = lbs.to(device)

                    out = model((pts, mot))
                    loss = losser(out, lbs)

                    run_loss += loss.item()
                    accu = accuracy_batch(out.detach().cpu().numpy(),
                                        lbs.detach().cpu().numpy())
                    run_accu += accu

                    y_preds.extend(out.argmax(1).detach().cpu().numpy())
                    y_trues.extend(lbs.argmax(1).cpu().numpy())

                    iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                        loss.item(), accu))
                    iterator.update()

            run_loss = run_loss / len(iterator)
            run_accu = run_accu / len(iterator)
            
            # Confusion metrix (frames)
            plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu: {:.4f}'.format(
                os.path.basename(data_file), run_loss, run_accu
            ), None, save=os.path.join(save_folder, '{}-confusion_matrix_frames_{}.png'.format(os.path.basename(data_file).split('.')[0],round)))
            
            # Confusion metrix (rates)
            plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu: {:.4f}'.format(
                os.path.basename(data_file), run_loss, run_accu
            ), 'true', save=os.path.join(save_folder, '{}-c{}.png'.format(os.path.basename(data_file).split('.')[0],round)))

            print('Eval Loss: {:.4f}, Accu: {:.4f}'.format(run_loss, run_accu))

            
            print(torch.unique(torch.tensor(y_test, dtype=torch.float32)))
            print(torch.unique(torch.tensor(y_preds, dtype=torch.float32)))
            y_preds = np.array(y_preds)
            # y_preds = np.argmax(y_preds,axis=1)
            y_preds = torch.tensor(y_preds, dtype=torch.float32)
            y_test = np.argmax(y_test,axis=1)
            # milk add in 2023/12/10
            # accuracies.append(accuracy_batch(torch.tensor(y_preds,dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)))
            accuracies.append(accuracy_score(torch.tensor(y_test, dtype=torch.float32), y_preds))
            precisions.append(precision_score(torch.tensor(y_test, dtype=torch.float32), y_preds))
            recalls.append(recall_score(torch.tensor(y_test, dtype=torch.float32), y_preds))
            f1_scores.append(f1_score(torch.tensor(y_test, dtype=torch.float32), y_preds))

        
        average_accuracy = sum(accuracies) / len(accuracies)
        average_precision = sum(precisions) / len(precisions)
        average_recall = sum(recalls) / len(recalls)
        average_f1_score = sum(f1_scores) / len(f1_scores)

        print(f'Average accuracy: {average_accuracy:.3f}')
        print(f'Average precision: {average_precision:.3f}')
        print(f'Average recall: {average_recall:.3f}')
        print(f'Average f1 score: {average_f1_score:.3f}')

        # milk end add in 2023/12/10
        