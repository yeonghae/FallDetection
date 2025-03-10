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
from shutil import copyfile
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from sklearn.model_selection import train_test_split
from Models import *

from Visualizer import plot_graphs, plot_confusion_metrix

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


device = 'cuda'
epochs = 500
batch_size = 256

train_files = glob.glob('/home/workspace/_SyntheticData/*.pkl')
train_files = natsort.natsorted(train_files)

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

if __name__ == '__main__':
    for i in range(0, len(train_files)):
        # DATA.
        train_loader, valid_loader = load_dataset(train_files[i:i+1], batch_size, 0.3)

        temp = train_files[i:i+1]
        fall_num_tmp = re.search(r"Fall\d+", temp[0])
        fall_num = fall_num_tmp.group(0)
        normal_num_tmp = re.search(r"Normal\d+", temp[0])
        normal_num = normal_num_tmp.group(0)
        ls_num_tmp = re.search(r"LS\d+", temp[0])
        ls_num = ls_num_tmp.group(0)

        save_folder = 'saved/' + str(fall_num) + '_' + str(normal_num) + '_EP' + str(epochs) + '_BS' + str(batch_size) + "_" + str(ls_num)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        train_loader = data.DataLoader(data.ConcatDataset([train_loader.dataset]), batch_size, shuffle=True)
        dataloader = {'train': train_loader, 'valid': valid_loader}    

        # MODEL.
        graph_args = {'strategy': 'spatial'}
        model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)

        # 가중치 불러오기
        weight_file_path = "/home/workspace/Models/falldown/_C_Fall500_Normal200_EP50_BS256_LS4.pth"
        model.load_state_dict(torch.load(weight_file_path))


        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = Adadelta(model.parameters())

        losser = torch.nn.BCELoss()

        # TRAINING. 
        loss_list = {'train': [], 'valid': []}
        accu_list = {'train': [], 'valid': []}
        for e in range(epochs):
            e = e + 1
            print('Epoch {}/{}'.format(e, epochs))
            out_pth_name = str(fall_num) + '_' + str(normal_num) + '_EP' + str(e) + '_BS' + str(batch_size) + '_'+str(ls_num)+'.pth'

            
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model = set_training(model, True)
                else:
                    model = set_training(model, False)

                run_loss = 0.0
                run_accu = 0.0
                with tqdm(dataloader[phase], desc=phase) as iterator:
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
            if (e)%10==0:
                torch.save(model.state_dict(), os.path.join(save_folder, out_pth_name))
            torch.save(model.state_dict(), os.path.join(save_folder, 'last_model.pth'))


            plot_graphs(list(loss_list.values()), list(loss_list.keys()),
                        'Last Train: {:.2f}, Valid: {:.2f}'.format(
                            loss_list['train'][-1], loss_list['valid'][-1]
                        ), 'Loss', xlim=[0, epochs],
                        save=os.path.join(save_folder, 'loss_graph.png'))
            plot_graphs(list(accu_list.values()), list(accu_list.keys()),
                        'Last Train: {:.2f}, Valid: {:.2f}'.format(
                            accu_list['train'][-1], accu_list['valid'][-1]
                        ), 'Accu', xlim=[0, epochs],
                        save=os.path.join(save_folder, 'accu_graph.png'))

        del train_loader, valid_loader

        # # EVALUATION.
        # model = set_training(model, False)
        # data_file = data_files[1]
        # eval_loader, _ = load_dataset([data_file], batch_size=batch_size)

        # print('Evaluation.')
        # run_loss = 0.0
        # run_accu = 0.0
        # y_preds = []
        # y_trues = []
        # with tqdm(eval_loader, desc='eval') as iterator:
        #     for pts, lbs in iterator:
        #         mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        #         mot = mot.to(device)
        #         pts = pts.to(device)
        #         lbs = lbs.to(device)

        #         out = model((pts, mot))
        #         loss = losser(out, lbs)

        #         run_loss += loss.item()
        #         accu = accuracy_batch(out.detach().cpu().numpy(),
        #                             lbs.detach().cpu().numpy())
        #         run_accu += accu

        #         y_preds.extend(out.argmax(1).detach().cpu().numpy())
        #         y_trues.extend(lbs.argmax(1).cpu().numpy())

        #         iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
        #             loss.item(), accu))
        #         iterator.update()

        # run_loss = run_loss / len(iterator)
        # run_accu = run_accu / len(iterator)

        # plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu{:.4f}'.format(
        #     os.path.basename(data_file), run_loss, run_accu
        # ), 'true', save=os.path.join(save_folder, '{}-confusion_matrix.png'.format(
        #     os.path.basename(data_file).split('.')[0])))

        # print('Eval Loss: {:.4f}, Accu: {:.4f}'.format(run_loss, run_accu))
