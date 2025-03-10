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

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Actionsrecognition.Models import *
from Visualizer import plot_graphs, plot_confusion_metrix
from sklearn.metrics import precision_recall_fscore_support

import pandas as pd

device = 'cuda'

epochs = 50
batch_size = 256

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
    eval_skel_list = []
    mname_list = []
    save_folder = 'Actionsrecognition/_saved/'        
    skel_folder_list = ['Actionsrecognition/_bed_day', 'Actionsrecognition/_bed_night', 'Actionsrecognition/_prison_day', 'Actionsrecognition/_prison_night']
    
    for mname in os.listdir(save_folder):
        if(mname[0] == '_'):
            continue
        sub_folder = os.path.join(save_folder, mname)

        if os.path.isdir(sub_folder):
            mname_list.append(mname)
    
    for skel_folder in skel_folder_list:
        sk_list = []
        
        for s in os.listdir(skel_folder):
            sk = os.path.join(skel_folder, s)
            sk_list.append(sk)
        eval_skel_list.append(sk_list)
        
    # MODEL.
    graph_args = {'strategy': 'spatial'}
    model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = Adadelta(model.parameters())

    losser = torch.nn.BCELoss()

    df = pd.DataFrame()
    reasonable_df = pd.DataFrame()

    for k, mname in enumerate(mname_list):       
        orf_val = mname.split('ODR')[-1].split('_')[0]
        n_val = mname.split('N')[-2].split('_')[0]
        ls_val = mname.split('LS')[-1].split('_')[0]
        nf_val = mname.split('NF')[-1].split('_')[0]

        working_folder = os.path.join(save_folder, mname)  
        pths_list = glob.glob(working_folder +'/*.pth')
        
        
        for pth_path in pths_list:
            model.load_state_dict(torch.load(pth_path))
            pth_name = os.path.basename(pth_path)

            # tsstg-model.pth은 고려하지 않음
            if(pth_name == 'tsstg-model.pth'):
                continue

            # EVALUATION.
            model = set_training(model, False)
            
            for i in range(len(eval_skel_list)):
                skel_list = eval_skel_list[i]
                skel_folder_name = skel_list[0].split('/')[-2]

                ret_folder = os.path.join(working_folder, os.path.basename(pth_path).split('.')[0], skel_folder_name)
                if not os.path.exists(ret_folder):
                    os.makedirs(ret_folder)

                for j in range(len(skel_list)):
                    data_file = skel_list[j]
                    eval_loader, _ = load_dataset([data_file], batch_size)

                    testSF_val = os.path.basename(skel_list[j]).split('-nf')[-1].split('.')[0]

                    print('Evaluation.')
                    run_loss = 0.0
                    run_accu = 0.0
                    y_preds = []
                    y_trues = []
                    with tqdm(eval_loader, desc='eval') as iterator:
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

                    y_trues_f = np.array(y_trues)
                    y_preds_f = np.array(y_preds)
                    precision, recall, f1_score, _ = precision_recall_fscore_support(y_trues_f, y_preds_f, average=None, zero_division=0)
                    f1_score_avg = np.mean(f1_score)

                    # Confusion metrix (frames)
                    plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu: {:.4f}, F1-Score: {:.4f}'.format(
                    os.path.basename(data_file), run_loss, run_accu, f1_score_avg),
                    save=os.path.join(ret_folder, 'F_' + skel_folder_name + str(j) + '_{}-confusion_matrix_frames.png'.format(os.path.basename(data_file).split('.')[0].split('-')[0])))

                    # Confusion metrix (rates)
                    plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu: {:.4f}'.format(
                        os.path.basename(data_file), run_loss, run_accu, f1_score_avg), 'true', 
                    save=os.path.join(ret_folder, 'R_'+ skel_folder_name + str(j) + '_{}-confusion_matrix_rates.png'.format(os.path.basename(data_file).split('.')[0].split('-')[0])))

                    print('Eval Loss: {:.4f}, Accu: {:.4f}'.format(run_loss, run_accu))
                    
                    ret_data = {'model_name':[mname], 'pth_name':[pth_name], 'ORF':[orf_val], 'N':[n_val], 'LS':[ls_val], 'TS':[nf_val], 
                                'testTS':[testSF_val],'Acc':[run_accu], 'F1-Score':[f1_score_avg]}

                    new_row = pd.DataFrame(ret_data)
                    df = pd.concat([df, new_row], ignore_index=True)

                    #정확도와 F1score가 0.6이상이면 저장
                    if(0.6 < f1_score_avg) and (0.6 < run_accu): 
                        reasonable_df = pd.concat([reasonable_df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_folder, str(k)+'_eval.csv'))
        reasonable_df.to_csv(os.path.join(save_folder, str(k)+'_reasonable_eval.csv'))

    df.to_csv(os.path.join(save_folder, 'eval.csv'))
    reasonable_df.to_csv(os.path.join(save_folder, 'reasonable_eval.csv'))
