import torch
import numpy as np
from misc import *
from hsic import *
from model import *

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import datetime
import os
from load_data import *

from tqdm import tqdm



def standard_train(cepoch, model, data_loader, optimizer, config_dict):

    batch_acc    = AverageMeter()
    batch_loss   = AverageMeter()
    batch_hischx = AverageMeter()
    batch_hischy = AverageMeter()

    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    prec1 = total_loss = hx_l = hy_l = -1

    batch_log = {}
    batch_log['batch_acc'] = []
    batch_log['batch_loss'] = []
    batch_log['batch_hsic_hx'] = []
    batch_log['batch_hsic_hy'] = []

    model = model.to(config_dict['device'])

    n_data = config_dict['batch_size'] * len(data_loader)
    
    pbar = tqdm(enumerate(data_loader), total=n_data/config_dict['batch_size'], ncols=150)

    for batch_idx, (data, target) in pbar:

        data   = data.to(config_dict['device'])
        target = target.to(config_dict['device'])
        output, hiddens = model(data)

        h_target = target.view(-1,1)
        h_target = to_categorical(h_target, num_classes=10).float()
        
        h_data = data.view(-1, np.prod(data.size()[1:]))

        optimizer.zero_grad()
        loss = cross_entropy_loss(output, target)
        loss.backward()
        optimizer.step()


        loss = float(loss.detach().cpu().numpy())
        prec1, prec5 = get_accuracy(output, target, topk=(1, 5)) 
        prec1 = float(prec1.cpu().numpy())
    
        batch_acc.update(prec1)   
        batch_loss.update(loss)  
        batch_hischx.update(hx_l)
        batch_hischy.update(hy_l)

        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] Loss:{loss:.4f} Acc:{acc:.4f} hsic_xz:{hsic_zx:.4f} hsic_yz:{hsic_zy:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*config_dict['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 
                        loss = batch_loss.avg, 
                        acc  = batch_acc.avg,
                        hsic_zx = batch_hischx.avg,
                        hsic_zy = batch_hischy.avg,
                    )

        # # # preparation log information and print progress # # #
        if ((batch_idx) % config_dict['log_batch_interval'] == 0): 
            batch_log['batch_acc'].append(batch_acc.val)
            batch_log['batch_loss'].append(batch_loss.val)
            batch_log['batch_hsic_hx'].append(batch_hischx.val)
            batch_log['batch_hsic_hy'].append(batch_hischy.val)

        pbar.set_description(msg)

    return batch_log


def hsic_train(cepoch, model, data_loader, config_dict):

    # cross_entropy_loss = torch.nn.CrossEntropyLoss()
    prec1 = total_loss = hx_l = hy_l = -1

    batch_acc    = AverageMeter()
    batch_loss   = AverageMeter()
    batch_hischx = AverageMeter()
    batch_hischy = AverageMeter()

    batch_log = {}
    batch_log['batch_acc'] = []
    batch_log['batch_loss'] = []
    batch_log['batch_hsic_hx'] = []
    batch_log['batch_hsic_hy'] = []

    model = model.to(config_dict['device'])

    n_data = config_dict['batch_size'] * len(data_loader)

    pbar = tqdm(enumerate(data_loader), total=n_data/config_dict['batch_size'], ncols=120)
    for batch_idx, (data, target) in pbar:

        data   = data.to(config_dict['device'])
        target = target.to(config_dict['device'])
        output, hiddens = model(data)

        h_target = target.view(-1,1)
        h_target = to_categorical(h_target, num_classes=10).float()
        h_data = data.view(-1, np.prod(data.size()[1:]))

        idx_range = []
        it = 0

        # Logic: Access wgt and bias of each layer - one layer at a time
        # hence - skip 2.

        for i in range(len(hiddens)):
            idx_range.append(np.arange(it, it+2).tolist())
            it += 2
    
        ## o/p: idx_range = [[0,1], [2,3], ...]
         
        for i in range(len(hiddens)):
            
            output, hiddens = model(data)
            params, param_names = get_layer_parameters(model=model, idx_range=idx_range[i]) # so we only optimize one layer at a time
            optimizer = optim.SGD(params, lr = config_dict['learning_rate'], momentum=.9, weight_decay=0.001)
            optimizer.zero_grad()
            
            if len(hiddens[i].size()) > 2:
                hiddens[i] = hiddens[i].view(-1, np.prod(hiddens[i].size()[1:]))

            hx_l, hy_l = hsic_loss_obj(
                    hiddens[i],
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=config_dict['sigma'])

            loss = (hx_l - config_dict['lambda_y']*hy_l)
            loss.backward()
            optimizer.step()

        batch_acc.update(prec1)
        batch_loss.update(total_loss)
        batch_hischx.update(hx_l.cpu().detach().numpy())
        batch_hischy.update(hy_l.cpu().detach().numpy())

        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] H_hx:{H_hx:.8f} H_hy:{H_hy:.8f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*config_dict['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 
                        H_hx = batch_hischx.avg, 
                        H_hy = batch_hischy.avg,
                )

        if ((batch_idx+1) % config_dict['log_batch_interval'] == 0):

            batch_log['batch_acc'].append(batch_acc.avg)
            batch_log['batch_loss'].append(batch_loss.avg)
            batch_log['batch_hsic_hx'].append(batch_hischx.avg)
            batch_log['batch_hsic_hy'].append(batch_hischy.avg)

        pbar.set_description(msg)

    return batch_log, model


from color import print_emph, print_highlight
from utils import model_save
import datetime

# 1. train the HSIC model with last hidden demensions != 10 and save the model.T_destination
## 2. Create ensemble model with hsic model + linear model with softmax and train for 10 epochs
## without backprop (only update last layer params by SGD optim.)

def training_hsic(config_dict):

    print_emph("HSIC-Bottleneck training")

    train_loader, test_loader = get_data(
        config_dict['data_code'], config_dict['batch_size'])

    model = ModelConv(**config_dict)
    nepoch = config_dict['epochs']
    epoch_range = range(1, nepoch+1)

    batch_log_list = []
    epoch_log_dict = {}
    epoch_log_dict['train_acc'] = []
    epoch_log_dict['test_acc'] = []

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for cepoch in epoch_range:

        log = hsic_train(cepoch, model, train_loader, config_dict)

        batch_log_list.append(log)

        # save with each indexed
        main_path = 'X:/Data Files/Huawei/'
        filename = os.path.join(main_path, config_dict['data_code'])
        model_report, model_data = model_save.create_dirs(filename, time_stamp)

        model_save_path = os.path.join(model_data, "model---{:04d}.pt".format(cepoch))
        torch.save(model.state_dict(), model_save_path)

        log_dict = {}
        log_dict['batch_log_list'] = batch_log_list
        log_dict['epoch_log_dict'] = epoch_log_dict
        log_dict['config_dict'] = config_dict

        model_report_path = os.path.join(model_report, "log---{:04d}.npy".format(cepoch))
        np.save(model_report_path, log_dict)

    return batch_log_list, epoch_log_dict


def training_format(config_dict, model_load_path):

    print_emph("Format training")

    train_loader, test_loader = get_data(
        config_dict['data_code'], config_dict['batch_size'])
    torch.manual_seed(config_dict['seed'])

    vanilla_model = ModelVanilla(**config_dict) # Implement this - Done
    torch.manual_seed(config_dict['seed'])

    hsic_model = ModelConv(**config_dict)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, vanilla_model.parameters()),
            lr = config_dict['learning_rate'], weight_decay=0.001)

    model = torch.load(model_load_path)

    hsic_model.load_state_dict(model)
    hsic_model.eval()

    ensemble_model = ModelEnsemble(hsic_model, vanilla_model) ## Implement this - Done

    batch_log_list = []
    epoch_log_dict = {}
    epoch_log_dict['train_acc'] = []
    epoch_log_dict['train_loss'] = []
    epoch_log_dict['test_acc'] = []
    epoch_log_dict['test_loss'] = []

    nepoch = config_dict['epochs']

    test_acc, test_loss = get_accuracy_epoch(ensemble_model, test_loader)
    epoch_log_dict['test_acc'].append(test_acc)
    epoch_log_dict['test_loss'].append(test_loss)

    train_acc, train_loss = get_accuracy_epoch(ensemble_model, train_loader)
    epoch_log_dict['train_acc'].append(train_acc)
    epoch_log_dict['train_loss'].append(train_loss)

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for cepoch in range(1, nepoch+1):
        log = standard_train(cepoch, ensemble_model, train_loader, optimizer, config_dict)
        batch_log_list.append(log)

        train_acc, train_loss = get_accuracy_epoch(ensemble_model, train_loader)
        epoch_log_dict['train_acc'].append(train_acc)
        epoch_log_dict['train_loss'].append(train_loss)
        test_acc, test_loss = get_accuracy_epoch(ensemble_model, test_loader)
        epoch_log_dict['test_acc'].append(test_acc)
        epoch_log_dict['test_loss'].append(test_loss)
        print_highlight("Epoch - [{:04d}]: Training Acc: {:.2f}".format(cepoch, train_acc), 'green')
        print_highlight("Epoch - [{:04d}]: Testing  Acc: {:.2f}".format(cepoch, test_acc), 'green')


        # save with each indexed
        main_path = 'X:/Data Files/Huawei/'
        filename = os.path.join(main_path, config_dict['data_code'])
        model_report, model_data = model_save.create_dirs(filename, time_stamp)

        model_save_path = os.path.join(model_data, "Ensem_model---{:04d}.pt".format(cepoch))
        torch.save(ensemble_model.state_dict(), model_save_path)

        log_dict = {}
        log_dict['batch_log_list'] = batch_log_list
        log_dict['epoch_log_dict'] = epoch_log_dict
        log_dict['config_dict'] = config_dict

        model_report_path = os.path.join(model_report, "Ensem_log---{:04d}.npy".format(cepoch))
        np.save(model_report_path, log_dict)

    return batch_log_list, epoch_log_dict