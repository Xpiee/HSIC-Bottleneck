import torch
import numpy as np

# misc functions (https://github.com/choasma/HSIC-bottleneck/blob/master/source/hsicbt/utils/misc.py)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.squeeze(torch.eye(num_classes)[y])

def get_layer_parameters(model, idx_range):

    param_out = []
    param_out_name = []
    for it, (name, param) in enumerate(model.named_parameters()):
        if it in idx_range:
            param_out.append(param)
            param_out_name.append(name)

    return param_out, param_out_name

# https://github.com/choasma/HSIC-bottleneck/blob/master/source/hsicbt/utils/meter.py
class AverageMeter(object):
    """Basic meter"""
    def __init__(self):
        self.reset()

    def reset(self):
        """ reset meter
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ incremental meter
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_accuracy_hsic(model, dataloader):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []
    for batch_idx, (data, target) in enumerate(dataloader):
        output, hiddens = model(data.to(next(model.parameters()).device))
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy().reshape(-1,1)
        output_list.append(output)
        target_list.append(target)
    output_arr = np.vstack(output_list)
    target_arr = np.vstack(target_list)
    avg_acc = 0
    reorder_list = []
    for i in range(10):
        indices = np.where(target_arr==i)[0]
        select_item = output_arr[indices]
        out = np.array([np.argmax(vec) for vec in select_item])
        y = np.mean(select_item, axis=0)
        while np.argmax(y) in reorder_list:
            y[np.argmax(y)] = 0
        reorder_list.append(np.argmax(y))
        num_correct = np.where(out==np.argmax(y))[0]
        accuracy = float(num_correct.shape[0])/float(out.shape[0])
        avg_acc += accuracy
    avg_acc /= 10.

    return avg_acc*100., reorder_list

def get_accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_accuracy_epoch(model, dataloader):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []
    acc = []
    loss = []
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    model = model.to('cuda')
    device = next(model.parameters()).device

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)
        output, hiddens = model(data)
        loss.append(cross_entropy_loss(output, target).cpu().detach().numpy())
        acc.append(get_accuracy(output, target)[0].cpu().detach().numpy())
    return np.mean(acc), np.mean(loss)
