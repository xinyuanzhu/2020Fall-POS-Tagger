import torch
import torch.nn as nn
import torch.optim as optim


def log_sum_exp_batch(log_Tensor, axis=-1):
    return torch.max(log_Tensor, axis)[0] + \
        torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)
                            [0].view(log_Tensor.shape[0], -1, 1)).sum(axis))


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def bilstm_crf_acc(preds, y, tag_pad_idx):

    batch_size = y.shape[0]
    target_len = y.shape[1]
    pad_pred = []
    for sen in preds:
        sen += [tag_pad_idx]*(target_len - len(sen))
        pad_pred.append(sen)
    target_pred = torch.tensor(pad_pred)
    train_correct = ((target_pred == y)).sum()
    pad_eq = (y == tag_pad_idx).sum()
    correct = train_correct.item() - pad_eq.item()
    ratio = correct / (batch_size * target_len - pad_eq.item())
    return ratio


def categorical_accuracy(preds, y, tag_pad_idx):
    # calculate acc for bilstm
    max_preds = preds.argmax(
        dim=1, keepdim=True)
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)