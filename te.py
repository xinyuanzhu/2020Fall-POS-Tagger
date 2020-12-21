import torch


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


a = [[1, 2, 3], [1]]
b = torch.tensor([[1, 3, 3, 0], [1, 0, 0, 0]])


bilstm_crf_acc(a, b, 0)
