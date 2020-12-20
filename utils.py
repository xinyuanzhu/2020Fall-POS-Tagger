import torch


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_batch(log_Tensor, axis=-1):  # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0] + \
        torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)
                            [0].view(log_Tensor.shape[0], -1, 1)).sum(axis))
