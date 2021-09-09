import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss,BCELoss
def _getMatrixTree_multi(scores, root):
    A = scores.exp()
    R = root.exp()

    L = torch.sum(A, 1)
    L = torch.diag_embed(L)
    L = L - A
    LL = L + torch.diag_embed(R)


    LL_inv = torch.inverse(LL)  # batch_l, doc_l, doc_l
    LL_inv_diag = torch.diagonal(LL_inv, 0, 1, 2)
    d0 = R * LL_inv_diag
    LL_inv_diag = torch.unsqueeze(LL_inv_diag, 2)

    _A = torch.transpose(A, 1, 2)
    _A = _A * LL_inv_diag
    tmp1 = torch.transpose(_A, 1, 2)
    tmp2 = A * torch.transpose(LL_inv, 1, 2)

    d = tmp1 - tmp2
    return d, d0

class StructuredAttention(nn.Module):
    def __init__(self, config):
        self.model_dim = config.hidden_size

        super(StructuredAttention, self).__init__()

        self.linear_keys = nn.Linear(config.hidden_size, self.model_dim)
        self.linear_query = nn.Linear(config.hidden_size, self.model_dim)
        self.linear_root = nn.Linear(config.hidden_size, 1) #

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def forward(self, x,mask = None,roots_label=None,root_mask=None):

        key = self.linear_keys(x)
        query = self.linear_query(x)
        root= self.linear_root(x).squeeze(-1)

        query = query / math.sqrt(self.model_dim)
        scores = torch.matmul(query, key.transpose(1, 2))


        mask=mask.squeeze(1)/-10000
        root = root - mask.squeeze(1) * 50
        root = torch.clamp(root, min=-40)
        scores = scores - mask * 50
        scores = scores - torch.transpose(mask, 1, 2) * 50
        scores = torch.clamp(scores, min=-40)

        d, d0 = _getMatrixTree_multi(scores, root) # d0-> B,L   d->B,L,L


        if roots_label is not None:

            loss_fct=BCELoss(reduction='none')
            if root_mask is not None:

                active_loss = root_mask.view(-1) == 1

                active_logits = d0.view(-1)

                active_labels = torch.where(
                    active_loss, roots_label.view(-1), torch.tensor(0.).type_as(roots_label)
                )

                active_logits=torch.clamp(active_logits,1e-5,1 - 1e-5)


                loss_root = loss_fct(active_logits, active_labels)

                loss_root = (loss_root*root_mask.view(-1).float()).mean()

        attn = torch.transpose(d, 1,2)#
        if mask is not None:
            mask = mask.expand_as(scores).bool()

            attn = attn.masked_fill(mask, 0)

        context = torch.matmul(attn,x)

        return context, d0,loss_root
