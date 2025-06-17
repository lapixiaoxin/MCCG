import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M, concat=True):
        h = torch.mm(input, self.W)

        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, alpha=0.2):
        super(GAT, self).__init__()
        self.conv1 = GATLayer(dim_input, dim_hidden, alpha)
        self.conv2 = GATLayer(dim_hidden, dim_output, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = F.dropout(h, 0.6, training=self.training)
        h = self.conv2(h, adj, M)

        return h


class MCCG(nn.Module):
    def __init__(self, encoder, dim_hidden, dim_proj_multiview, dim_proj_cluster):
        super(MCCG, self).__init__()
        self.encoder = encoder
        self.multiview_projector = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ELU(),
            nn.Linear(dim_hidden, dim_proj_multiview)
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ELU(),
            nn.Linear(dim_hidden, dim_proj_cluster)
        )
        self.project = nn.Linear(dim_proj_cluster, 32)
        self.MLP = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1, adj1, M1, x2, adj2, M2):
        z1 = self.encoder(x1, adj1, M1)
        z2 = self.encoder(x2, adj2, M2)
        z = (z1 + z2) / 2

        z_view1 = F.normalize(self.multiview_projector(z1), dim=1)
        z_view2 = F.normalize(self.multiview_projector(z2), dim=1)
        z_cluster = F.normalize(self.cluster_projector(z), dim=1)
        z_multiview = torch.cat([z_view1.unsqueeze(1), z_view2.unsqueeze(1)], dim=1)

        return z_multiview, z_cluster

    def SelfSupConLoss(self, features, labels=None, mask=None, temperature=0.2, contrast_mode='all'):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = exp_logits * logits_mask
        if contrast_mode == "one":
            w_anchor = self.project(anchor_feature)
            w = w_anchor
            N = w_anchor.size(0)
            w_anchor = w_anchor.unsqueeze(1).repeat(1, N, 1).reshape(N * N, -1)
            w = w.unsqueeze(0).repeat(N, 1, 1).reshape(N * N, -1)
            weight = w_anchor + w
            weight = self.MLP(weight).reshape(N, N)
            weight = weight / temperature

            exp_logits = torch.mul(exp_logits, weight)
            logits = exp_logits

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -mean_log_prob_pos / temperature
        loss = loss.view(anchor_count, batch_size).mean()

        return loss