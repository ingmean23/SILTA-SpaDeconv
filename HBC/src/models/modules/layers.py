import math
import torch
import torch.nn.init as init
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
try:
    import dgl
    import dgl.nn as dglnn
except ImportError:  # SILTA HBC does not use the legacy DGL block.
    dgl = None
    dglnn = None


class GumbelSoftmax(nn.Module):

  def __init__(self, f_dim, c_dim):
    super(GumbelSoftmax, self).__init__()
    self.logits = nn.Linear(f_dim, c_dim)
    self.f_dim = f_dim
    self.c_dim = c_dim
     
  def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, logits, temperature):
    y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
    return F.softmax(y / temperature, dim=-1)

  def gumbel_softmax(self, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard 
  
  def forward(self, x, temperature=1.0, hard=False):
    logits = self.logits(x).view(-1, self.c_dim)
    prob = F.softmax(logits, dim=-1)
    y = self.gumbel_softmax(logits, temperature, hard)
    return logits, prob, y


class Gaussian(nn.Module):
  def __init__(self, in_dim, z_dim):
    super(Gaussian, self).__init__()
    self.mu = nn.Linear(in_dim, z_dim)
    self.var = nn.Linear(in_dim, z_dim)

  def reparameterize(self, mu, var):
    std = torch.sqrt(var + 1e-10)
    noise = torch.randn_like(std)
    z = mu + noise * std
    return z      

  def forward(self, x):
    mu = self.mu(x)
    var = F.softplus(self.var(x))
    z = self.reparameterize(mu, var)
    return mu, var, z


class GraphAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        use_bias=True,
    ):
        """ Attention block performed on graph.
        
        Args:
            input_dim (int): input dimension of embedding.
            output_dim (int): output dimension of embedding.
            num_heads (int): number of heads.
            use_bias (boolean): whether to use bias in attention block.

        Attrs:
            Q (nn.Linear): Query matrix.
            K (nn.Linear): Keys matrix.
            V (nn.Linear): Values matrix.
        """
        super(GraphAttentionBlock, self).__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dim_per_head = output_dim // num_heads

        self.Q = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.K = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.V = nn.Linear(input_dim, output_dim, bias=use_bias)

    def propagate_graph_attention(self, Q_h, K_h, V_h, adj_matrix):
        """ This function performs for the aggregation of graph network based on the
        output of single head Q_h, K_h, V_h, adjacent matrix, mask matrix.

        Args:
            Q_h (tensor): Querys, dimention is (B, N, D);
            K_h (tensor): Keys, dimention is (B, N, D);
            V_h (tensor): Values, dimention is (B, N, D);
            adj_matrix (tensor): normalized adjacent matirx, dimension is (B, N, N);

        Returns:
            h (tensor): calculation results of self-attention mechanisms.
        """
        K_h_transpose = torch.transpose(K_h, 1, 2) # dim B, D, N
        score = torch.matmul(Q_h, K_h_transpose) / np.sqrt(self.dim_per_head)
        # masked attention: non-neighbors get -inf so softmax ignores them
        mask = (adj_matrix == 0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        score = score.masked_fill(mask, float('-inf'))
        attention_score = nn.Softmax(dim=2)(score) # dim B, N, N
        # safety: if a row is all -inf, softmax produces nan → replace with 0
        attention_score = torch.nan_to_num(attention_score, nan=0.0)
        attention_output = torch.matmul(attention_score, V_h) # dim B, N, D
        return attention_output

    def forward(self, h, adj_matrix):
        """forward function of Attention block.

        Args:
            h (tensor): input of graph tensor, dimension is (B, N, D);
            adj_matrix (tensor): normalized adjacent matrix;

        Returns:
            output (tensor): result from the multi-head attention block.
        """
        B, N, D = h.shape
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        Q_multi_head_h = Q_h.view(B, N, self.num_heads, self.dim_per_head) # dim B, N, H, D
        K_multi_head_h = K_h.view(B, N, self.num_heads, self.dim_per_head) # dim B, N, H, D
        V_multi_head_h = V_h.view(B, N, self.num_heads, self.dim_per_head) # dim B, N, H, D
        
        heads_output_list = []
        for i in range(self.num_heads):
            single_head_output = self.propagate_graph_attention(
                Q_h=Q_multi_head_h[:, :, i],
                K_h=K_multi_head_h[:, :, i],
                V_h=V_multi_head_h[:, :, i],
                adj_matrix=adj_matrix,
            ) # dim B, N, D
            heads_output_list.append(single_head_output)
        output = torch.cat(heads_output_list, dim=2) # dim B, N, D*H
        return output


class GraphTransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        use_bias=True,
    ):
        """ GraphTransformer block integrated attention.
        TODO: could further add resdual strategy in transformer block.

        Args:
            input_dim (int): input dimension of embedding.
            output_dim (int): output dimension of embedding.
            num_heads (int): number of heads.
            use_bias (boolean): whether to use bias in attention block.

        Attrs:
            attention (GraphAttentionBlock): compute the graph embedding using
                attention mechanism.
            batch_norm (nn.BatchNorm1d): normalization.
        """
        super(GraphTransformerBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.attention = GraphAttentionBlock(input_dim, output_dim, num_heads, use_bias)

        # optional
        self.projection_layer = nn.Linear(output_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, h, adj_matrix):
        """forward function of transformer block, support one block of attention for now.
        
        Args:
            h (tensor): input of graph tensor, dimension is (B, N, D);
            adj_matrix (tensor): normalized adjacent matrix;

        Returns:
            attention_out (tensor): result from transformer block.
        """
        attention_output = self.attention(h, adj_matrix)
        attention_output = F.relu(attention_output)
        # attention_output = self.batch_norm(attention_output) # one graph per batch.
        return attention_output


class GraphConvolutionBlock(nn.Module):
    """GraphConvolution Block which performs graph convolution when encoding.

    Args:
        in_features (int): input feature dimension.
        out_features (int): output feature dimension.
        use_bias (boolean): whether use bias in when calculating.
    """
    def __init__(
        self,
        in_features,
        out_features,
        use_bias=True
    ):
        super(GraphConvolutionBlock, self).__init__()
        if dglnn is None:
            raise ImportError(
                "DGL is required only for the legacy GraphConvolutionBlock; "
                "the released SILTA HBC checkpoint does not use this block."
            )
        self.model = dglnn.GraphConv(in_feats=in_features, out_feats=out_features,bias=use_bias,activation=F.relu)
        # self.in_features = in_features
        # self.out_features = out_features
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.gcn = dglnn.SAGEGraph()
        # if use_bias:
        #     self.bias = Parameter(torch.FloatTensor(out_features))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = torch.squeeze(input)
        # 将邻接矩阵转换为 DGL 图
        if not isinstance(adj, dgl.DGLGraph):
            adj_sq = adj.squeeze() if adj.dim() > 2 else adj
            src, dst = torch.nonzero(adj_sq, as_tuple=True)
            g = dgl.graph((src, dst), num_nodes=adj_sq.shape[0])
            g = g.to(input.device)
        else:
            g = adj
        output = self.model(g, input)
        output = torch.unsqueeze(output, 0)
        return output


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * 1.0
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


class Scaler(nn.Module):
    """特殊的scale层，用于防止KL散度消失
    mode='positive': scale = tau + (1-tau) * sigmoid(s)，范围[tau, 1]，防止均值坍缩
    mode='negative': scale = (1-tau) * sigmoid(-s)，范围[0, 1-tau]，控制方差/离散度

    Args:
        dim: 特征维度
        tau: 缩放下界，默认0.5
    """
    def __init__(self, dim, tau=0.4):
        super().__init__()
        self.tau = tau
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, inputs, mode='positive'):
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        return inputs * torch.sqrt(scale)


class ZeroInflatedNegativeBinomial(nn.Module):
    """零膨胀负二项分布(ZINB)模块，替代原Gaussian分布
    编码器输出经 Linear → BN → Scaler → 激活 得到ZINB参数，
    BN + Scaler 防止KL散度消失并稳定训练。

    Args:
        in_dim: 输入特征维度（编码器输出维度）
        z_dim: 潜变量维度（ZINB输出维度）
        tau: Scaler的缩放下界，默认0.5
    """

    def __init__(self, in_dim, z_dim, tau=0.5):
        super().__init__()
        # 预测ZINB的三个核心参数：π(零膨胀概率)、μ(均值)、θ(离散度)
        self.pi_layer = nn.Linear(in_dim, z_dim)
        self.mu_layer = nn.Linear(in_dim, z_dim)
        self.theta_layer = nn.Linear(in_dim, z_dim)

        # BN层（affine=False 对应 Keras 的 scale=False, center=False）
        self.bn_pi = nn.BatchNorm1d(z_dim, affine=False, eps=1e-8)
        self.bn_mu = nn.BatchNorm1d(z_dim, affine=False, eps=1e-8)
        self.bn_theta = nn.BatchNorm1d(z_dim, affine=False, eps=1e-8)

        # Scaler层
        self.scaler_pi = Scaler(z_dim, tau=tau)
        self.scaler_mu = Scaler(z_dim, tau=tau)
        self.scaler_theta = Scaler(z_dim, tau=tau)

    def forward(self, x):
        """前向传播：Linear → BN → Scaler → 激活
        Returns:
            pi: 零膨胀概率 (B, N, z_dim)
            mu: 负二项均值 (B, N, z_dim)
            theta: 负二项离散度 (B, N, z_dim)
            z: 连续潜变量，即mu (B, N, z_dim)
        """
        shape = x.shape  # (B, N, in_dim)

        # 1. Linear
        pi_raw = self.pi_layer(x)
        mu_raw = self.mu_layer(x)
        theta_raw = self.theta_layer(x)

        # 2. BN（reshape: (B,N,D) → (B*N,D) → BN → (B,N,D)）
        out_shape = pi_raw.shape
        pi_raw = self.bn_pi(pi_raw.reshape(-1, out_shape[-1])).reshape(out_shape)
        mu_raw = self.bn_mu(mu_raw.reshape(-1, out_shape[-1])).reshape(out_shape)
        theta_raw = self.bn_theta(theta_raw.reshape(-1, out_shape[-1])).reshape(out_shape)

        # 3. Scaler（mu用positive防止坍缩，pi/theta用negative控制范围）
        mu_raw = self.scaler_mu(mu_raw, mode='positive')
        pi_raw = self.scaler_pi(pi_raw, mode='negative')
        theta_raw = self.scaler_theta(theta_raw, mode='negative')

        # 4. 激活函数约束参数范围（clamp 防止 exp 溢出）
        pi = torch.sigmoid(pi_raw)                              # 0 <= pi <= 1
        mu = torch.exp(torch.clamp(mu_raw, min=-10, max=10))   # mu > 0, 上限 e^10 ≈ 22026
        theta = F.softplus(torch.clamp(theta_raw, min=-10, max=10))  # theta > 0

        # 5. 使用mu作为连续潜变量（可微分，类似scVI做法）
        z = mu
        return pi, mu, theta, z

    def log_likelihood(self, x, pi, mu, theta, eps=1e-10):
        """计算ZINB对数似然（用于VAE的重构损失）
        Args:
            x: 原始输入数据 (B, N, z_dim)
            pi/mu/theta: ZINB参数（同forward输出）
        """
        mu = torch.clamp(mu, min=eps, max=1e6)
        theta = torch.clamp(theta, min=eps, max=1e6)
        x = torch.clamp(x, min=0)

        # 1. 负二项分布的对数概率
        log_nb = (
                torch.lgamma(x + theta)
                - torch.lgamma(theta)
                - torch.lgamma(x + 1)
                + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
                + x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
        )

        # 2. 零膨胀部分的对数概率
        nb_zero_prob = torch.pow(theta / (theta + mu + eps), theta)
        log_zero = torch.log(pi + (1 - pi) * nb_zero_prob + eps)
        log_non_zero = torch.log(1 - pi + eps) + log_nb

        # 3. 结合零/非零情况的对数似然
        ll = torch.where(x < 0.5, log_zero, log_non_zero)
        return ll

if __name__ == "__main__":
    ## number of nodes: 200, number of features: 512
    x_genes = torch.rand((1, 100, 512))
    adj = torch.randint(0, 2, (100,100)).float()
    # adj = adj.fill_diagonal_(1.)
    gcn = GraphConvolutionBlock(512, 200)
    gat = GraphTransformerBlock(512, 200, 4)
    src, dst = np.nonzero(adj).T
    g = dgl.graph((src, dst))
    y_gcn = gcn(x_genes, g)
    y_gat = gat(x_genes, adj)
    print(y_gcn.shape)
    print(y_gat.shape)


