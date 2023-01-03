import torch
import torch.nn.functional as F
import torch_geometric
#from torch_geometric.nn import GCNConv as GraphConv
from torch_geometric.nn import SAGEConv as GraphConv
from torch_geometric.utils import to_dense_batch
#from torch_geometric.utils import to_batch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool
import random
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.inits import uniform
from torch_geometric.utils.num_nodes import maybe_num_nodes
import pdb
from torch_geometric.nn import Set2Set


def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0][batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = torch.nonzero(x > scores_min).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm


def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class global_pool(torch.nn.Module):
    def __init__(self,nhid):
        super(global_pool,self).__init__()
        
        self.W = torch.nn.Parameter(torch.randn(nhid,nhid),requires_grad=True)
        
        torch.nn.init.xavier_normal_(self.W)


    def forward(self, x, batch, c_size):      
        x_sum = gap(x,batch)

        c = torch.tanh(torch.matmul(x_sum,self.W))  # context vector

        c = torch.cat([p.repeat(c_size[idx],1) for idx,p in enumerate(c)],dim=0)

        alpha = torch.sigmoid((x*c).sum(dim=1).view(-1,1))

        return global_add_pool(x*alpha, batch)

class CAGPool_layer(torch.nn.Module):
    def __init__(self, in_channels, ratio):
        super(CAGPool_layer, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio

    def forward(self, x, edge_index, batch, pool_vector, c_size): 

        x_ = torch.cat([p.repeat(c_size[idx],1) for idx,p in enumerate(pool_vector)],dim=0)

        score = torch.sum(x*x_,dim=1)/x_.norm(p=2, dim=1)
        
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        perm = topk(score, self.ratio, batch)

        x = x[perm] * torch.sigmoid(score[perm]).view(-1, 1)
        batch = batch[perm]

        edge_index, _ = filter_adj(
            edge_index, None, perm, num_nodes=score.size(0))


        return x, edge_index, batch, perm

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                                         self.in_channels)


class CAGpool(torch.nn.Module):
    def __init__(self, cfg):
        super(CAGpool, self).__init__()
        cfg = cfg['BASE']['MODEL']
        self.nhid = cfg['NUM_HIDDEN']
        self.nfeat = cfg['NUM_FEATURES']
        self.min_nodes = cfg['MIN_NODES']
        self.conv_ch = cfg['CONV_CHANNEL']

        self.num_layer = cfg['NUM_LAYER']
        self.K = cfg['K']
        self.ratio = cfg['POOL_RATIO']

        
        self.conv1 = GraphConv(self.nfeat, self.nhid)
        self.convs = torch.nn.ModuleList()
        self.convs.extend([GraphConv(self.nhid,self.nhid) for i in range(self.num_layer -1)])

        self.att_global_pool = global_pool(self.nhid*self.num_layer)

        self.att_lin = torch.nn.Linear(self.nhid*self.num_layer*2 + 964, self.nhid*self.num_layer*2)

        self.pool = CAGPool_layer(self.nhid*self.num_layer, self.ratio)

        self.final_conv = GraphConv(self.nhid*self.num_layer, self.nhid)
        
        self.final_global_pool = global_pool(self.nhid)

        #self.bilinear = torch.nn.Bilinear(self.nhid, self.nhid, self.K, bias=False)
        #self.V = torch.nn.Linear(self.nhid*2,self.K)
        #self.lin1 = torch.nn.Linear(self.K,self.K)
        #self.lin2 = torch.nn.Linear(self.K,int(self.K/2))
        #self.lin3 = torch.nn.Linear(int(self.K/2),1,bias=True)
        self.lin1 = torch.nn.Linear(self.nhid*2,self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid,int(self.nhid/2))
        self.lin3 = torch.nn.Linear(int(self.nhid/2),1)
        


    def forward(self, data):
        x, edge_index, batch, c1, c2, side_effect = data.x, data.edge_index, data.batch, data.c1_size, data.c2_size, data.side_effect
        batch_size = c1.shape[0]

        x = F.leaky_relu(self.conv1(x, edge_index),0.05)
        xs = [x]
        for i, conv in enumerate(self.convs):
            x = F.leaky_relu(conv(x,edge_index),0.05)
            xs += [x]
        x = torch.cat(xs,dim=1)

        c1_idx = []
        c2_idx = []
        c3 = [torch.LongTensor([0])]
        c3.extend(c1+c2)
        sum_c3 = 0
        for i in range(batch_size):
            sum_c3 += c3[i]
            c1_idx.extend(list(range(sum_c3,sum_c3+c1[i])))
            c2_idx.extend(list(range(sum_c3+c1[i],sum_c3+c1[i]+c2[i])))

        x_c1 = x[c1_idx]
        x_c2 = x[c2_idx]
        batch_c1 = batch[c1_idx]
        batch_c2 = batch[c2_idx]
        batch_ = torch.cat([batch_c1,batch_c2+batch_size],dim=0) # batch_ : concat of c1 batch & c2 batch
        
        #edge_index_c1 = torch_geometric.utils.dense_to_sparse(torch_geometric.utils.sparse_to_dense(edge_index)[c1_idx,:][:,c1_idx])[0]
        #edge_index_c2 = torch_geometric.utils.dense_to_sparse(torch_geometric.utils.sparse_to_dense(edge_index)[c2_idx,:][:,c2_idx])[0]
        #edge_index_c1 = torch_geometric.utils.dense_to_sparse(torch_geometric.utils.to_dense_adj(edge_index)[c1_idx,:][:,c1_idx])[0]
        #edge_index_c2 = torch_geometric.utils.dense_to_sparse(torch_geometric.utils.to_dense_adj(edge_index)[c2_idx,:][:,c2_idx])[0]
        
        num_nodes = sum(c1)+sum(c2)
        edge_index_c1 = filter_adj(edge_index,None,torch.LongTensor(c1_idx),num_nodes)[0]
        edge_index_c2 = filter_adj(edge_index,None,torch.LongTensor(c2_idx),num_nodes)[0]
        
        global_pool_c = self.att_global_pool(torch.cat([x_c1,x_c2],dim=0), batch_, torch.cat([c1,c2],dim=0))
        global_pool_c1 = global_pool_c[:batch_size,:]
        global_pool_c2 = global_pool_c[batch_size:,:]

        pool_vector = self.att_lin(torch.cat([global_pool_c1,global_pool_c2,side_effect],dim=1))
        pool_vector_for_c1 = pool_vector[:,:self.nhid*self.num_layer]
        pool_vector_for_c2 = pool_vector[:,self.nhid*self.num_layer:]

        pooled_c1, edge_index_c1, batch_c1, perm_c1 = self.pool(x_c1, edge_index_c1, batch_c1, pool_vector_for_c1, c1)
        pooled_c2, edge_index_c2, batch_c2, perm_c2 = self.pool(x_c2, edge_index_c2, batch_c2, pool_vector_for_c2, c2)

        num_pooled_c1 = scatter_add(torch.ones(batch_c1.shape[0]).to(batch.device),batch_c1).long()
        num_pooled_c2 = scatter_add(torch.ones(batch_c2.shape[0]).to(batch.device),batch_c2).long()

        x_c1 = F.leaky_relu(self.final_conv(pooled_c1, edge_index_c1),0.05)
        x_c2 = F.leaky_relu(self.final_conv(pooled_c2, edge_index_c2),0.05)

        x_c1 = self.final_global_pool(x_c1, batch_c1, num_pooled_c1)
        x_c2 = self.final_global_pool(x_c2, batch_c2, num_pooled_c2)

        #pred = F.leaky_relu(self.bilinear(x_c1,x_c2) + self.V(torch.cat([x_c1,x_c2],dim=1)),0.05)

        pred = torch.cat([x_c1,x_c2],dim=1)        
        
        pred = F.leaky_relu(self.lin1(pred),0.05)
        pred = F.leaky_relu(self.lin2(pred),0.05)
        pred = self.lin3(pred)
        pred = pred.view(-1, 1)

        return batch_, None, pred

