#!/usr/bin/env python
import torch.utils.checkpoint as checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.utils import mkldnn as mkldnn_utils
from torch_geometric.nn import (DynamicEdgeConv, EdgeConv, EdgePooling, GCN2Conv,
                                GCNConv, MessagePassing)
from torch_geometric.utils import add_self_loops, degree

from .graph_utils import *


class EdgeConvNet(GraphNetwork):
    def __init__(self,
                 gene_embedding=256,
                 hidden_channels=[256, 512, 512, 1024],
                 ks=[None, 4, 3, 2],
                 embed_dim=EMBED_DIM,
                 resnet_model='resnet18',
                 dropout=.1,
                 final_dim=None,
                 graph_batch_norm=False,
                 output='metric',
                 device='cpu', **kwargs):
        '''Embedding layer followed by resnet layer followed by a stack of num_layers EdgeConv or DynamicEdgeConv layers'''
        super().__init__()
        self.dims = [gene_embedding] + hidden_channels
        self.ks = ks
        self.device = device

        self.embed = EmbedLayer(embed_dim, dropout)
        if resnet_model is not None:
            self.gene_encoder = resnet_models[resnet_model](in_channels=embed_dim,
                                                            n_classes=gene_embedding)
        else:
            self.gene_encoder = nn.Flatten()
            # gene_embedding = embed_dim

        self.elu = nn.functional.elu
        self.relu = nn.functional.relu

        self.norm1 = nn.BatchNorm1d(gene_embedding)
        last_layer_dim = hidden_channels[-1]
        layers = [2*gene_embedding]+hidden_channels
        edge_mapper = build_fc_network(
            layers=layers, batch_norm=graph_batch_norm)
        layers = [2*hidden_channels[0]]+hidden_channels[1:]

        self.graph_layer = nn.Sequential()
        for i, k in enumerate(ks):
            if k is not None:
                edge_gcn = DynamicEdgeConv(edge_mapper, k=k, aggr='mean')
            else:
                k = 'static'
                edge_gcn = EdgeConv(edge_mapper, aggr='mean')

            self.graph_layer.add_module(f'{k}-EdgeConv_{i}', edge_gcn)
            edge_mapper = build_fc_network(layers, batch_norm=graph_batch_norm)

        if final_dim is None:
            final_dim = last_layer_dim
        self.final_layer = build_fc_network(
            last_layer_dim,
            final_dim,
            graph_batch_norm)
        if output == 'distance':
            self.output_layer = MetricDecoder(clip=1e6, as_list=False)
        elif output == 'covariance':
            self.output_layer = CovarianceDecoder(clip=1e6, as_list=False)

        self.set_devices(device)

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        x = data.x
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x)
        x = self.embed(x).to(self.device)
        x = self.gene_encoder(x)
        x = self.elu(x).squeeze()  # hack to avoid rewriting ResNet
        for name, layer in self.graph_layer.named_children():
            if name.startswith('static'):
                x = layer(x, edge_index)
            else:
                x = layer(x, batches, num_workers=3)
            x = self.elu(x)
        x = self.final_layer(x)
        x = self.elu(x)
        x = self.output_layer(x, batches)  # no trainable params
        return x


class ResidualEdgeConvNet(EdgeConvNet):
    def __init__(self, *args, **kwargs):
        '''remembers intermediate activations for final FC layer'''
        super().__init__(*args, **kwargs)
        self.final_layer = build_fc_network(
            sum(self.dims),
            self.dims[-1],
            self.edge_net_depth)

        self.set_devices(self.device)

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        x = data.x
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x)
        x = self.embed(x).to(self.device)
        x = self.gene_encoder(x)
        x = self.elu(x).squeeze()  # hack to avoid rewriting ResNet
        activations = [x]
        for name, layer in self.graph_layer.named_children():
            if name.startswith('full'):
                x = layer(x, edge_index)
            else:
                x = layer(x)
            activations.append(x)
            x = self.relu(x)
        x = self.final_layer(torch.cat(activations, dim=1))
        x = self.relu(x)
        x = self.output_layer(x, batches)  # no trainable params
        return x


class TransposeLayer(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.transpose(x, 1, -1)


class RNNEncoder(Module):
    dropout = .1
    hidden_size = 64

    def __init__(self, num_rnn_layers, pretransform, pretransform_output_dim, output_size):
        super().__init__()

        self.pretransform = pretransform
        self.rnn = nn.LSTM(
            input_size=pretransform_output_dim,
            hidden_size=self.hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout
        )  # output,(h_n,c_n), c_n and h_n have shape(num_layers * 2, batch, hidden_size)
        num_features = 4 * num_rnn_layers * self.hidden_size

        self.norm5 = nn.BatchNorm1d(num_features)
        self.dense = nn.Linear(
            in_features=num_features,
            out_features=output_size
        )

    def forward(self, x):
        x = self.pretransform(x)

        with autocast(enabled=False):
            x = torch.transpose(x, -2, -1).float()
            _, hidden = self.rnn(x)
        x = torch.cat(hidden)  # seq embedding.
        # shape after transpose: batch, num_layers * num_directions, hidden_size
        x = torch.transpose(x, 0, 1)
        x = torch.flatten(x, 1, -1).relu()

        x = self.norm5(x)
        # NOTE: if we add layers before here, need to transfer earlier
        return self.dense(x).relu()

        with autocast(enabled=False):
            x = torch.transpose(x, -2, -1).float()
            _, hidden = self.rnn(x)
        x = torch.cat(hidden)  # seq embedding.
        # shape after transpose: batch, num_layers * num_directions, hidden_size
        x = torch.transpose(x, 0, 1)
        x = torch.flatten(x, 1, -1).relu()

        x = self.norm5(x)
        # NOTE: if we add layers before here, need to transfer earlier
        return self.dense(x).relu()


class GraphEncoder(Module):
    dropout = .1

    def __init__(self,
                 graph_features,
                 num_subgraphs,
                 return_edges=False,
                 input_size=128,
                 ):
        super().__init__()

        self.edge_pool_layers = []
        for i in range(num_subgraphs):
            m = EdgePooling(
                input_size,
                dropout=self.dropout,
                edge_score_method=EdgePooling.compute_edge_score_softmax
            )
            self.add_module(
                name=f'edge_pool{i}',
                module=m
            )
            self.edge_pool_layers.append(m)

        self.gcn_layers = []
        for i in range(num_subgraphs):
            m = GCNConv(
                input_size,
                graph_features,
                normalize=True,
            )
            self.add_module(name=f'edge_gcn{i}',
                            module=m)
            self.gcn_layers.append(m)

        self.edge_dense = nn.Linear(
            2 * (input_size + num_subgraphs * graph_features),
            input_size
        )
        self.output_layer = nn.Linear(
            input_size,
            1)

        self.return_edges = return_edges

    def forward(self, x, e, b):
        with autocast(enabled=False):
            batch_index = b
            new_subgraph = e.clone()
            # subgraphs = [new_subgraph]
            new_x = x.float()
            x_pooled = [x]
            unpool_info = []
            # n_nodes = len(x)
            for i, (edge_pool, gcn) in enumerate(zip(self.edge_pool_layers, self.gcn_layers)):
                new_x, new_subgraph, batch_index, unpool = edge_pool(
                    new_x,
                    new_subgraph,
                    batch_index)
                new_x = new_x.relu()
                x_gcn = gcn(new_x, new_subgraph)
                unpool_info.append(unpool)
                for j in range(i, -1, -1):
                    x_gcn, _, _ = self.edge_pool_layers[j].unpool(
                        x_gcn,  unpool_info[j])
                x_pooled.append(x_gcn)
                # subgraphs.append(new_subgraph + n_nodes+i)

        x_pooled = torch.cat(x_pooled, 1).relu()
        # subgraphs=torch.cat(subgraphs, 1)
        # x=self.gcn(x_pooled, subgraphs).relu()

        x_j = torch.index_select(x_pooled, 0, e[0])
        x_i = torch.index_select(x_pooled, 0, e[1])
        edge_features = torch.cat((x_i, x_j),  dim=1)
        preds = self.edge_dense(edge_features).relu()
        with autocast(enabled=False):
            preds = self.output_layer(preds.float())  # .relu().squeeze()
        if self.return_edges:
            return e, preds
        else:
            preds = array_to_mat(e, preds).squeeze()
            preds = preds.mul(preds.T)
            return preds


class KnnGraphEncoder(Module):

    def __init__(self,
                 ks=[5, 3, 1],
                 in_channels=10,
                 channels=10,
                 ):

        super().__init__()

        ks = sorted(ks)
        m = DynamicEdgeConv(
            nn.Linear(2*in_channels, channels), ks[0], num_workers=2)
        self.gcn_layers = nn.ModuleList([m])

        for k in ks[1:]:
            m = DynamicEdgeConv(nn.Linear(2*channels, channels), k)
            self.gcn_layers.append(m)

        num_features = 2 * (len(ks) * channels + in_channels)

        self.relu = nn.functional.relu

        self.norm1 = nn.BatchNorm1d(num_features)
        self.edge_dense = nn.Linear(
            num_features,
            channels
        )
        self.norm2 = nn.BatchNorm1d(channels)
        self.hidden_layer = nn.Linear(
            channels,
            channels)
        self.output_layer = nn.Linear(
            channels,
            1)

    def forward(self, x, e):
        with autocast(enabled=False):
            new_x = x
            x_pooled = [new_x]
            for gcn in self.gcn_layers:
                new_x = gcn(new_x)
                # new_x = new_x.relu()
                x_pooled.append(new_x)

        x_pooled = torch.cat(x_pooled, 1)
        x_pooled = self.relu(x_pooled)

        x_j = torch.index_select(x_pooled, 0, e[0])
        x_i = torch.index_select(x_pooled, 0, e[1])
        edge_features = torch.cat((x_i, x_j),  dim=1)

        edge_features = self.norm1(edge_features)
        preds = self.edge_dense(edge_features)
        preds = self.relu(preds)

        preds = self.norm2(preds)
        # with autocast(enabled=False):
        preds = self.hidden_layer(preds)
        preds = self.relu(preds)

        preds = self.output_layer(preds).squeeze()  # .relu().squeeze()
        n = x.size(0)
        preds = array_to_mat(e, preds, size=(n, n))
        preds = preds+preds.T
        return preds


class EdgeGCN(Module):
    num_filters = 2048
    dim_rnn_output = 256
    stride = 3

    def __init__(self,
                 hidden_channels,
                 num_rnn_layers,
                 num_subgraphs,
                 device=None,
                 kernel_size=64,
                 seed=12345,
                 return_edges=False):
        super().__init__()
        torch.manual_seed(seed)
        self.device = device

        dim1 = self.num_filters
        dim2 = self.num_filters//4
        dim3 = self.num_filters//8
        dim4 = self.num_filters//16

        kernel_size_2 = kernel_size//2
        kernel_size_3 = kernel_size//4
        kernel_size_4 = kernel_size//8

        conv1 = nn.Conv1d(
            in_channels=EMBED_DIM,
            out_channels=dim1,
            kernel_size=kernel_size,
            stride=self.stride,
        )

        norm1 = nn.BatchNorm1d(dim1)
        conv2 = nn.Conv1d(
            in_channels=dim1,
            out_channels=dim2,
            kernel_size=kernel_size_2,
            stride=self.stride,
        )

        norm2 = nn.BatchNorm1d(dim2)

        conv3 = nn.Conv1d(
            in_channels=dim2,
            out_channels=dim3,
            kernel_size=kernel_size_3,
            stride=self.stride,
        )

        norm3 = nn.BatchNorm1d(dim3)
        conv4 = nn.Conv1d(
            in_channels=dim3,
            out_channels=dim4,
            kernel_size=kernel_size_4,
            stride=self.stride,
        )

        norm4 = nn.BatchNorm1d(dim4)

        conv_block1 = nn.Sequential(
            conv1, norm1, conv2, norm2, )

        conv_block2 = nn.Sequential(
            conv3, norm3, conv4, norm4)

        # add layers

        self.embed = nn.Embedding(N_STATES, EMBED_DIM)
        self.conv_block1 = conv_block1

        self.rnn_encoding = RNNEncoder(
            num_rnn_layers,
            pretransform=conv_block2,
            pretransform_output_dim=dim4,
            output_size=self.dim_rnn_output)

        self.graph_encoding = GraphEncoder(
            hidden_channels, num_subgraphs, input_size=self.dim_rnn_output)

        if device is not None:
            self.to(device)
        self.embed.to('cpu')
        self.conv_block1.to('cpu')

    @property
    def cpu_params(self):
        return chain.from_iterable(l.parameters() for n, l in self.named_children() if n == 'embed')

    @property
    def gpu_params(self):
        return chain.from_iterable(l.parameters() for n, l in self.named_children() if n != 'embed')

    def custom(self, module):
        '''use for checkpointing large modules'''
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def sequence_encoding(self, x_input):
        x = self.embed(x_input)
        x = torch.transpose(x, 1, -1)
        x = self.conv_block1(x).relu()
        # x = checkpoint.checkpoint(self.custom(self.conv_block1), x).relu()

        return x.to(self.device)

    def set_devices(self, device=None):
        if device is not None:
            self.device = device
        for name, layer in self.named_children():
            layer.to(self.device)

    def dispatch_data(self, data):
        '''unpack Batch object and send to devices'''
        # data = data.to(self.device, non_blocking=True)
        x = data.x.long().cpu()  # .to(torch.int8)  # .long()
        e = data.edge_index.to(self.device, non_blocking=True)
        b = data.batch.to(self.device, non_blocking=True)
        return x, e, b

    def forward(self, data):
        '''input shape is [batches, ntaxa, alignment_length]'''

        # first two layers are largest; compute on cpu.
        # with autocast(enabled=False):
        # x = self.embed(x_input)
        x_input, edge_index, batch_indices = self.dispatch_data(data)

        x = self.sequence_encoding(x_input)
        # x = checkpoint.checkpoint(self.custom(self.sequence_encoding), x_input)
        # x = checkpoint.checkpoint(self.custom(self.rnn_encoding), x)
        x = self.rnn_encoding(x)

        x = self.graph_encoding(x, edge_index, batch_indices)

        return torch.block_diag(x)


class ModelParallelEdgeGCN(EdgeGCN):
    def __init__(self, devices=['cuda:0', 'cuda:1'], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dev1, self.dev2 = devices
        self.set_devices()

    def set_devices(self, devices=None):
        if devices is not None:
            self.dev1, self.dev2 = devices

        for name, layer in self.named_children():
            # if name in ('embed', ):  # 'conv1'
            #     layer = layer.to(self.devices[0])
            #     self._gpu_layers.append(layer)
            #     # layer = layer.to('cpu')
            #     # self._cpu_layers.append(layer)
            #     # layer = mkldnn_utils.to_mkldnn(layer)
            if name in ('embed', 'conv_block1'):  # , 'conv3', 'norm3',
                layer.to(self.dev1)
            else:
                layer.to(self.dev2)

    def dispatch_data(self, data):
        '''unpack Batch object and send to devices'''
        x = data.x.long().to(self.dev1, non_blocking=True)
        e = data.edge_index.to(self.dev2, non_blocking=True)
        b = data.batch.to(self.dev2, non_blocking=True)
        return x, e, b

    def forward(self, splits):
        '''input shape is [batches, ntaxa, alignment_length].  This '''
        # first two layers are largest; compute on cpu.
        # with autocast(enabled=False):
        x_next, e_prev, b_prev = self.dispatch_data(splits[0])
        x_prev = (self.sequence_encoding(x_next)
                      .to(self.dev2, non_blocking=True))
        x_prev = self.rnn_encoding(x_prev)

        # TODO: make sure this is thread-safe.  separating e_next from e_prev & b_next from b_prev should do this
        ret = []
        for x_next, e_next, b_next in map(self.dispatch_data, splits[1:]):
            # A. s_prev runs on cuda:1
            x_prev = self.graph_encoding(x_prev, e_prev, b_prev)
            ret.append(x_prev)

            # B. s_next runs on cuda:0, which can run concurrently with A
            x_prev = (self.sequence_encoding(x_next)
                      .to(self.dev2, non_blocking=True))
            x_prev = self.rnn_encoding(x_prev)
            e_prev, b_prev = e_next, b_next

            torch.cuda.empty_cache()

        x_prev = self.graph_encoding(x_prev, e_prev, b_prev)
        ret.append(x_prev)

        return torch.block_diag(*ret)


class DeepSet(Module):

    def __init__(self, in_channels, out_channels, hidden_size=25, num_layers=4):
        super().__init__()
        layers = [nn.Linear(in_channels, hidden_size), nn.ReLU(inplace=False)]

        for _ in range(num_layers):
            layers.extend(
                [nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=False)])

        layers.append(nn.Linear(hidden_size, out_channels))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # assumes shape is [n_batches,n_genes,n_features] sum along gene dim
        return self.fc(x).sum(-2)


class LocusGCN(Module):
    max_genes = 1000

    def __init__(self,
                 gene_embedding=20,
                 genome_embedding=200,
                 graph_embedding=50,
                 num_layers=2,
                 shared_weights=False,
                 device='cpu'):
        super().__init__()
        self.embed = nn.Embedding(N_STATES, EMBED_DIM)
        self.gene_encoder = ResNet(block=ResidualBlock,
                                   layers=[
                                       2, 4, 3],
                                   num_classes=gene_embedding,
                                   input_dim=EMBED_DIM)
        self.set_encoder = DeepSet(in_channels=gene_embedding,
                                   out_channels=genome_embedding)  # TODO: implement
        self.relu = nn.functional.relu
        self.norm1 = nn.BatchNorm1d(genome_embedding)

        # self.convs = nn.ModuleList()
        # for layer in range(num_layers):
        #     self.convs.append(
        #         GCN2Conv(genome_embedding,
        #                  alpha=.1,
        #                  theta=.1,
        #                  layer=layer + 1,
        #                  shared_weights=shared_weights,
        #                  normalize=False))
        # self.norm2 = nn.BatchNorm1d(genome_embedding)

        # TODO: copy edge gcn from above
        # TOD: try using several  DynamicEdgeConv with different values of k
        self.output_layer = KnnGraphEncoder(
            in_channels=genome_embedding,
            channels=graph_embedding)

    def set_devices(self, device=None):
        if device is not None:
            self.device = device
        for name, layer in self.named_children():
            if name == 'embed':  # name == 'gene_encoder' or name == 'set_encoder':
                layer.to('cpu')
            else:
                layer.to(device)

    def genome_encoding(self, x):
        '''forward pass thru resnet+set_encoder'''
        x = self.embed(x).transpose(-1, -3).to(self.device)
        x = self.gene_encoder(x)
        x = self.relu(x)
        x = self.set_encoder(x)
        return x

    def graph_encoding(self, x_0, A):
        '''forward pass thru GNN with residual connections'''
        x = x_0
        edge_weight = torch.ones(A.size(1))
        for conv in self.convs:
            h = conv(x, x_0, A, edge_weight=edge_weight)
            x = self.relu(h + x)
        return x

    def forward(self, data):
        '''input shape is single batch of [ntaxa, ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)

        # TODO: can this be sped up?
        x = torch.cat([self.genome_encoding(x_batch.long())
                       for x_batch in data.x])  # .to(self.device)
        x = self.relu(x)
        x = self.norm1(x)
#        x = self.graph_encoding(x, edge_index)
#        x = self.norm2(x)
        x = self.output_layer(x, edge_index)  # output must be positive
        x = self.relu(x)
        return x


class DilatedConvNet(Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
