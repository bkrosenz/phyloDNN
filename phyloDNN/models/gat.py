from typing import Iterable
from sklearn import datasets
import torch.nn.functional as F
from torch import nn
import torch_geometric.utils as gm_utils
from torch_geometric.nn import GATv2Conv, GraphMultisetTransformer, SuperGATConv, pool, global_mean_pool
from .edge_conv import *
from .graph_utils import *


class GAE(nn.Module):
    def __init__(self, encoder, decoder):
        from losses import LogDetLoss
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = LogDetLoss(eta=.5)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, m_true):
        n = z.size(0)
        m = squareform(z, n)
        return self.loss(m, m_true)


class SuperGATNet(Module):
    def __init__(self, in_channels, out_channels, num_layers, dropout=.4, heads=4):
        super().__init__()
        layers = []
        self.dropout = dropout
        for i in range(num_layers-1):
            mlp = SuperGATConv(in_channels, out_channels, heads,
                               dropout=dropout, attention_type='MX',
                               edge_sample_ratio=0.8,
                               is_undirected=True)
            self.add_module(f'GAT{i}', mlp)
            in_channels = out_channels*heads
        self.add_module('GAT_out',
                        SuperGATConv(in_channels, out_channels, heads,
                                     concat=False, dropout=dropout,
                                     attention_type='MX', edge_sample_ratio=0.8,
                                     is_undirected=True)
                        )

    def forward(self, x, edge_index):
        # TODO: need to change the guts so that att_loss is proportional to edge weights (y values) in training data, not presence/absence
        att_loss = 0
        for layer in self.children():
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(layer(x, edge_index))
            att_loss += layer.get_attention_loss()
        return x, att_loss


class SuperAttentionNetwork(GraphNetwork):
    def __init__(self,
                 gene_embedding=64,
                 graph_embedding=128,
                 num_layers=4,
                 shared_weights=False,
                 resnet=resnet18,
                 output='metric',
                 device='cpu'):
        super().__init__()
        self.embed = nn.Embedding(N_STATES, EMBED_DIM)
        self.gene_encoder = resnet(in_channels=EMBED_DIM,
                                   n_classes=gene_embedding)
        self.elu = nn.ELU(inplace=True)
        self.norm1 = nn.BatchNorm1d(gene_embedding)

        self.graph_layer = SuperGATNet(
            gene_embedding, graph_embedding, num_layers)
        self.norm2 = nn.BatchNorm1d(graph_embedding)
        if output == 'metric':
            self.output_layer = MetricDecoder()
        elif output == 'covariance':
            self.output_layer = CovarianceDecoder()

        self.set_devices(device)

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        x = data.x
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x)
        x = self.embed(x.long()).transpose(-1, -3).to(self.device)
        x = self.gene_encoder(x)
        x = self.elu(x).squeeze()  # hack to avoid rewriting ResNet
        x = self.norm1(x)
        x, attn_loss = self.graph_layer(x, edge_index)
        x = self.norm2(x)
        x = self.output_layer(x, batches)  # output must be positive
        return x, attn_loss


class BiAttention(Module):
    def __init__(self,
                 block_size=32,
                 heads=4,
                 layers=4,
                 device='cuda'
                 ):
        super().__init__()
        block_transforms = nn.ModuleList()
        for _ in range(layers):
            block_transforms.append(nn.TransformerEncoderLayer(
                block_size, nhead=heads, batch_first=True))
        self.device = device
        self.block_size = block_size
        self.block_stride = block_size
        self.add_module('block_transforms', block_transforms)
        self.to(self.device)

    def forward(self, data):
        x = data.x.to(self.device)
        batches = data.batches.to(self.device)
        output = []
        for batch in batch_iter(x, batches):
            for transform in self.block_transforms:
                o = torch.zeros_like(batch)
                for i, site in enumerate(batch.permute(-1, 0, 1)):
                    o[i] = transform(site)
                # o = []
                # for block in batch.unfold(1, self.block_size, self.block_stride):
                #     o.append(block)
                batch = torch.cat(o, 0)
            output.append(batch)
        output = torch.cat(output, 0)
        return output


class GATNet(Module):
    """Multi-layer graph attention network."""

    def __init__(self, in_channels: int = None,
                 hidden_channels: list = [],
                 output_channels: int = None,
                 heads=8,
                 batch_norm=False,
                 dropout=.1):
        """hidden_channels: list of channel sizes. If None, returns a FC layer with no GAT."""
        super().__init__()
        self.batch_norm = batch_norm
        self.nchannels = 2+len(hidden_channels)
        modules = nn.ModuleDict()

        if hidden_channels:
            for i, c_out in enumerate(hidden_channels):
                if batch_norm:
                    modules[f'BN{i}'] = nn.BatchNorm1d(in_channels)

                mlp = GATv2Conv(in_channels, c_out,
                                heads, bias=not batch_norm,
                                concat=True,
                                add_self_loops=True,
                                dropout=dropout)
                modules[f'GAT{i}'] = mlp
                in_channels = c_out*heads  # since we're concatenating attention
            if output_channels is None:
                output_channels = c_out
        if batch_norm:
            bn = nn.BatchNorm1d(in_channels)
            modules[f'BN{i}'] = bn

        modules[f'GAT{i+1}'] = GATv2Conv(
            in_channels, output_channels,
            heads, bias=not batch_norm,
            concat=False, dropout=dropout)
        self.module_dict = modules

    def forward(self, x, edge_index):
        # x = x.float()
        for i in range(len(self.module_dict)):
            if self.batch_norm:
                x = self.module_dict[f'BN{i}'](x)
            x = self.module_dict[f'GAT{i}'](x, edge_index)
            x = F.elu(x)
        return x


class AttentionNetwork(GraphNetwork):
    def __init__(self,
                 gene_embedding=256,
                 hidden_channels=[256, 512, 1024, 1024],
                 output_channels=1024,
                 embed_dim=EMBED_DIM,
                 resnet_model='resnet18',
                 dropout=.1,
                 heads=8,
                 num_layers=1,
                 graph_batch_norm=False,
                 fc_layers: list = [],
                 output='metric',
                 device1=None,
                 device='cpu'):
        """Embedding layer followed by resnet layer followed by num_layers GAT layers

        Args:
            gene_embedding (int, optional): _description_. Defaults to 256.
            hidden_channels (list, optional): _description_. Defaults to [256, 512, 1024, 1024].
            embed_dim (_type_, optional): _description_. Defaults to EMBED_DIM.
            resnet_model (str, optional): _description_. Defaults to 'resnet18'.
            dropout (float, optional): _description_. Defaults to .1.
            heads (int, optional): _description_. Defaults to 8.
            num_layers (int, optional): _description_. Defaults to 1.
            graph_batch_norm (bool, optional): _description_. Defaults to False.
            output (str, optional): _description_. Defaults to 'metric'.
            device1 (_type_, optional): If set, will put resnet (gene embedding) onto the specified device.
                Useful for long sequences that won't fit into memory. Defaults to None.
            device (str, optional): _description_. Defaults to 'cpu'.

        Raises:
            ValueError: _description_
        """

        self.debug = False
        if num_layers > 1 and hidden_channels[0] != hidden_channels[-1]:
            # TODO: should this match gene_embedding too?
            raise ValueError('input must match output dim for multilayer GAT')
        super().__init__()
        # TODO: use self.add_module('',) to make these accessible to set_devices
        self.device = device

        self.add_module('embed', EmbedLayer(embed_dim, dropout))
        self.add_module('gene_encoder',
                        resnet_models[resnet_model](in_channels=embed_dim,
                                                    n_classes=gene_embedding))  # .to(self.device)
        self.add_module('elu', nn.ELU(inplace=False))
        # self.add_module('norm1', nn.BatchNorm1d(
        #     gene_embedding))

        graph_layers = torch.nn.ModuleList([GATNet(
            in_channels=gene_embedding,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            heads=heads,
            batch_norm=graph_batch_norm,
            dropout=dropout) for _ in range(num_layers)])
        self.add_module('graph_layers', graph_layers)
        if len(fc_layers):
            in_channels = output_channels
            output_channels = fc_layers[0]
            if len(fc_layers) == 1:
                self.add_module('FC', build_fc_network(in_channels,
                                                       output_channels,
                                                       nonlinearity=nn.ELU,
                                                       batch_norm=graph_batch_norm,))
            else:

                fc_net = build_fc_network(layers=[in_channels, output_channels]+fc_layers[:-1],
                                          nonlinearity=nn.ELU,
                                          batch_norm=graph_batch_norm,
                                          )
                self.add_module('FC', fc_net)

        if output == 'metric':
            self.add_module('output_layer',
                            MetricDecoder(
                                clip=1e9, as_list=False))
        elif output == 'covariance':
            self.add_module('output_layer',
                            CovarianceDecoder(
                                clip=1e9, as_list=False))

        self.to(self.device)
        if device1 is not None:  # use different device for the sequence embedding
            self.device1 = device1
            self.gene_encoder.to(device1)
            self.elu.to(device1)
            self.norm1.to(device1)
        else:
            self.gene_encoder.to(device)
            self.gene_encoder.device = device
        self.embed.to('cpu')

    def __repr__(self):
        return super().__repr__()+f"\ngraph layers:{', '.join(map(str,self.graph_layers))}"
        # self.set_devices(device)

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)

        x = data.x
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x)
        with torch.cuda.amp.autocast():
            x = self.embed(x).to(self.gene_encoder.device)
            # hack to avoid rewriting ResNet
            x = self.gene_encoder(x).squeeze()
            # x = self.norm1(x).to(self.device)
            for graph_layer in self.graph_layers:  # dont need to cast to float since GATNet does this
                x = self.elu(x)
                x = graph_layer(x, edge_index)
            try:
                x = self.FC(x)
            except AttributeError:
                pass
        x = self.output_layer(x.float(), batches)  # no trainable params
        return x

    def forward_debug(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        from .. import utils as u
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        x, y = data.x, data.y
        print('raw', self.rf_distance(x, batches, y))
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x)
        with torch.cuda.amp.autocast():
            x = self.embed(x).to(self.gene_encoder.device)
            print('embed', self.rf_distance(x, batches, y))
            x = self.gene_encoder(x)
            print('resnet', self.rf_distance(x, batches, y))
            x = x.squeeze()  # hack to avoid rewriting ResNet
            # x = self.norm1(x).to(self.device)
            for graph_layer in self.graph_layers:  # dont need to cast to float since GATNet does this
                x = self.elu(x)
                x = graph_layer(x, edge_index)
                print('graph layer', self.rf_distance(x, batches, y))

            try:
                x = self.FC(x)
                print('final fc', self.rf_distance(x, batches, y))
            except AttributeError:
                pass

        x = self.output_layer(x.float(), batches)  # no trainable params
        return x


class SequentialAttentionNetwork(GraphNetwork):
    def __init__(self,
                 kernel: int = 200,
                 stride: int = 100,
                 gat_conv_layers: list = [1024]*4,
                 char_embedding_dim: int = EMBED_DIM,
                 dropout: float = .1,
                 heads: int = 8,
                 num_layers: int = 1,
                 graph_batch_norm: bool = False,
                 seq_embedding_layers: list = [512]*3,
                 output_layers: list = [256]*3,
                 output='metric',
                 device='cpu', **kwargs):
        """Embedding layer followed by resnet layer followed by num_layers GAT layers

        Args:
            gene_embedding (int, optional): _description_. Defaults to 256.
            hidden_channels (list, optional): _description_. Defaults to [256, 512, 1024, 1024].
            embed_dim (_type_, optional): _description_. Defaults to EMBED_DIM.
            resnet_model (str, optional): _description_. Defaults to 'resnet18'.
            dropout (float, optional): _description_. Defaults to .1.
            heads (int, optional): _description_. Defaults to 8.
            num_layers (int, optional): _description_. Defaults to 1.
            graph_batch_norm (bool, optional): _description_. Defaults to False.
            output (str, optional): _description_. Defaults to 'metric'.
            device1 (_type_, optional): If set, will put resnet (gene embedding) onto the specified device.
                Useful for long sequences that won't fit into memory. Defaults to None.
            device (str, optional): _description_. Defaults to 'cpu'.

        Raises:
            ValueError: _description_
        """

        self.debug = False
        super().__init__()
        self.device = device
        self.latent_dim = seq_embedding_layers[0]
        conv1_channels = 1024
        conv2_channels = 512
        conv3_channels = 128
        embedding_network = nn.Sequential(
            # nn.BatchNorm1d(char_embedding_dim),
            nn.Conv1d(in_channels=char_embedding_dim,
                      out_channels=conv1_channels,
                      kernel_size=kernel,
                      stride=stride,
                      bias=True
                      ),
            nn.BatchNorm1d(conv1_channels),
            nn.ELU(),
            nn.Conv1d(in_channels=conv1_channels,
                      out_channels=conv2_channels,
                      kernel_size=kernel//2,
                      stride=max(1, stride//2),
                      bias=False
                      ),
            nn.BatchNorm1d(conv2_channels),
            nn.ELU(),
            # nn.Conv1d(in_channels=conv2_channels,
            #           out_channels=conv3_channels,
            #           kernel_size=kernel//4,
            #           stride=max(1, stride//4),
            #           bias=False
            #           ),
            # nn.BatchNorm1d(conv3_channels),
            # nn.ELU(),
            nn.Flatten(),
            nn.AdaptiveAvgPool1d(self.latent_dim),
            *build_fc_network(
                layers=seq_embedding_layers,
                nonlinearity=nn.ELU,
                batch_norm=graph_batch_norm,
            )

        )
        *hidden, output_channels = gat_conv_layers

        graph_layers = nn.ModuleList([GATNet(
            in_channels=seq_embedding_layers[-1],
            hidden_channels=hidden,
            output_channels=output_channels,
            heads=heads,
            batch_norm=graph_batch_norm,
            dropout=dropout)])
        for _ in range(num_layers-1):
            graph_layers.append(GATNet(
                in_channels=output_channels,
                hidden_channels=hidden,
                output_channels=output_channels,
                heads=heads,
                batch_norm=graph_batch_norm,
                dropout=dropout))

        output_FC = build_fc_network(
            in_channels=output_channels,
            layers=output_layers,
            nonlinearity=nn.ELU,
            batch_norm=graph_batch_norm)

        self.add_module('embed',
                        EmbedLayer(char_embedding_dim, dropout))

        self.add_module('embedding_FC', embedding_network)

        self.add_module('graph_layers', graph_layers)

        self.add_module('output_FC', output_FC)

        if output == 'metric':
            self.add_module(
                'output_layer',
                MetricDecoder(clip=1e9, as_list=False))
        elif output == 'covariance':
            self.add_module(
                'output_layer',
                CovarianceDecoder(clip=1e9, as_list=False))

        self.to(self.device)
        # self.embed.to('cpu')

    def __repr__(self):
        # +f"\ngraph layers:{', '.join(map(str,self.graph_layers))}"
        return super().__repr__()
        # self.set_devices(device)

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        input = data.x.to(self.device)
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        with torch.cuda.amp.autocast():
            x = (self
                 .embed(input)
                 .squeeze()
                 )
            x = self.embedding_FC(x)
            for graph_layer in self.graph_layers:  # dont need to cast to float since GATNet does this
                x = graph_layer(x, edge_index)
            x = self.output_FC(x)

        x = self.output_layer(
            x.float(),
            batches)  # no trainable params
        return x

    def forward_debug(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        from .. import utils as u
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        x = data.x.to(self.device)
        y = data.y.to(self.device, non_blocking=True)
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x)
        stats = {}
        stats['raw'] = self.rf_distance(x, batches, y)
        with torch.cuda.amp.autocast():
            x = self.embed(x).squeeze().to(self.device)
            stats['embed'] = self.rf_distance(x, batches, y)

            rfs = []
            x = self.embedding_FC(x)
            stats['embed_FC'] = self.rf_distance(x, batches, y)
            # dont need to cast to float since GATNet does this
            for i, graph_layer in enumerate(self.graph_layers):
                x = graph_layer(x, edge_index)
                stats[f'GAT{i}'] = self.rf_distance(x, batches, y)

            x = self.output_FC(x)
            stats['output_FC'] = self.rf_distance(x, batches, y)

        x = self.output_layer(
            x.float(), batches)  # no trainable params

        return x, stats


class SequentialUnfoldAttentionNetwork(GraphNetwork):
    def __init__(self,
                 kernel: int = 200,
                 stride: int = 100,
                 gat_conv_layers: list = [1024]*4,
                 char_embedding_dim: int = EMBED_DIM,
                 dropout: float = .1,
                 heads: int = 8,
                 num_layers: int = 1,
                 graph_batch_norm: bool = False,
                 seq_embedding_layers: list = [512]*3,
                 output_layers: list = [256]*3,
                 output='metric',
                 device='cpu', **kwargs):
        """Embedding layer followed by resnet layer followed by num_layers GAT layers

        Args:
            gene_embedding (int, optional): _description_. Defaults to 256.
            hidden_channels (list, optional): _description_. Defaults to [256, 512, 1024, 1024].
            embed_dim (_type_, optional): _description_. Defaults to EMBED_DIM.
            resnet_model (str, optional): _description_. Defaults to 'resnet18'.
            dropout (float, optional): _description_. Defaults to .1.
            heads (int, optional): _description_. Defaults to 8.
            num_layers (int, optional): _description_. Defaults to 1.
            graph_batch_norm (bool, optional): _description_. Defaults to False.
            output (str, optional): _description_. Defaults to 'metric'.
            device1 (_type_, optional): If set, will put resnet (gene embedding) onto the specified device.
                Useful for long sequences that won't fit into memory. Defaults to None.
            device (str, optional): _description_. Defaults to 'cpu'.

        Raises:
            ValueError: _description_
        """

        self.debug = False
        in_channels, *hidden, output_channels = gat_conv_layers
        # if num_layers > 1 and in_channels != output_channels:
        #     raise ValueError(
        #         'input dim must match output dim for multilayer GAT')
        super().__init__()
        self.device = device

        seq_slice_dim = kernel*char_embedding_dim
        if len(seq_embedding_layers):
            if len(seq_embedding_layers) == 1:
                self.latent_dim = output_channels
                embedding_network = build_fc_network(seq_slice_dim,
                                                     self.latent_dim,
                                                     nonlinearity=nn.ELU,
                                                     batch_norm=graph_batch_norm,
                                                     )
            else:
                self.latent_dim = in_channels
                seq_embedding_layers = [seq_slice_dim] + \
                    seq_embedding_layers+[self.latent_dim]
                embedding_network = build_fc_network(
                    layers=seq_embedding_layers,
                    nonlinearity=nn.ELU,
                    batch_norm=graph_batch_norm,
                )
        else:
            embedding_network = nn.Identity()
            self.latent_dim = seq_slice_dim

        graph_layers = nn.ModuleList([GATNet(
            in_channels=output_channels,
            hidden_channels=hidden,
            output_channels=output_channels,
            heads=heads,
            batch_norm=graph_batch_norm,
            dropout=dropout)])
        for _ in range(num_layers-1):
            graph_layers.append(GATNet(
                in_channels=output_channels,
                hidden_channels=hidden,
                output_channels=output_channels,
                heads=heads,
                batch_norm=graph_batch_norm,
                dropout=dropout))

        output_FC = build_fc_network(layers=[self.latent_dim]+output_layers,
                                     nonlinearity=nn.ELU,
                                     batch_norm=graph_batch_norm)

        self.add_module('embed', EmbedLayer(char_embedding_dim, dropout))

        self.add_module('unfold', nn.Unfold(
            kernel_size=(kernel, 1),
            stride=stride))

        self.add_module('embedding_FC', embedding_network)

        self.add_module('graph_layers', graph_layers)

        self.add_module('output_FC', output_FC)

        if output == 'metric':
            self.add_module('output_layer',
                            MetricDecoder(
                                clip=1e9, as_list=False))
        elif output == 'covariance':
            self.add_module('output_layer',
                            CovarianceDecoder(
                                clip=1e9, as_list=False))

        self.to(self.device)
        self.embed.to('cpu')

    def __repr__(self):
        return super().__repr__()+f"\ngraph layers:{', '.join(map(str,self.graph_layers))}"
        # self.set_devices(device)

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)

        input = data.x
        ntaxa = input.shape[0]
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        with torch.cuda.amp.autocast():
            input = self.embed(input).to(self.device)

            output = torch.zeros((ntaxa, self.latent_dim), device=self.device)
            n = 1.
            for x in self.unfold(input).permute(2, 0, 1):
                x = self.embedding_FC(x)
                for graph_layer in self.graph_layers:  # dont need to cast to float since GATNet does this
                    x = graph_layer(x, edge_index)
                output += x
                n += 1
            output = self.output_FC(output/n)
        output = self.output_layer(
            output.float(), batches)  # no trainable params
        return output

    def forward_debug(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        from .. import utils as u
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        data.y = u.njtree(data.y)

        input = data.x
        ntaxa = input.shape[0]
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        print('raw', self.rf_distance(input, batches, data.y))
        with torch.cuda.amp.autocast():
            input = self.embed(input).to(self.device)
            print('embed', self.rf_distance(input, batches, data.y))

            output = torch.zeros((ntaxa, self.latent_dim), device=self.device)
            n = 0.
            rfs = []
            for x in self.unfold(input).permute(2, 0, 1):
                x = self.embedding_FC(x)
                for graph_layer in self.graph_layers:  # dont need to cast to float since GATNet does this
                    x = graph_layer(x, edge_index)
                output += x
                n += 1
                rfs.append(self.rf_distance(x, batches, data.y))
            print('GAT', n, 'subseqs:', rfs, 'sum:',
                  self.rf_distance(x, batches, data.y))

            output = self.output_FC(output/n)
            print('output FC', self.rf_distance(output, batches, data.y))

        output = self.output_layer(
            output.float(), batches)  # no trainable params

        return output


class SequentialGMTNetwork(GraphNetwork):
    def __init__(self,
                 kernel: int = 11,
                 stride: int = 3,
                 block: int = 32,
                 block_stride: int = 7,
                 k: int = 3,
                 heads: int = 4,
                 latent_dim: int = 1024,
                 char_embedding_dim: int = EMBED_DIM,
                 dropout: float = .1,
                 num_layers: int = 1,
                 n_rounds: int = 1,
                 graph_batch_norm: bool = False,
                 output='metric',
                 device='cpu', **kwargs):
        """Embedding layer followed by sequence of dynamic edge conv graphs.
        Each graph computes a new CNN along the input (must have stride 1 since all graph outputs are concatenated)

        Args:
            gene_embedding (int, optional): _description_. Defaults to 256.
            hidden_channels (list, optional): _description_. Defaults to [256, 512, 1024, 1024].
            embed_dim (_type_, optional): _description_. Defaults to EMBED_DIM.
            resnet_model (str, optional): _description_. Defaults to 'resnet18'.
            dropout (float, optional): _description_. Defaults to .1.
            heads (int, optional): _description_. Defaults to 8.
            num_layers (int, optional): _description_. Defaults to 1.
            graph_batch_norm (bool, optional): _description_. Defaults to False.
            output (str, optional): _description_. Defaults to 'metric'.
            device1 (_type_, optional): If set, will put resnet (gene embedding) onto the specified device.
                Useful for long sequences that won't fit into memory. Defaults to None.
            device (str, optional): _description_. Defaults to 'cpu'.

        Raises:
            ValueError: _description_
        """
        from torch_geometric.nn import EdgeConv
        self.debug = False
        # if num_layers > 1 and in_channels != output_channels:
        #     raise ValueError(
        #         'input dim must match output dim for multilayer GAT')
        super().__init__()
        self.device = device
        self.k = k
        self.latent_dim = latent_dim
        self.n_rounds = n_rounds

        embedding_network = nn.Sequential(
            nn.Conv1d(in_channels=char_embedding_dim,
                      out_channels=self.latent_dim,
                      kernel_size=kernel,
                      stride=stride,
                      bias=True
                      ),
            nn.Dropout(p=dropout),
        )
        graph_embed_dim = self.latent_dim*heads
        for r in range(n_rounds):
            seq_edge_mapper = build_fc_network(
                # layers=[self.latent_dim*i for i in range(heads*2, 0, -2)],
                layers=[latent_dim*2]*3,
                bias=True,
                nonlinearity=nn.ELU,
                batch_norm=False)
            seq_graph = EdgeConv(seq_edge_mapper,  aggr='mean')
            self.add_module(f'r{r}_seq_graph', seq_graph)

            site_graph = GraphMultisetTransformer(
                in_channels=latent_dim*2,
                hidden_channels=latent_dim,
                out_channels=latent_dim,
                num_heads=heads,
                Conv=GATv2Conv)
            self.add_module(f'r{r}_graph_net', site_graph)

            # conv_layer_sizes = [self.latent_dim//2**i for i in range(2)]
            projection_layer = make_conv_net(
                [latent_dim, latent_dim//2],
                kernel=1,
                stride=1,
                dropout=dropout)
            self.add_module(f'r{r}_projection', projection_layer)

        self.add_module('char_embed', EmbedLayer(char_embedding_dim, dropout))

        self.add_module('site_embed', embedding_network)

        if output == 'metric':
            self.add_module('output_layer',
                            MetricDecoder(
                                clip=1e9, as_list=False))
        elif output == 'covariance':
            self.add_module('output_layer',
                            CovarianceDecoder(
                                clip=1e9, as_list=False))

        self.to(self.device)
        self.char_embed.to('cpu')

    def apply_graph(self, graph, x, batches=None):
        edges = pool.knn_graph(
            x=x.flatten(1, -1),
            k=self.k,
            batch=batches)  # BlockEdgeConv
        x = graph(x, edge_index=edges, batch=batches)
        return x

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)

        input = data.x
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        with torch.cuda.amp.autocast(enabled=False):
            output = self.char_embed(input).squeeze().to(self.device)
            output = self.site_embed(output)
            ntaxa, hidden_dim, nsites = output.shape

            for r in range(self.n_rounds):
                # convolve similar sites
                output_list = []
                graph = getattr(self, f'r{r}_seq_graph')
                seq_features = global_mean_pool(output, batches)
                for taxon_id, x in enumerate(output):
                    site_edges = pool.knn_graph(
                        seq_features[batches[taxon_id]].T, k=self.k)
                    output_list.append(graph(x.T, site_edges).T[None])
                output = torch.cat(output_list, 0)

                # convolve similar taxa
                graph = getattr(self, f'r{r}_graph_net')
                output_list = []
                for x in output.permute(2, 0, 1):
                    x = self.apply_graph(graph, x, batches)
                    # x = self.apply_graph(graph, x, batches)
                    output_list.append(x[..., None])
                output = torch.cat(output_list, -1)

                # reduce dimensionality
                graph = getattr(self, f'r{r}_projection')
                output = graph(output)

        output = self.output_layer(
            output.flatten(1, -1).float(), batches)  # no trainable params
        return output

    def forward_debug(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        from .. import utils as u
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        y = u.njtree(data.y)
        input = data.x
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        dists = dict()
        dists['raw'] = self.rf_distance(input, batches, y)
        with torch.cuda.amp.autocast():
            output = self.char_embed(input).squeeze().to(self.device)
            dists['embed'] = self.rf_distance(input, batches, y)
            output = self.site_embed(output)
            dists['site embed'] = self.rf_distance(
                output, batches, y)

            ntaxa, hidden_dim, nsites = output.shape

            for r in range(self.n_rounds):

                seq_features = global_mean_pool(output, batches)

                # convolve similar sites
                output_list = []
                graph = getattr(self, f'r{r}_seq_graph')
                for taxon_id, x in enumerate(output):
                    site_edges = pool.knn_graph(
                        seq_features[batches[taxon_id]].T, k=self.k)
                    output_list.append(graph(x.T, site_edges).T[None])
                output = torch.cat(output_list, 0)

                dists[f'r{r}seq graph'] = self.rf_distance(
                    output.flatten(1, - 1), batches, y)

                # convolve similar taxa
                graph = getattr(self, f'r{r}_graph_net')
                output_list = []
                for x in output.permute(2, 0, 1):
                    x = self.apply_graph(graph, x, batches)
                    output_list.append(x[..., None])
                output = torch.cat(output_list, -1)
                dists[f'r{r}_graph'] = self.rf_distance(
                    output.reshape(ntaxa, -1), batches, y)

                graph = getattr(self, f'r{r}_projection')
                output = graph(output)
                dists[f'r{r}_proj'] = self.rf_distance(
                    output.flatten(1, - 1),
                    batches,
                    y)

        output = self.output_layer(
            output.flatten(1, -1).float(), batches)  # no trainable params
        return output, dists


class GATBlockConvNetwork(GraphNetwork):
    def __init__(self,
                 kernel: int = 11,
                 stride: int = 3,
                 block: int = 32,
                 block_stride: int = 7,
                 k: int = 3,
                 heads: int = 4,
                 latent_dim: int = 1024,
                 char_embedding_dim: int = EMBED_DIM,
                 dropout: float = .1,
                 num_layers: int = 1,
                 n_rounds: int = 1,
                 graph_batch_norm: bool = False,
                 dynamic_graphs: bool = False,
                 output='metric',
                 device='cpu', **kwargs):
        """Embedding layer followed by sequence of dynamic edge conv graphs.
        Each graph computes a new CNN along the input (must have stride 1 since all graph outputs are concatenated)

        Args:
            gene_embedding (int, optional): _description_. Defaults to 256.
            hidden_channels (list, optional): _description_. Defaults to [256, 512, 1024, 1024].
            embed_dim (_type_, optional): _description_. Defaults to EMBED_DIM.
            resnet_model (str, optional): _description_. Defaults to 'resnet18'.
            dropout (float, optional): _description_. Defaults to .1.
            heads (int, optional): _description_. Defaults to 8.
            num_layers (int, optional): _description_. Defaults to 1.
            graph_batch_norm (bool, optional): _description_. Defaults to False.
            output (str, optional): _description_. Defaults to 'metric'.
            device1 (_type_, optional): If set, will put resnet (gene embedding) onto the specified device.
                Useful for long sequences that won't fit into memory. Defaults to None.
            device (str, optional): _description_. Defaults to 'cpu'.

        Raises:
            ValueError: _description_
        """
        from torch_geometric.nn import EdgeConv
        self.debug = False
        # if num_layers > 1 and in_channels != output_channels:
        #     raise ValueError(
        #         'input dim must match output dim for multilayer GAT')
        super().__init__()
        self.device = device
        self.k = k
        self.latent_dim = latent_dim
        self.n_rounds = n_rounds
        self.dynamic_graphs = dynamic_graphs

        embedding_network = nn.Sequential(
            nn.BatchNorm1d(char_embedding_dim),
            nn.Conv1d(in_channels=char_embedding_dim,
                      out_channels=latent_dim,
                      kernel_size=kernel,
                      stride=stride,
                      bias=False,
                      ),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(p=dropout),
        )
        graph_embed_dim = self.latent_dim*heads
        for r in range(n_rounds):
            site_graph = nn.ModuleList([
                GATv2Conv(in_channels=latent_dim,
                          out_channels=latent_dim,
                          heads=4, bias=False,
                          concat=True,
                          dropout=dropout),
                GATv2Conv(in_channels=latent_dim*heads,
                          out_channels=latent_dim,
                          heads=4, bias=True,
                          concat=False,
                          dropout=dropout), ]
            )

            self.add_module(f'r{r}_graph_net', site_graph)
            seq_edge_mapper = build_fc_network(
                # layers=[self.latent_dim*i for i in range(heads*2, 0, -2)],
                layers=[latent_dim*2]*3,
                nonlinearity=nn.ELU,
                batch_norm=True)
            seq_graph = EdgeConv(seq_edge_mapper,  aggr='mean')
            self.add_module(f'r{r}_seq_graph', seq_graph)

            # conv_layer_sizes = [self.latent_dim//2**i for i in range(2)]
            projection_layer = make_conv_net(
                [latent_dim*2, latent_dim],
                kernel=1,
                stride=1,
                dropout=dropout)
            self.add_module(f'r{r}_projection', projection_layer)

        self.add_module('char_embed', EmbedLayer(char_embedding_dim, dropout))

        self.add_module('site_embed', embedding_network)

        if output == 'metric':
            self.add_module('output_layer',
                            MetricDecoder(
                                clip=1e9, as_list=False))
        elif output == 'covariance':
            self.add_module('output_layer',
                            CovarianceDecoder(
                                clip=1e9, as_list=False))

        self.to(self.device)
        self.char_embed.to('cpu')

    def __repr__(self):
        # +f"\ngraph layers:{', '.join(map(str,self.graph_layers))}"
        return super().__repr__()
        # self.set_devices(device)

    def apply_graph(self, graph, x, batches=None):
        edges = pool.knn_graph(
            x=x.flatten(1, -1),
            k=self.k,
            batch=batches)  # BlockEdgeConv
        x = graph(x, edge_index=edges)
        return x

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)

        input = data.x
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        with torch.cuda.amp.autocast(enabled=False):
            output = self.char_embed(input).squeeze().to(self.device)
            output = self.site_embed(output)
            ntaxa, hidden_dim, nsites = output.shape

            for r in range(self.n_rounds):

                # convolve similar taxa
                gats = getattr(self, f'r{r}_graph_net')
                output_list = []
                for x in output.permute(2, 0, 1):
                    for graph in gats:
                        x = F.elu(x)
                        if self.dynamic_graphs:
                            x = self.apply_graph(graph, x, batches)
                        else:
                            x = graph(x, edge_index)
                    output_list.append(x[..., None])
                output = torch.cat(output_list, -1)

                # convolve similar sites
                seq_features = global_mean_pool(output, batches)
                output_list = []
                graph = getattr(self, f'r{r}_seq_graph')
                for taxon_id, x in enumerate(output):
                    site_edges = pool.knn_graph(
                        seq_features[batches[taxon_id]].T, k=self.k)
                    output_list.append(graph(x.T, site_edges).T[None])
                output = torch.cat(output_list, 0)

                # projection layer
                graph = getattr(self, f'r{r}_projection')
                output = graph(output)

                # taxa_per_batch = torch.unique(
                #     batches, return_counts=True)[-1]

        output = self.output_layer(
            output.flatten(1, -1).float(), batches)  # no trainable params
        return output

    def forward_debug(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        from .. import utils as u
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        y = u.njtree(data.y)
        input = data.x
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        dists = dict()
        dists['raw'] = self.rf_distance(input, batches, y)
        with torch.cuda.amp.autocast():
            output = self.char_embed(input).squeeze().to(self.device)
            dists['embed'] = self.rf_distance(input, batches, y)
            output = self.site_embed(output)
            dists['site embed'] = self.rf_distance(
                output, batches, y)

            ntaxa, hidden_dim, nsites = output.shape

            for r in range(self.n_rounds):

                # convolve similar taxa
                gats = getattr(self, f'r{r}_graph_net')
                output_list = []
                for x in output.permute(2, 0, 1):
                    for graph in gats:
                        x = F.elu(x)
                        if self.dynamic_graphs:
                            x = self.apply_graph(graph, x, batches)
                        else:
                            x = graph(x, edge_index)
                    output_list.append(x[..., None])
                output = torch.cat(output_list, -1)
                dists[f'r{r}_graph'] = self.rf_distance(
                    output.reshape(ntaxa, -1), batches, y)

                # convolve similar sites
                seq_features = global_mean_pool(output, batches)
                output_list = []
                graph = getattr(self, f'r{r}_seq_graph')
                for taxon_id, x in enumerate(output):
                    site_edges = pool.knn_graph(
                        seq_features[batches[taxon_id]].T, k=self.k)
                    output_list.append(graph(x.T, site_edges).T[None])
                output = torch.cat(output_list, 0)

                dists[f'r{r}seq graph'] = self.rf_distance(
                    output.flatten(1, - 1), batches, y)

                # taxa_per_batch = torch.unique(
                #     batches, return_counts=True)[-1]

                graph = getattr(self, f'r{r}_projection')
                output = graph(output)
                dists[f'r{r}_proj'] = self.rf_distance(
                    output.flatten(1, - 1),
                    batches,
                    y)

        output = self.output_layer(
            output.flatten(1, -1).float(), batches)  # no trainable params
        return output, dists


class DoubleBlockConvNetwork(GraphNetwork):
    def __init__(self,
                 kernel: int = 200,
                 stride: int = 100,
                 k: int = 3,
                 n_graph_nets: int = 1,
                 kernel_size_1D: int = 3,
                 proj_kernel_size: int = 1,
                 #  stride_1D: int = 2,
                 site_conv_layers: list = [1024]*4,
                 taxon_conv_layers: list = None,
                 char_embedding_dim: int = EMBED_DIM,
                 dropout: float = .1,
                 num_layers: int = 1,
                 n_rounds: int = 1,
                 batch_norm: bool = False,
                 output_layers: list = [256]*3,
                 output='metric',
                 device='cpu', **kwargs):
        """Char Embedding --> CNN Embedding --> [[taxa GCN] x N --> subsequence GCN --> projection NN] x M  --> MetricDecoder
        Each graph computes a new CNN along the input (must have stride 1 since all graph outputs are concatenated)

        Args:
            kernel (int, optional): size of char embed kernel. Defaults to 200.
            stride (int, optional): stride over alignment input. Defaults to 100.
            k (int, optional): k-nearest neighbor graph for message passing. Defaults to 3.
            n_graph_nets (int, optional): graph nets to pass through. Defaults to 1.
            kernel_size_1D (int, optional): edge feature conv. Defaults to 3.
            proj_kernel_size (int, optional): final feature conv. Defaults to 1.
            stride_1D (int, optional): stride. Defaults to 2.
            site_conv_layers (list, optional): last layer must have 1/2 the features as 1st. Defaults to [1024]*4.
            taxon_conv_layers (list, optional): unused. Defaults to [1024]*4.
            char_embedding_dim (int, optional): embed each character in n-dim Euclidean space. Defaults to EMBED_DIM.
            dropout (float, optional): dropout between layers. Defaults to .1.
            num_layers (int, optional): unused. Defaults to 1.
            n_rounds (int, optional): number of edge+site conv iterations. Defaults to 1.
            batch_norm (bool, optional): do not use. Defaults to False.
            output_layers (list, optional): unused. Defaults to [256]*3.
            output (str, optional): metric or covariance. Defaults to 'metric'.
            device (str, optional): all layers EXCEPT char_embed will be on device. Defaults to 'cpu'.
        """
        from torch_geometric.nn import EdgeConv
        self.debug = False
        # if num_layers > 1 and in_channels != output_channels:
        #     raise ValueError(
        #         'input dim must match output dim for multilayer GAT')
        super().__init__()
        # self.device = device
        self.k = k
        self.latent_dim = site_conv_layers[0]//2
        self.embedding_dim = site_conv_layers[-1]
        self.n_graph_nets = n_graph_nets
        self.n_rounds = n_rounds

        embedding_network = nn.Sequential(*make_conv_net(
            [char_embedding_dim, self.latent_dim],
            kernel=kernel, stride=stride, dropout=dropout),
            nn.ELU()
        )
        self.norm = nn.LayerNorm(self.latent_dim)

        graph_embed_dim = self.embedding_dim*n_graph_nets+self.latent_dim
        if taxon_conv_layers is None:
            taxon_conv_layers = [graph_embed_dim//2,
                                 graph_embed_dim//4]
        for r in range(n_rounds):
            for i in range(2):
                graph_nets = nn.ModuleList()
                for _ in range(n_graph_nets):
                    edge_mapper = make_conv_net(
                        site_conv_layers,
                        kernel_size_1D,
                        stride=1,
                        batch_norm=batch_norm,
                        nonlinearity=nn.ELU,
                        dropout=dropout)
                    site_graph = EdgeBlockConv(edge_mapper,  aggr='mean')
                    graph_nets.append(site_graph)
                self.add_module(f'r{r}_graph_nets_{i}', graph_nets)
                seq_edge_mapper = make_conv_net(
                    [graph_embed_dim*2, *
                     taxon_conv_layers, self.latent_dim],
                    kernel=1, stride=1,
                    nonlinearity=nn.ELU,
                    batch_norm=batch_norm,
                    dropout=dropout)

                seq_graph = EdgeBlockConv(seq_edge_mapper,  aggr='mean')
                self.add_module(f'r{r}_seq_graph{i}', seq_graph)

            # conv_layer_sizes = [graph_embed_dim // 2 **
            #                     i for i in range(2, 4)]+[self.latent_dim]
            conv_layer_sizes = [self.latent_dim]*2
            projection_layer = make_conv_net_2d(
                conv_layer_sizes,
                kernel=proj_kernel_size,
                stride=max(1, proj_kernel_size//2),
                batch_norm=batch_norm,
                dropout=dropout)
            self.add_module(f'r{r}_projection', projection_layer)

        self.add_module('char_embed',
                        nn.Embedding(
                            num_embeddings=22, embedding_dim=char_embedding_dim))

        self.add_module('site_embed', embedding_network)

        # self.add_module('unfold', nn.Unfold(
        #     kernel_size=(kernel, 1), stride=stride))

        if output == 'metric':
            self.add_module('output_layer',
                            MetricDecoder(
                                clip=1e9, as_list=True))
        elif output == 'covariance':
            self.add_module('output_layer',
                            CovarianceDecoder(
                                clip=1e9, as_list=False))

        # self.to(self.device)
        # self.char_embed.to('cpu')

    def __repr__(self):
        # +f"\ngraph layers:{', '.join(map(str,self.graph_layers))}"
        return super().__repr__()
        # self.set_devices(device)

    def apply_graph(self, graph, x, batches=None):
        edges = pool.knn_graph(
            x=x.flatten(1),
            k=self.k,
            loop=True,
            batch=batches)  # BlockEdgeConv
        x = graph(x, edge_index=edges)
        return x

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index  # .to(self.device, non_blocking=True)
        batches = data.batch  # .to(self.device, non_blocking=True)
        x = data.x
        nbatches, ntaxa, seq_length = x.shape
        x = x.long().flatten(0, 1)
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x)
        out = self.char_embed(x).transpose(-1, -2)
        # .permute(
        # 0, 3, 1, 2)  # .to(self.device)
        out = self.site_embed(out)
        out = self.norm(out.float().transpose(-1, -2)
                        ).transpose(-1, -2)  # layernorm

        with torch.cuda.amp.autocast(enabled=False):
            for r in range(self.n_rounds):
                # convolve similar taxa

                output_list = [out]
                for net_no, graph in enumerate(getattr(self, f'r{r}_graph_nets_0'), 1):
                    x = output_list[-1]
                    x = self.apply_graph(graph, x, batches)
                    output_list.append(x)
                out = torch.cat(output_list, 1)

                # taxa_per_batch = torch.unique(batches, return_counts=True)[-1]
                # seq_features = global_mean_pool(    out.flatten(start_dim=1),      batches)

                # convolve similar sites BY ALIGNMENT (could also do BY SEQUENCE but this would be slower)
                output_list = []
                out = out.unflatten(0, (nbatches, ntaxa))
                graph = getattr(self, f'r{r}_seq_graph0')
                seq_knn_graphs = [pool.knn_graph(
                    x.T, k=self.k, loop=True) for x in out.flatten(1, 2)]
                output = [out]
                for i, x in enumerate(out):
                    # edges = pool.knn_graph(
                    #     seq_features[batches[i]].T, k=self.k)
                    edges = seq_knn_graphs[i]
                    output_list.append(graph(x.permute(2, 1, 0), edges).T)
                out = torch.stack(output_list, 0)

                # convolve similar taxa
                output_list = [out.flatten(0, 1)]
                for net_no, graph in enumerate(getattr(self, f'r{r}_graph_nets_1'), 1):
                    x = output_list[-1]
                    x = self.apply_graph(graph, x, batches)
                    output_list.append(x)
                out = torch.cat(output_list, 1)

                # sites
                # output_list = []
                # graph = getattr(self, f'r{r}_seq_graph1')

                # seq_features = global_mean_pool(out, batches)
                # knn_graphs = [pool.knn_graph(s.T, k=self.k)
                #               for s in seq_features]
                # #   TODO: iterate across batches
                # for i, x in enumerate(out):
                #     # edges = pool.knn_graph(
                #     #     seq_features[batches[i]].T, k=self.k)
                #     edges = knn_graphs[batches[i]]
                #     output_list.append(graph(x.T, edges).T[None])
                # out = torch.cat(output_list, 0)
                output_list = []
                out = out.unflatten(0, (nbatches, ntaxa))
                graph = getattr(self, f'r{r}_seq_graph0')
                seq_knn_graphs = [pool.knn_graph(
                    x.T, k=self.k, loop=True) for x in out.flatten(1, 2)]
                output = [out]
                for i, x in enumerate(out):
                    # edges = pool.knn_graph(
                    #     seq_features[batches[i]].T, k=self.k)
                    edges = seq_knn_graphs[i]
                    output_list.append(graph(x.permute(2, 1, 0), edges).T)
                out = torch.stack(output_list, 0)
                graph = getattr(self, f'r{r}_projection')
                out = graph(out.transpose(1, 2))

        # out = self.output_layer(
        #     out.flatten(2, -1).float(), batches)  # no trainable params
        out = torch.stack(self.output_layer(out.transpose(1, 2).flatten(2)))
        return out

    def forward_debug(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        from .. import utils as u
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        y = u.njtree(data.y)
        input = data.x
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        dists = dict()
        dists['raw'] = self.rf_distance(input, batches, y)
        with torch.cuda.amp.autocast():
            output = self.char_embed(input).squeeze().to(self.device)
            dists['embed'] = self.rf_distance(input, batches, y)
            output = self.site_embed(output)
            dists['site embed'] = self.rf_distance(
                output, batches, y)

            ntaxa, hidden_dim, nsites = output.shape

            for r in range(self.n_rounds):
                output_list = [output]
                for net_no, graph in enumerate(getattr(self, f'r{r}_graph_nets_0'), 1):
                    x = output_list[-1]
                    x = self.apply_graph(graph, x, batches)
                    output_list.append(x)
                    dists[f'r{r}_graph1 {net_no}'] = self.rf_distance(
                        x, batches, y)
                output = torch.cat(output_list, 1)

                output_list = []
                graph = getattr(self, f'r{r}_seq_graph0')
                #   TODO: iterate across batches
                for i, x in enumerate(output):
                    # edges = pool.knn_graph(
                    #     seq_features[batches[i]].T, k=self.k)
                    edges = knn_graphs[batches[i]]
                    output_list.append(graph(x.T, edges).T[None])
                output = torch.cat(output_list, 0)
                dists[f'r{r}seq graph'] = self.rf_distance(
                    output.flatten(1, - 1), batches, y)
                output_list = [output]
                for net_no, graph in enumerate(getattr(self, f'r{r}_graph_nets_1'), 1):
                    x = output_list[-1]
                    x = self.apply_graph(graph, x, batches)
                    output_list.append(x)
                    dists[f'r{r}_graph2 {net_no}'] = self.rf_distance(
                        x, batches, y)
                output = torch.cat(output_list, 1)

                output_list = []
                graph = getattr(self, f'r{r}_seq_graph1')
                seq_features = global_mean_pool(output, batches)
                knn_graphs = [pool.knn_graph(s.T, k=self.k)
                              for s in seq_features]
                #   TODO: iterate across batches
                for i, x in enumerate(output):
                    # edges = pool.knn_graph(
                    #     seq_features[batches[i]].T, k=self.k)
                    edges = knn_graphs[batches[i]]
                    output_list.append(graph(x.T, edges).T[None])
                output = torch.cat(output_list, 0)
                dists[f'r{r}seq graph1'] = self.rf_distance(
                    output.flatten(1, - 1), batches, y)

                dists[f'r{r}_final graphs'] = self.rf_distance(
                    output.flatten(1, - 1), batches, y)

                graph = getattr(self, f'r{r}_projection')
                output = graph(output)
                dists[f'r{r}_proj'] = self.rf_distance(
                    output.flatten(1, - 1), batches, y)
        output = self.output_layer(
            output.flatten(1, -1).float(), batches)  # no trainable params
        return output, dists


class BlockConvNetwork(GraphNetwork):
    def __init__(self,
                 kernel: int = 200,
                 stride: int = 100,
                 k: int = 3,
                 n_graph_nets: int = 1,
                 kernel_size_1D: int = 3,
                 stride_1D: int = 2,
                 site_conv_layers: list = [1024]*4,
                 taxon_conv_layers: list = [1024]*4,
                 char_embedding_dim: int = EMBED_DIM,
                 dropout: float = .1,
                 num_layers: int = 1,
                 graph_batch_norm: bool = False,
                 output_layers: list = [256]*3,
                 output='metric',
                 device=None, **kwargs):
        """Embedding layer followed by sequence of dynamic edge conv graphs.
        Each graph computes a new CNN along the input (must have stride 1 since all graph outputs are concatenated)

        Args:
            gene_embedding (int, optional): _description_. Defaults to 256.
            hidden_channels (list, optional): _description_. Defaults to [256, 512, 1024, 1024].
            embed_dim (_type_, optional): _description_. Defaults to EMBED_DIM.
            resnet_model (str, optional): _description_. Defaults to 'resnet18'.
            dropout (float, optional): _description_. Defaults to .1.
            heads (int, optional): _description_. Defaults to 8.
            num_layers (int, optional): _description_. Defaults to 1.
            graph_batch_norm (bool, optional): _description_. Defaults to False.
            output (str, optional): _description_. Defaults to 'metric'.
            device1 (_type_, optional): If set, will put resnet (gene embedding) onto the specified device.
                Useful for long sequences that won't fit into memory. Defaults to None.
            device (str, optional): _description_. Defaults to 'cpu'.

        Raises:
            ValueError: _description_
        """
        self.debug = False
        # if num_layers > 1 and in_channels != output_channels:
        #     raise ValueError(
        #         'input dim must match output dim for multilayer GAT')
        super().__init__()
        self.k = k
        self.latent_dim = site_conv_layers[0]//2
        self.embedding_dim = site_conv_layers[-1]
        self.n_graph_nets = n_graph_nets

        embedding_network = nn.Sequential(
            nn.Conv1d(in_channels=char_embedding_dim,
                      out_channels=self.latent_dim,
                      kernel_size=kernel,
                      stride=stride,
                      ),
            nn.Dropout(p=dropout),
        )
        graph_nets = nn.ModuleList()
        for _ in range(n_graph_nets):
            edge_mapper = make_conv_net(
                site_conv_layers, kernel_size_1D, stride=1, dropout=dropout)
            site_graph = EdgeBlockConv(edge_mapper,  aggr='mean')
            graph_nets.append(site_graph)

        conv_layer_sizes = [self.embedding_dim *
                            (n_graph_nets+1) // 2**i for i in range(3)]
        f_1 = make_conv_net(conv_layer_sizes,
                            1,
                            1,
                            dropout)

        self.add_module('char_embed', EmbedLayer(char_embedding_dim, dropout))

        self.add_module('site_embed', embedding_network)

        # self.add_module('unfold', nn.Unfold(
        #     kernel_size=(kernel, 1), stride=stride))

        self.add_module('graph_nets', graph_nets)
        self.add_module('f_1', f_1)

        if output == 'metric':
            self.add_module('output_layer',
                            MetricDecoder(
                                clip=1e9, as_list=False))
        elif output == 'covariance':
            self.add_module('output_layer',
                            CovarianceDecoder(
                                clip=1e9, as_list=False))

        if device is not None:
            self.device = device
        self.to(self.device)
        self.char_embed.to('cpu')

    def __repr__(self):
        # +f"\ngraph layers:{', '.join(map(str,self.graph_layers))}"
        return super().__repr__()
        # self.set_devices(device)

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        # edge_index = data.edge_index.to(self.device, non_blocking=True)
        # batches = data.batch.to(self.device, non_blocking=True)

        input = data.x
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        with torch.cuda.amp.autocast():
            input = self.char_embed(input).squeeze().to(self.device)
            input = self.site_embed(input)
            ntaxa, hidden_dim, nsites = input.shape

            output = [input]
            for net_no, graph in enumerate(self.graph_nets, 1):
                x = output[-1]
                edges = pool.knn_graph(
                    x=x.flatten(1, -1),
                    k=self.k,
                    batch=batches)
                x = graph(x, edge_index=edges)
                output.append(x)
            output = torch.cat(output, 1)
            output = self.f_1(output)

        output = self.output_layer(
            output.flatten(1, -1).float(), batches)  # no trainable params
        return output

    def forward_debug(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        from .. import utils as u
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        y = u.njtree(data.y)
        input = data.x
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        dists = dict()
        dists['raw'] = self.rf_distance(input, batches, y)
        with torch.cuda.amp.autocast():
            input = self.char_embed(input).squeeze().to(self.device)
            input = F.elu(input)
            dists['embed'] = self.rf_distance(input, batches, y)
            input = self.site_embed(input)
            dists['site embed'] = self.rf_distance(
                input, batches, y)
            ntaxa, hidden_dim, nsites = input.shape
            output = [input]
            for net_no, graph in enumerate(self.graph_nets, 1):
                x = output[-1]
                edges = pool.knn_graph(x.flatten(
                    1, -1),
                    self.k,
                    batch=batches)
                x = graph(x, edge_index=edges)
                output.append(x)
                dists[f'graph {net_no}'] = self.rf_distance(x, batches, y)

            output = torch.cat(output, 1)
            dists['site graph'] = self.rf_distance(
                output.flatten(1, - 1), batches, y)
            output = self.f_1(output)

            output = output.reshape((ntaxa, -1))
        dists['conv layer'] = self.rf_distance(output, batches, y)
        output = self.output_layer(
            output.float(), batches)  # no trainable params
        return output, dists


class DynamicConvNetwork(GraphNetwork):
    def __init__(self,
                 kernel: int = 200,
                 stride: int = 100,
                 k: int = 3,
                 n_graph_nets: int = 1,
                 kernel_size_1D: int = 3,
                 stride_1D: int = 2,
                 site_conv_layers: list = [1024]*4,
                 taxon_conv_layers: list = [1024]*4,
                 char_embedding_dim: int = EMBED_DIM,
                 dropout: float = .1,
                 num_layers: int = 1,
                 graph_batch_norm: bool = False,
                 output_layers: list = [256]*3,
                 output='metric',
                 device='cpu', **kwargs):
        """Embedding layer followed by resnet layer followed by num_layers GAT layers

        Args:
            gene_embedding (int, optional): _description_. Defaults to 256.
            hidden_channels (list, optional): _description_. Defaults to [256, 512, 1024, 1024].
            embed_dim (_type_, optional): _description_. Defaults to EMBED_DIM.
            resnet_model (str, optional): _description_. Defaults to 'resnet18'.
            dropout (float, optional): _description_. Defaults to .1.
            heads (int, optional): _description_. Defaults to 8.
            num_layers (int, optional): _description_. Defaults to 1.
            graph_batch_norm (bool, optional): _description_. Defaults to False.
            output (str, optional): _description_. Defaults to 'metric'.
            device1 (_type_, optional): If set, will put resnet (gene embedding) onto the specified device.
                Useful for long sequences that won't fit into memory. Defaults to None.
            device (str, optional): _description_. Defaults to 'cpu'.

        Raises:
            ValueError: _description_
        """
        from torch_geometric.nn import DynamicEdgeConv, EdgeConv
        self.debug = False
        # if num_layers > 1 and in_channels != output_channels:
        #     raise ValueError(
        #         'input dim must match output dim for multilayer GAT')
        super().__init__()
        self.device = device
        self.k = k
        self.latent_dim = site_conv_layers[0]//2
        self.embedding_dim = site_conv_layers[-1]
        self.n_graph_nets = n_graph_nets

        embedding_network = nn.Sequential(
            nn.Conv1d(in_channels=char_embedding_dim,
                      out_channels=self.latent_dim,
                      kernel_size=kernel,
                      stride=stride,
                      ),
            nn.Dropout(p=dropout),
        )
        graph_nets = nn.ModuleList()
        for _ in range(n_graph_nets):
            edge_mapper = build_fc_network(
                layers=site_conv_layers,
                batch_norm=graph_batch_norm,
                nonlinearity=nn.ELU)
            site_graph = EdgeConv(edge_mapper,  aggr='mean')
            graph_nets.append(site_graph)

        conv_layer_sizes = [self.embedding_dim *
                            (n_graph_nets+1) // 2**i for i in range(3)]
        f_1 = make_conv_net(conv_layer_sizes,
                            kernel_size_1D,
                            stride_1D,
                            dropout)

        self.add_module('char_embed', EmbedLayer(char_embedding_dim, dropout))

        self.add_module('site_embed', embedding_network)

        # self.add_module('unfold', nn.Unfold(
        #     kernel_size=(kernel, 1), stride=stride))

        self.add_module('graph_nets', graph_nets)
        self.add_module('f_1', f_1)

        if output == 'metric':
            self.add_module('output_layer',
                            MetricDecoder(
                                clip=1e9, as_list=False))
        elif output == 'covariance':
            self.add_module('output_layer',
                            CovarianceDecoder(
                                clip=1e9, as_list=False))

        self.to(self.device)
        self.char_embed.to('cpu')

    def __repr__(self):
        # +f"\ngraph layers:{', '.join(map(str,self.graph_layers))}"
        return super().__repr__()
        # self.set_devices(device)

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)

        input = data.x
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        with torch.cuda.amp.autocast():
            input = self.char_embed(input).squeeze().to(self.device)
            input = self.site_embed(input)
            ntaxa, hidden_dim, nsites = input.shape

            output = [input]
            for net_no, graph in enumerate(self.graph_nets, 1):
                x_out = torch.empty(
                    (ntaxa, self.embedding_dim, nsites),
                    device=self.device)
                input = output[-1]
                edges = pool.knn_graph(input.flatten(
                    1, -1), self.k, batch=batches)
                for slice_no, x in enumerate(input.permute(2, 0, 1)):
                    x = graph(x, edge_index=edges)
                    x_out[..., slice_no] = x
                output.append(x_out)
            output = torch.cat(output, 1)
            output = self.f_1(output)

        output = self.output_layer(
            output.flatten(1, -1).float(), batches)  # no trainable params
        return output

    def forward_debug(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        from .. import utils as u
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        y = u.njtree(data.y)
        input = data.x
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        dists = dict()
        dists['raw'] = self.rf_distance(input, batches, y)
        with torch.cuda.amp.autocast():
            input = self.char_embed(input).squeeze().to(self.device)
            input = F.elu(input)
            dists['embed'] = self.rf_distance(input, batches, y)
            input = self.site_embed(input)
            dists['site embed'] = self.rf_distance(
                input, batches, y)
            ntaxa, hidden_dim, nsites = input.shape
            output = [input]
            for net_no, graph in enumerate(self.graph_nets, 1):
                x_out = torch.empty(
                    (ntaxa, self.embedding_dim, nsites),
                    device=self.device)
                input = output[-1]
                edges = pool.knn_graph(input.flatten(
                    1, -1), self.k, batch=batches)
                for slice_no, x in enumerate(input.permute(2, 0, 1)):
                    x = graph(x, edge_index=edges)
                    x_out[..., slice_no] = x
                    # if slice_no == 0:
                    #     print('slice', self.rf_distance(x, batches, y))
                dists[f'graph {net_no}'] = self.rf_distance(x_out, batches, y)
                output.append(x_out)
            output = torch.cat(output, 1)
            dists['site graph'] = self.rf_distance(
                output.flatten(1, - 1), batches, y)
            output = self.f_1(output)

            output = output.reshape((ntaxa, -1))
        dists['conv layer'] = self.rf_distance(output, batches, y)
        output = self.output_layer(
            output.float(), batches)  # no trainable params
        return output, dists


class DoubleDynamicConvNetwork(GraphNetwork):
    def __init__(self,
                 kernel: int = 200,
                 stride: int = 100,
                 kernel_size_1D: int = 1,
                 stride_1D: int = 1,
                 ks: Iterable = [3]*2,
                 site_conv_layers: list = [1024]*4,
                 taxon_conv_layers: list = [1024]*4,
                 char_embedding_dim: int = EMBED_DIM,
                 dropout: float = .1,
                 num_layers: int = 1,
                 graph_batch_norm: bool = False,
                 output_layers: list = [256]*3,
                 output='metric',
                 device='cpu', **kwargs):
        """Embedding layer followed by resnet layer followed by num_layers GAT layers

        Args:
            gene_embedding (int, optional): _description_. Defaults to 256.
            hidden_channels (list, optional): _description_. Defaults to [256, 512, 1024, 1024].
            embed_dim (_type_, optional): _description_. Defaults to EMBED_DIM.
            resnet_model (str, optional): _description_. Defaults to 'resnet18'.
            dropout (float, optional): _description_. Defaults to .1.
            heads (int, optional): _description_. Defaults to 8.
            num_layers (int, optional): _description_. Defaults to 1.
            graph_batch_norm (bool, optional): _description_. Defaults to False.
            output (str, optional): _description_. Defaults to 'metric'.
            device1 (_type_, optional): If set, will put resnet (gene embedding) onto the specified device.
                Useful for long sequences that won't fit into memory. Defaults to None.
            device (str, optional): _description_. Defaults to 'cpu'.

        Raises:
            ValueError: _description_
        """
        from torch_geometric.nn import DynamicEdgeConv
        self.debug = False
        super().__init__()
        self.device = device

        self.latent_dim = 128
        site_graph_neighbors, seq_graph_neighbors = ks

        embedding_network = nn.Sequential(
            nn.Conv1d(in_channels=char_embedding_dim,
                      out_channels=self.latent_dim,
                      kernel_size=kernel,
                      stride=stride,
                      ),
            nn.Dropout(p=dropout),
        )
        edge_mapper = build_fc_network(
            layers=[self.latent_dim*2] + site_conv_layers,
            batch_norm=graph_batch_norm,
            nonlinearity=nn.ELU)
        site_graph = DynamicEdgeConv(
            edge_mapper,
            k=site_graph_neighbors,
            aggr='mean')
        self.embedding_dim_1 = site_conv_layers[-1]
        self.embedding_dim_2 = taxon_conv_layers[-1]

        conv_layer_sizes = [self.embedding_dim_1 //
                            2**i for i in range(3)]+[taxon_conv_layers[-1]//2]

        f_1 = make_conv_net(
            conv_layer_sizes, kernel_size_1D, stride_1D, dropout)

        edge_mapper = build_fc_network(
            layers=taxon_conv_layers,
            batch_norm=graph_batch_norm)
        taxon_graph = DynamicEdgeConv(
            edge_mapper,
            k=seq_graph_neighbors,
            aggr='mean')

        # build output convnet
        conv_layer_sizes = [taxon_conv_layers[-1] //
                            2**i for i in range(3)]

        f_2 = make_conv_net(
            conv_layer_sizes, kernel_size_1D, stride_1D, dropout)

        self.add_module('char_embed', EmbedLayer(char_embedding_dim, dropout))

        self.add_module('site_embed', embedding_network)

        # self.add_module('unfold', nn.Unfold(
        #     kernel_size=(kernel, 1), stride=stride))

        self.add_module('site_graph', site_graph)
        self.add_module('f_1', f_1)

        self.add_module('taxon_graph', taxon_graph)
        self.add_module('f_2', f_2)

        if output == 'metric':
            self.add_module('output_layer',
                            MetricDecoder(
                                clip=1e9, as_list=False))
        elif output == 'covariance':
            self.add_module('output_layer',
                            CovarianceDecoder(
                                clip=1e9, as_list=False))

        self.to(self.device)
        self.char_embed.to('cpu')

    def __repr__(self):
        # +f"\ngraph layers:{', '.join(map(str,self.graph_layers))}"
        return super().__repr__()
        # self.set_devices(device)

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)

        input = data.x
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        with torch.cuda.amp.autocast():
            input = self.char_embed(input).squeeze().to(self.device)
            input = self.site_embed(input)
            ntaxa, hidden_dim, nsites = input.shape
            output1 = torch.empty(
                (ntaxa, self.embedding_dim_1, nsites),
                device=self.device)
            for i, x in enumerate(input.permute(2, 0, 1)):
                x = self.site_graph(x, batch=batches)  # , edge_index)
                output1[..., i] = x
            # output = (output.reshape((-1, self.f_1_dim)))
            output1 = self.f_1(output1).squeeze()
            * _, nsites = output1.shape
            output2 = torch.empty(
                (ntaxa, self.embedding_dim_2, nsites),
                device=self.device)
            for i, x in enumerate(output1.permute(2, 0, 1)):
                x = self.taxon_graph(x,
                                     batch=batches)  # TODO: use repeat or repeat_interleave?
                output2[..., i] = x

            # output = output.reshape((ntaxa, -1))
        output2 = self.f_2(output2).squeeze()
        output = self.output_layer(
            output2.flatten(1, -1).float(), batches)  # no trainable params
        return output

    def forward_debug(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        from .. import utils as u
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)
        y = u.njtree(data.y)
        input = data.x
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        print('raw', self.rf_distance(input, batches, y))
        with torch.cuda.amp.autocast():
            input = self.char_embed(input).squeeze().to(self.device)
            print('embed', self.rf_distance(input, batches, y))
            input = self.site_embed(input)
            print('site embed', self.rf_distance(
                input, batches, y))
            ntaxa, hidden_dim, nsites = input.shape
            output = torch.empty(
                (ntaxa, self.embedding_dim_1, nsites),
                device=self.device)
            for i, x in enumerate(input.permute(2, 0, 1)):
                x = self.site_graph(x)  # , edge_index)
                output[..., i] = x
                if i == 0:
                    print('slice', self.rf_distance(x, batches, y))
            print('site_graph', self.rf_distance(
                output.flatten(1, - 1), batches, y))
            output = self.f_1(output).squeeze()
            print('pooling', self.rf_distance(
                output.flatten(1, - 1), batches, y))
            * _, nsites = output.shape
            output2 = torch.empty(
                (ntaxa, self.embedding_dim_2, nsites),
                device=self.device)
            for i, x in enumerate(output.permute(2, 0, 1)):
                x = self.taxon_graph(x,
                                     batch=batches)  # TODO: use repeat or repeat_interleave?
                output2[..., i] = x

        print('taxon graph', self.rf_distance(output2, batches, y))
        output = self.output_layer(
            output2.flatten(1, -1).float().float(),
            batches)  # no trainable params
        return output


class SequentialConvAttentionNetwork(GraphNetwork):
    def __init__(self,
                 kernel: int = 200,
                 stride: int = 100,
                 gat_conv_layers: list = [1024]*4,
                 char_embedding_dim: int = EMBED_DIM,
                 dropout: float = .1,
                 heads: int = 8,
                 num_layers: int = 1,
                 graph_batch_norm: bool = False,
                 seq_embedding_layers: list = [512]*3,
                 output_layers: list = [256]*3,
                 output='metric',
                 device='cpu', **kwargs):
        """Embedding layer followed by resnet layer followed by num_layers GAT layers

        Args:
            gene_embedding (int, optional): _description_. Defaults to 256.
            hidden_channels (list, optional): _description_. Defaults to [256, 512, 1024, 1024].
            embed_dim (_type_, optional): _description_. Defaults to EMBED_DIM.
            resnet_model (str, optional): _description_. Defaults to 'resnet18'.
            dropout (float, optional): _description_. Defaults to .1.
            heads (int, optional): _description_. Defaults to 8.
            num_layers (int, optional): _description_. Defaults to 1.
            graph_batch_norm (bool, optional): _description_. Defaults to False.
            output (str, optional): _description_. Defaults to 'metric'.
            device1 (_type_, optional): If set, will put resnet (gene embedding) onto the specified device.
                Useful for long sequences that won't fit into memory. Defaults to None.
            device (str, optional): _description_. Defaults to 'cpu'.

        Raises:
            ValueError: _description_
        """

        self.debug = False
        in_channels, *hidden, output_channels = gat_conv_layers
        # if num_layers > 1 and in_channels != output_channels:
        #     raise ValueError(
        #         'input dim must match output dim for multilayer GAT')
        super().__init__()
        self.device = device

        seq_slice_dim = kernel*char_embedding_dim
        if len(seq_embedding_layers):
            if len(seq_embedding_layers) == 1:
                self.latent_dim = output_channels
                fc_net = build_fc_network(seq_slice_dim,
                                          self.latent_dim,
                                          nonlinearity=nn.ELU(),
                                          batch_norm=graph_batch_norm,
                                          )
            else:
                self.latent_dim = in_channels
                seq_embedding_layers = [seq_slice_dim] + \
                    seq_embedding_layers+[self.latent_dim]
                embedding_network = build_fc_network(
                    layers=seq_embedding_layers,
                    nonlinearity=nn.ELU(),
                    batch_norm=graph_batch_norm,
                )

        graph_layers = nn.ModuleList([GATNet(
            in_channels=output_channels,
            hidden_channels=hidden,
            output_channels=output_channels,
            heads=heads,
            batch_norm=graph_batch_norm,
            dropout=dropout)])
        for _ in range(num_layers-1):
            graph_layers.append(GATNet(
                in_channels=in_channels,
                hidden_channels=hidden,
                output_channels=output_channels,
                heads=heads,
                batch_norm=graph_batch_norm,
                dropout=dropout))

        modules = build_fc_network(layers=[self.latent_dim]+output_layers,
                                   nonlinearity=nn.ELU(),
                                   batch_norm=graph_batch_norm)

        self.add_module('embed', EmbedLayer(char_embedding_dim, dropout))

        self.add_module('conv', nn.Conv2d(in_channels=char_embedding_dim,
                                          out_channels=self.latent_dim,
                                          kernel_size=(kernel, 1), stride=stride))

        self.add_module('embedding_FC', embedding_network)

        self.add_module('graph_layers', graph_layers)

        self.add_module('output_FC', nn.Sequential(*modules))

        if output == 'metric':
            self.add_module('output_layer',
                            MetricDecoder(
                                clip=1e9, as_list=False))
        elif output == 'covariance':
            self.add_module('output_layer',
                            CovarianceDecoder(
                                clip=1e9, as_list=False))

        self.to(self.device)
        self.embed.to('cpu')

    def __repr__(self):
        return super().__repr__()+f"\ngraph layers:{', '.join(map(str,self.graph_layers))}"
        # self.set_devices(device)

    def forward(self, data):
        '''input x shape is [batches, ntaxa,ngenes, alignment_length]'''
        edge_index = data.edge_index.to(self.device, non_blocking=True)
        batches = data.batch.to(self.device, non_blocking=True)

        input = data.x
        ntaxa = input.shape[0]
        if not isinstance(input, torch.Tensor):
            input = torch.cat(input)
        with torch.cuda.amp.autocast():
            input = self.embed(input).to(self.device)

            output = torch.zeros((ntaxa, self.latent_dim), device=self.device)
            n = 0.
            for x in self.unfold(input).permute(2, 0, 1):
                x = self.embedding_FC(x)
                for graph_layer in self.graph_layers:  # dont need to cast to float since GATNet does this
                    x = graph_layer(x, edge_index)
                output += x
                n += 1
            output = self.output_FC(output/n)
        output = self.output_layer(
            output.float(), batches)  # no trainable params
        return output

        self.add_module('conv', nn.Conv2d(in_channels=char_embedding_dim,

                                          kernel_size=(kernel, 1), stride=stride))
