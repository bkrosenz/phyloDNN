
class EdgeConvNet(GraphNetwork):
    def __init__(self,
                 gene_embedding=256,
                 hidden_channels=[256, 512, 512, 1024],
                 ks=[None, 4, 3, 2],
                 embed_dim=EMBED_DIM,
                 resnet_model='resnet18',
                 edge_net_depth=3,
                 dropout=.1,
                 heads=8,
                 graph_batch_norm=False,
                 device='cpu'):
        '''Embedding layer followed by resnet layer followed by num_layers GAT layers'''
        super().__init__()
        self.embed = EmbedLayer(embed_dim, dropout)
        self.gene_encoder = resnet_models[resnet_model](in_channels=embed_dim,
                                                        n_classes=gene_embedding)
        self.elu = nn.functional.elu
        self.norm1 = nn.BatchNorm1d(gene_embedding)
        in_dim = gene_embedding
        self.graph_layer = nn.Sequential()
        for k, out_dim in zip(ks, hidden_channels):
            layers = [fc_layer(2*in_dim, 2 * in_dim, graph_batch_norm)
                      for _ in range(edge_net_depth)]
            edge_mapper = nn.Sequential(
                *layers, fc_layer(in_dim, out_dim, graph_batch_norm)
            )
            if k is not None:
                edge_gcn = DynamicEdgeConv(edge_mapper, k=k, aggr='max')
            else:
            edge_gcn = EdgeConv(edge_mapper, aggr='max')

            self.graph_layer.add_module(edge_gcn)
            in_dim = out_dim

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
        x = self.norm1(x)
        x = self.graph_layer(x, edge_index)
        x = self.output_layer(x, batches)  # no trainable params
        return x
