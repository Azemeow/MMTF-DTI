import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class GradientReversalLayer(torch.autograd.Function):
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.alpha * -grad_output, None


class TextFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=3, dropout=0.3):
        super(TextFeatureExtractor, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(1)
        transformer_out = self.transformer_encoder(x)
        transformer_out = transformer_out.squeeze(1)
        return transformer_out


class GraphStructureEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(GraphStructureEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.layers.append(GraphConv(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, graph, h):
        for conv in self.layers:
            h = conv(graph, h)
            h = self.activation(h)
            h = self.dropout(h)
        graph.ndata['h'] = h
        return h


class RecursiveFusionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, num_heads=2, dropout=0.3):
        super(RecursiveFusionModule, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, hidden_dim)

        # Attention layer for dynamic weighting
        self.attention_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear(x)
        transformer_out = self.transformer_encoder(x)

        # Compute dynamic weights
        weights = self.attention_weight(transformer_out)
        weighted_features = weights * transformer_out
        transformer_out = weighted_features.sum(dim=1)

        return transformer_out


class CrossAttentionFusionModule(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, num_heads=2, dropout=0.2):
        super(CrossAttentionFusionModule, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(input_dim1 + output_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        # Apply cross-attention: x1 attends to x2
        attn_output, _ = self.cross_attention(x1, x2, x2)
        attn_output = attn_output.squeeze(1)

        # Concatenate and project
        fused = torch.cat((x1.squeeze(1), attn_output), dim=1)
        fused = self.linear(fused)
        fused = self.relu(fused)

        return fused


class MMTFDTINet(nn.Module):
    def __init__(self, esm_hidden_size, mol2vec_size, compound_dim=128, protein_dim=128,
                 h_dim=128, h_out=4, dropout=0.2, graph_feat_dim=72):
        super(MMTFDTINet, self).__init__()
        self.dtype = torch.float32

        # Graph feature dimension alignment
        self.compound_feature_projection = nn.Linear(72, 128)
        self.protein_feature_projection = nn.Linear(72, 128)

        # Drug text feature extraction
        self.compound_text_transformer = TextFeatureExtractor(
            input_dim=mol2vec_size,
            hidden_dim=compound_dim,
            num_heads=4,
            dropout=dropout
        )

        # Drug graph structure feature extraction
        self.compound_graph_encoder = GraphStructureEncoder(
            in_dim=128,
            hidden_dim=128,
            num_layers=3,
            dropout=dropout
        )

        # Feature dimension alignment
        self.compound_proj = nn.Sequential(
            nn.Linear(compound_dim + 128, 128),
            nn.ReLU()
        )
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim + 128, 128),
            nn.ReLU()
        )

        # Protein text feature extraction
        self.protein_text_transformer = TextFeatureExtractor(
            input_dim=esm_hidden_size,
            hidden_dim=protein_dim,
            num_heads=4,
            dropout=dropout
        )

        # Protein graph structure feature extraction
        self.protein_graph_encoder = GraphStructureEncoder(
            in_dim=128,
            hidden_dim=128,
            num_layers=3,
            dropout=dropout
        )

        # Additional linear layers for structure feature projection
        self.compound_proj_extra = nn.Linear(128, 128)
        self.protein_proj_extra = nn.Linear(128, 128)

        # Feature projection to 256 dimensions
        self.compound_projection = nn.Sequential(
            nn.Linear(compound_dim + protein_dim, 256),
            nn.ReLU()
        )
        self.protein_projection = nn.Sequential(
            nn.Linear(compound_dim + protein_dim, 256),
            nn.ReLU()
        )

        # Recursive fusion modules
        self.compound_recursive_fusion = RecursiveFusionModule(
            input_dim=256,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=dropout
        )
        self.protein_recursive_fusion = RecursiveFusionModule(
            input_dim=256,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=dropout
        )

        # Cross-attention fusion module
        self.fusion = CrossAttentionFusionModule(
            input_dim1=128,
            input_dim2=128,
            output_dim=128,
            num_heads=4,
            dropout=dropout
        )

        # Transformer fusion network
        self.fusion_network = RecursiveFusionModule(
            input_dim=128,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=dropout
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, compound_graph, protein_graph, chem_features, prot_features, domain_adapt=False):
        # Graph feature dimension alignment
        if compound_graph.ndata['h'].shape[1] == 72:
            compound_graph.ndata['h'] = self.compound_feature_projection(compound_graph.ndata['h'])

        if protein_graph.ndata['h'].shape[1] == 72:
            protein_graph.ndata['h'] = self.protein_feature_projection(protein_graph.ndata['h'])

        # Drug text feature extraction
        compound_text_rep = self.compound_text_transformer(chem_features)

        # Drug structure feature extraction
        compound_struct_rep = self.compound_graph_encoder(compound_graph, compound_graph.ndata['h'])
        compound_struct_rep = dgl.mean_nodes(compound_graph, 'h')
        compound_struct_rep = self.compound_proj_extra(compound_struct_rep)

        # Protein text feature extraction
        protein_text_rep = self.protein_text_transformer(prot_features)

        # Protein structure feature extraction
        protein_struct_rep = self.protein_graph_encoder(protein_graph, protein_graph.ndata['h'])
        protein_struct_rep = dgl.mean_nodes(protein_graph, 'h')
        protein_struct_rep = self.protein_proj_extra(protein_struct_rep)

        # Fuse drug and protein text features
        text_rep = torch.cat((compound_text_rep, protein_text_rep), dim=1)
        text_rep = self.compound_projection(text_rep)
        fused_text_rep = self.compound_recursive_fusion(text_rep.unsqueeze(1))

        # Fuse drug and protein structure features
        struct_rep = torch.cat((compound_struct_rep, protein_struct_rep), dim=1)
        struct_rep = self.protein_projection(struct_rep)
        fused_struct_rep = self.protein_recursive_fusion(struct_rep.unsqueeze(1))

        # Cross-attention fusion of text and structure features
        fused_rep = self.fusion(fused_text_rep, fused_struct_rep)

        # Transformer fusion network
        fused_rep_expanded = fused_rep.unsqueeze(1)
        fused_rep_transformed = self.fusion_network(fused_rep_expanded)

        # Domain adaptation
        if domain_adapt:
            reversed_feature = GradientReversalLayer.apply(fused_rep_transformed, 0.5)
            domain_pred = self.domain_classifier(reversed_feature)
        else:
            domain_pred = None

        # Classifier output
        output = self.classifier(fused_rep_transformed)

        return {
            'pred': output,
            'domain': domain_pred,
            'features': fused_rep_transformed
        }


class MMTF_DTI(nn.Module):
    def __init__(self, esm_hidden_size, mol2vec_size, compound_dim=128, protein_dim=128, out_dim=1):
        super(MMTF_DTI, self).__init__()
        self.feature_extractor = MMTFDTINet(
            esm_hidden_size=esm_hidden_size,
            mol2vec_size=mol2vec_size,
            compound_dim=compound_dim,
            protein_dim=protein_dim,
            h_dim=128,
            h_out=4,
            dropout=0.1
        )

    def forward(self, compound_graph, protein_graph, chem_features, prot_features, domain_adapt=False):
        return self.feature_extractor(
            compound_graph, protein_graph, chem_features, prot_features, domain_adapt
        )
