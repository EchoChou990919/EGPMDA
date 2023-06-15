import torch
from torch import Tensor
import torch.nn as nn

from torch_geometric.data import HeteroData

from torch_geometric.nn import HGTConv
# Comment the last line and uncomment the following line for case study
# from hgt_conv_case_study import HGTConv



# The Node Feature Encoder Layer
class Embedding(torch.nn.Module):
    def __init__(self, k, dim, ablation_type):
        super().__init__()
        
        self.mirna_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(k, 4))
        self.mirna_emb = nn.Linear(235 - k + 1, dim)
        self.disease_emb = nn.Linear(1536, dim)
        self.ablation_type = ablation_type
        if ablation_type > 2:
            self.mrna_emb = nn.Linear(1536, dim)
    
    def forward(self, x_dict):
        
        mirna_x = x_dict['mirna'].unsqueeze(1)
        mirna_x = self.mirna_conv(mirna_x).squeeze(-1).squeeze(1)
        mirna_x = self.mirna_emb(mirna_x)
        disease_x = self.disease_emb(x_dict['disease'])
        
        embedding_x_dict = {
            'mirna': mirna_x,
            'disease': disease_x
        }
        
        if self.ablation_type > 2:
            mrna_x = self.mrna_emb(x_dict['mrna'])
            embedding_x_dict['mrna'] = mrna_x
            
        return embedding_x_dict

# The Graph Neural Network Layers
class GNN(torch.nn.Module):
    def __init__(self, dim, num_heads, num_layers, group_type, feature_ablation_type):
        super().__init__()

        self.feature_ablation_type = feature_ablation_type

        metadata = (['mirna', 'disease'],
                    [('mirna', 'family', 'mirna'),
                     ('disease', 'fatherson', 'disease'),
                     ])
        
        if feature_ablation_type > 2:
            metadata[0].append('mrna')
            metadata[1].append(('mrna', 'group', 'mrna'))
            metadata[1].append(('mirna', 'association', 'mrna'))
            metadata[1].append(('mrna', 'association', 'disease'))
            metadata[1].append(('mrna', 'rev_association', 'mirna'))
            metadata[1].append(('disease', 'rev_association', 'mrna'))
        
        if feature_ablation_type > 3:
            metadata[1].append(('mirna', 'association', 'disease'))
            metadata[1].append(('disease', 'rev_association', 'mirna'))
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(in_channels=dim, out_channels=dim, metadata=metadata, heads=num_heads, group=group_type)
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        
        conv_edge_index_dict = {
            ('mirna', 'family', 'mirna'): edge_index_dict[('mirna', 'family', 'mirna')],
            ('disease', 'fatherson', 'disease'): edge_index_dict[('disease', 'fatherson', 'disease')],
        }
        
        if self.feature_ablation_type > 2:
            conv_edge_index_dict[('mirna', 'association', 'mrna')] = edge_index_dict[('mirna', 'association', 'mrna')]
            conv_edge_index_dict[('mrna', 'rev_association', 'mirna')] = edge_index_dict[('mrna', 'rev_association', 'mirna')]
            conv_edge_index_dict[('mrna', 'association', 'disease')] = edge_index_dict[('mrna', 'association', 'disease')]
            conv_edge_index_dict[('disease', 'rev_association', 'mrna')] = edge_index_dict[('disease', 'rev_association', 'mrna')]
            conv_edge_index_dict[('mrna', 'group', 'mrna')] = edge_index_dict[('mrna', 'group', 'mrna')]
        
        if self.feature_ablation_type > 3:
            conv_edge_index_dict[('mirna', 'association', 'disease')] = edge_index_dict[('mirna', 'association', 'disease')]
            conv_edge_index_dict[('disease', 'rev_association', 'mirna')] = edge_index_dict[('disease', 'rev_association', 'mirna')]
        
        for conv in self.convs:
            x_dict = conv(x_dict, conv_edge_index_dict)
        
        return x_dict

# The Predictor Layer
class Classifier(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_mirna, x_disease, edge_label_index):
        
        edge_feature_mirna = x_mirna[edge_label_index[0]]
        edge_feature_disease = x_disease[edge_label_index[1]]
        
        edge_feature = torch.cat([edge_feature_mirna, edge_feature_disease], 1)
        
        result = self.mlp_layers(edge_feature).squeeze(-1)
        
        return result


# Ablation Tpye: 
# 1: miRNA sequence / disease text
# 2: 1 + miRNA family associations / disease fatherson associations
# **3**: 1 + 2 + miRNA / disease - mRNA associations
# 4: 1 + 2 + 3 + existing miRNA - disease associations

class Model(torch.nn.Module):
    def __init__(self, k, dim, num_heads, num_layers, group_type, feature_ablation_type):
        super().__init__()
        
        self.feature_ablation_type = feature_ablation_type
        self.embeddings = Embedding(k, dim, feature_ablation_type)
        if feature_ablation_type > 1:
            self.gnn = GNN(dim, num_heads, num_layers, group_type, feature_ablation_type)
        self.classifier = Classifier(dim)

    def forward(self, data: HeteroData) -> Tensor:
        
        x_dict = {
            'mirna': data['mirna'].x.float(),
            'disease': data['disease'].x.float()
        }
        if self.feature_ablation_type > 2:
            x_dict['mrna'] = data['mrna'].x.float()
        
        x_dict = self.embeddings(x_dict)
        
        if self.feature_ablation_type > 1:
            x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict['mirna'],
            x_dict['disease'],
            data['mirna', 'association', 'disease'].edge_label_index,
        )

        return pred