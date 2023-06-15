from utils import *
from model import *

paths = {
    'mirna_df': 'data/our_data/nodes/mirnas.tsv',
    'disease_df': 'data/our_data/nodes/diseases.tsv',
    'mrna_df': 'data/our_data/nodes/mrnas.tsv',
    'train_val_test': 'data/our_data/train_val_test_with_9block_test_sparse.npy',
    'graph_without_mid': 'data/our_data/data_without_mid.pt',
    'graph_feature_randn_without_mid': 'data/our_data/data_feature_randn_without_mid.pt',
    'model_save_folder': 'files/models/',
    'model_save_path': ''
}

settings = {
    'which_graph': 'graph_without_mid',
    'feature_ablation_type': 3,
    'feature_randn': False,

    'num_neighbors': [-1] * 4,
    'k': 4,
    'dim': 64,
    'num_heads': 4,
    'num_layers': 2,
    'group_type': 'sum',
    
    'batch_size': 256,
    'epoch_num': 50,
    'lr': 0.001,

    # test_evaluation_and_ablation_study
    'early_stop_type': 'self_fitting',
    'patience': 1,
    'train_modes': ['train', 'val'],
    'evaluate_modes': ['test'],
    # 'test_nega_ratio': 1,

    # val_hyperparameter_selection
    # 'early_stop_type': 'self_fitting',
    # 'patience': 5,
    # 'train_modes': ['train'],
    # 'evaluate_modes': ['val']

    # case_study
    # 'early_stop_type': 'self_fitting',
    # 'patience': 1,
    # 'train_modes': ['train', 'val', 'test'],
    # 'evaluate_modes': []
}

# Let's go!
# Please set the hyperparameters correctly and write loop to implement experiments

# Hyper-Parameter Selection:  
# train_modes': ['train'], 'evaluate_modes': ['val'], 'early_stop_type': 'val_fitting', 'patience': 5
# grid search, repeat 5 times
# dim: 32, **64**, 128, 256
# num_heads: 1, 2, **4**, 8
# num_layers: 1, **2**, 3
# **the best choice**

# Evaluation and Ablation Study:
# 'train_modes': ['train', 'val'], 'evaluate_modes': ['test'], 'early_stop_type': 'self_fitting', 'patience': 1
# ablation of each part
# feature_ablation_type: 1, 2, **3**, 4
# feature_randn: True, **False**
# num_layers: 1, **2**, 3, 4
# **the best choice**

# Case Study:
# 'train_modes': ['train', 'val', 'test'], 'evaluate_modes': [], 'early_stop_type': 'self_fitting', 'patience': 1, 'test_nega_ratio': 1
# Train Only! Please delete the evaluation part
# feature_ablation_type: **3**
# feature_randn: **False**
# num_layers: **2**

print('Train Modes:', settings['train_modes'])
print('Evaluate Modes:', settings['evaluate_modes'])
print('Early Stop Type:', settings['early_stop_type'])
print('Patience:', settings['patience'])

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
criterion = nn.BCELoss()
                        
model = Model(k=settings['k'], dim=settings['dim'], num_heads=settings['num_heads'], num_layers=settings['num_layers'],
            group_type=settings['group_type'], feature_ablation_type=settings['feature_ablation_type']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'])

if settings['feature_randn']:
    settings['which_graph'] = 'graph_feature_randn_without_mid'
else:
    settings['which_graph'] = 'graph_without_mid'

# Load data and train the model
data, best_weights = get_data_and_train(paths, settings, optimizer, device, model, criterion)

model.load_state_dict(best_weights)

# Attention! Save the model in where you want.
# While excuting loop, name the model carefully to avoid non-ideal overwritten!
paths['model_save_path'] = paths['model_save_folder'] + 'egpmda.pth'
torch.save({
    'settings': settings,
    'best_weights': best_weights
    }, paths['model_save_path'])

# Evaluation
settings['test_nega_ratio'] = 1
pred, label, test_sparse_edge_label_index = evaluate(data, paths, settings, device, model)
AUC, AUPR, ACC, P, R, F1 = get_metrics(label, pred, 0.5)
sparse_R = get_sparse_r(pred, test_sparse_edge_label_index, 0.5)

