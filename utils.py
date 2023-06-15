import torch
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import copy


def id_to_index(ids, mirna_id, disease_id):
    
    mirna_disease_df = pd.DataFrame({
        'mirna': [tmp_id[0] for tmp_id in ids],
        'disease': [tmp_id[1] for tmp_id in ids]
    })
    
    mid_mirna_index = pd.merge(mirna_disease_df['mirna'], mirna_id, left_on='mirna', right_on='mirna_id', how='left')
    mid_mirna_index = torch.from_numpy(mid_mirna_index['mapped_index'].values)
    mid_disease_index = pd.merge(mirna_disease_df['disease'], disease_id, left_on='disease', right_on='disease_id', how='left')
    mid_disease_index = torch.from_numpy(mid_disease_index['mapped_index'].values)
    
    return torch.stack([mid_mirna_index, mid_disease_index], dim=0)


def get_data(path, posi_edge_index):
    
    # It can be:
    # 'which_graph': 'graph_without_mid',
    # 'which_graph': 'graph_feature_randn_without_mid'
    data = torch.load(path)
    
    # Attention! Although we load the graph with miRNA-disease associations here, they are not utilized by our method.
    # The meta-relations ('mirna', 'association', 'disease') and ('disease', 'rev_association', 'mirna') are not included in the learning and prediction process. 
    # See model.py, the ablation of topological features is controlled by the settings of meta-relations.
    data['mirna', 'association', 'disease'].edge_index = posi_edge_index
    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)
    
    return data


def get_edge_index(paths, settings, modes=['train']):
    
    mirnas_df = pd.read_table(paths['mirna_df'])
    assert len(mirnas_df['Accession']) == len(mirnas_df['Accession'].unique())
    mirna_id = pd.DataFrame(data={
        'mirna_id': mirnas_df['Accession'],
        'mapped_index': pd.RangeIndex(len(mirnas_df['Accession']))
    })
    
    diseases_df = pd.read_table(paths['disease_df'])
    assert len(diseases_df['ID']) == len(diseases_df['ID'].unique())
    disease_id = pd.DataFrame(data={
        'disease_id': diseases_df['ID'],
        'mapped_index': pd.RangeIndex(len(diseases_df['ID']))
    })
    
    # split by year: train: year <= 2019; val: year = 2020 or 2021; test: year = 2021 or 2022
    train_val_test = np.load(paths['train_val_test'], allow_pickle=True).item()

    posi_samples = []
    nega_samples = []
    for mode in modes:
        posi_samples = posi_samples + train_val_test[mode]['posi']
        if mode in ['train', 'val']:
            nega_samples = nega_samples + train_val_test[mode]['nega']
            sparse_edge_label_index = None
        else:
            if settings['test_nega_ratio'] != None:
                nega_samples = nega_samples + train_val_test[mode]['nega'][: len(train_val_test[mode]['posi']) * settings['test_nega_ratio']]
            else:
                nega_samples = nega_samples + train_val_test[mode]['nega']
            sparse_edge_label_index = train_val_test['test_sparse']

    posi_edge_index = id_to_index(posi_samples, mirna_id, disease_id)
    nega_edge_index = id_to_index(nega_samples, mirna_id, disease_id)
    
    return posi_edge_index, nega_edge_index, sparse_edge_label_index


def get_data_and_train(paths, settings, optimizer, device, model, criterion):

    posi_edge_index, nega_edge_index, _ = get_edge_index(paths=paths, settings=settings, modes=settings['train_modes'])
    
    data = get_data(paths[settings['which_graph']], posi_edge_index)
    edge_label_index = torch.cat([posi_edge_index, nega_edge_index], 1)
    edge_label = torch.Tensor([1] * posi_edge_index.shape[1] + [0] * nega_edge_index.shape[1])

    # print(data)
    # print(edge_label_index.shape, edge_label.shape)

    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=settings['num_neighbors'],
        edge_label_index=(('mirna', 'association', 'disease'), edge_label_index),
        edge_label=edge_label,
        batch_size=settings['batch_size'],
        shuffle=True
    )

    min_loss = 1
    # max_auc = 0
    max_acc = 0
    patience_count = 0
    for epoch in range(1, settings['epoch_num'] + 1):
        total_loss = total_examples = 0
        for sampled_data in train_loader:
            optimizer.zero_grad()
            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)
            loss = criterion(pred, sampled_data['mirna', 'association', 'disease'].edge_label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        avg_loss = total_loss / total_examples
        if settings['early_stop_type'] == 'self_fitting':
            if avg_loss < min_loss:
                min_loss = avg_loss
                patience_count = 0
                # torch.save(model, paths['model_save_path'])
                best_weights = copy.deepcopy(model.state_dict())
            else:
                patience_count = patience_count + 1
            print(f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, Min Loss: {min_loss:.4f}")
            if patience_count > settings['patience']:
                break
        elif settings['early_stop_type'] == 'val_fitting':
            val_pred, val_label, _ = evaluate(data, paths, settings, device, model)
            # auc = roc_auc_score(val_label, val_pred)
            # if auc > max_auc:
            #     max_auc = auc
            acc = accuracy_score(val_label, np.array([1 if p > 0.5 else 0 for p in val_pred]))
            if acc > max_acc:
                max_acc = acc
                patience_count = 0
                # torch.save(model, paths['model_save_path'])
                best_weights = copy.deepcopy(model.state_dict())
            else:
                patience_count = patience_count + 1
            # print(f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, Validation AUC: {auc:.4f}, Max Validation AUC: {max_auc:.4f}")
            print(f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, Validation ACC: {acc:.4f}, Max Validation ACC: {max_acc:.4f}")
            if patience_count > settings['patience']:
                break
                
    return data, best_weights


def evaluate(data, paths, settings, device, model):

    posi_edge_index, nega_edge_index, sparse_edge_label_index = get_edge_index(paths=paths, settings=settings, modes=settings['evaluate_modes'])
    
    edge_label_index = torch.cat([posi_edge_index, nega_edge_index], 1)
    edge_label = torch.Tensor([1] * posi_edge_index.shape[1] + [0] * nega_edge_index.shape[1])

    evaluate_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=settings['num_neighbors'],
        edge_label_index=(('mirna', 'association', 'disease'), edge_label_index),
        edge_label=edge_label,
        batch_size=settings['batch_size'] * 4,
        shuffle=False
    )

    model.eval()
    preds = []
    labels = []
    for sampled_data in evaluate_loader:
        with torch.no_grad():
            sampled_data = sampled_data.to(device)
            preds.append(model(sampled_data))
            labels.append(sampled_data['mirna', 'association', 'disease'].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    label = torch.cat(labels, dim=0).cpu().numpy()

    return pred, label, sparse_edge_label_index


# "which_threshold" should be in {0.5, 5, 10}
def get_metrics(labels, preds, which_threshold):
    AUC = roc_auc_score(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    AUPR = auc(recall, precision)

    if which_threshold == 0.5:
        threshold = 0.5
    else:
        threshold = np.percentile(preds, 100 - which_threshold)

    top_n_preds = np.array([1 if p > threshold else 0 for p in preds])
    
    ACC = accuracy_score(labels, top_n_preds)
    P = precision_score(labels, top_n_preds)
    R = recall_score(labels, top_n_preds)
    F1 = f1_score(labels, top_n_preds)

    print(round(AUC, 3), round(AUPR, 3), round(ACC, 3), round(P, 3), round(R, 3), round(F1, 3))
    return AUC, AUPR, ACC, P, R, F1, threshold


# "which_threshold" should be in {0.5, 5, 10}
def get_sparse_r(preds, sparse_edge_label_index, which_threshold):

    sparse_R = []

    if which_threshold == 0.5:
        threshold = 0.5
    else:
        threshold = np.percentile(preds, 100 - which_threshold)

    for sparse_type in sparse_edge_label_index:
        test_sparse_edge_label_index = sparse_edge_label_index[sparse_type]
        sparse_preds = preds[: len(test_sparse_edge_label_index)]
        sparse_preds = sparse_preds[test_sparse_edge_label_index]
        sparse_labels = np.array([1] * len(sparse_preds))
        
        top_n_sparse_preds = np.array([1 if p > threshold else 0 for p in sparse_preds])
        sparse_R.append(recall_score(sparse_labels, top_n_sparse_preds))

    print(round(sparse_R[0], 3), round(sparse_R[1], 3), round(sparse_R[2], 3), \
          round(sparse_R[3], 3), round(sparse_R[4], 3), round(sparse_R[5], 3), round(sparse_R[6], 3))
    return sparse_R