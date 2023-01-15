import torch 
import numpy as np 
from spicy.stats import kurtosis, skew


def get_layer_features(model,feat_list):
    
    #* feat types: min, max, mean, std, skewness, kurtosis, svd, p_norm
    weight_features = []
    for layer in model.modules():
        layer_feature = []
        if isinstance(layer, torch.nn.Linear):
            raw_weight = layer.weight.data
            flatten_weight = raw_weight.flatten().cpu().numpy()
            
            
            if 'min' in feat_list:
                layer_feature.append(np.min(flatten_weight))
                
            if 'max' in feat_list:
                layer_feature.append(np.max(flatten_weight))
            
            if 'mean' in feat_list:
                layer_feature.append(np.mean(flatten_weight))
            
            if 'std' in feat_list:
                layer_feature.append(np.std(flatten_weight))
            
            if 'skewness' in feat_list:
                layer_feature.append(skew(flatten_weight))
            
            if 'kurtosis' in feat_list:
                layer_feature.append(kurtosis(flatten_weight))
            
            if 'svd' in feat_list:
                _, s, _ = np.linalg.svd(raw_weight.cpu().numpy())
                layer_feature.extend(s)
            
            if 'norm' in feat_list:
                
                layer_feature.append(torch.norm(raw_weight,p=2).item())
            
            
            if len(layer_feature) == 0:
                raise ValueError('No valid feature type provided')

            
            weight_features.append(layer_feature)
            print(len(layer_feature))
    
    
    print(len(weight_features))
    return weight_features
            
            
                
#? align features of different models to the same scale
def align_layer_features(feats,temp_dim=7):
    raise NotImplementedError






