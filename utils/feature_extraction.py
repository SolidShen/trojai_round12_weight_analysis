import torch 
import numpy as np 
import sys 
from scipy.stats import kurtosis, skew
sys.path.append('../')
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s



def get_layer_features(model,feat_list,layer_selection):
    
    #* feat types: min, max, mean, std, skewness, kurtosis, svd, p_norm
    
    
    
    
    weight_features = []
    
    selected_weight_features = []
    
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
            
    
    if layer_selection == 'all':
        selected_weight_features = weight_features
    
    elif layer_selection == None:
        raise ValueError('No layer selection provided')
    
    else:
        for layer_id in layer_selection:
            selected_weight_features.append(weight_features[layer_id])
    
    # print(len(selected_weight_features))
    
    
    return selected_weight_features
            
            
                
#? align features of different models to the same scale
def align_layer_features(feats):
    
    # concate feates of different layers 
    
    align_feat = None 
    
    
    for layer_id in range(len(feats)):
        
        
        
        
        if align_feat is None:
            align_feat = feats[layer_id]
            continue
            
        align_feat.extend(feats[layer_id])
        
            
    
    
    align_feat = np.array(align_feat)
    
    # print(align_feat.shape)
    
    return align_feat
    
    
    






if __name__ == '__main__':
    
    model_filepath = '/data3/share/trojai/trojai-round12-cyber-v1-dataset/models/id-00000007/model.pt'
    model = torch.load(model_filepath).cuda()
    model = model.eval()
    feat_list = ['min','max','mean','std','skewness','kurtosis','svd','norm']
    feat = get_layer_features(model,feat_list=feat_list,layer_selection=[-1])
    align_layer_features(feat)
    