import torch 
import numpy as np 
import sys 
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
sys.path.append('../')
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s


def get_model_features(model,feat_list):
    
    model = model.cuda()
    model = model.eval()
    
    
    weight_features = []

    weight_distribution = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            
            weights = layer.weight.data.flatten().cpu().numpy()
            weight_distribution.extend(weights)

    

    if 'min' in feat_list:
        weight_features.append(np.min(weight_distribution))
        
    if 'max' in feat_list:
        weight_features.append(np.max(weight_distribution))
    
    if 'mean' in feat_list:
        weight_features.append(np.mean(weight_distribution))
    
    if 'std' in feat_list:
        weight_features.append(np.std(weight_distribution))
    
    if 'skewness' in feat_list:
        weight_features.append(skew(weight_distribution))
    
    if 'kurtosis' in feat_list:
        weight_features.append(kurtosis(weight_distribution))
    
    # if 'svd' in feat_list:
    #     _, s, _ = np.linalg.svd(raw_weight.cpu().numpy())
        
    #     # layer_feature.extend(s[0:2])
    #     weight_features.append(s[0])
    
    # if 'l2_norm' in feat_list:
        
    #     weight_features.append(torch.norm(weight_distribution,p=2).item())
    
    # if 'l1_norm' in feat_list:
    #     weight_features.append(torch.norm(weight_distribution,p=1).item())
    
    # if 'l_inf_norm' in feat_list:
    #     weight_features.append(torch.norm(weight_distribution,p=float('inf')).item())
    
    
    if len(weight_features) == 0:
        raise ValueError('No valid feature type provided')

    

    
    return weight_features
    
def get_weight_product(model,if_bias=True):
    
    model = model.cuda()
    model = model.eval()
    
    
    weights = []
    biases = []
    
    weight_product = None 
    
    bias_product = None 
    
    for layer in model.modules():
        
        if isinstance(layer, torch.nn.Linear):
            
            
            weights.append(layer.weight.data)
            biases.append(layer.bias.data)
            

    for i in range(len(weights)):
        
        weight = weights[i]
        bias = biases[i]

        
        if weight_product is None:
            weight_product = weight
        else:
            weight_product = torch.matmul(weight,weight_product)


        
        
        
        
        if bias_product is None:
            bias_product = bias.unsqueeze(1)
        
        
        else:
            bias_product = torch.matmul(weight,bias_product) + bias.unsqueeze(1)    
        
    
    if if_bias:
        
        weight_features = torch.cat((weight_product,bias_product),dim=1)
        
        weight_features/=weight_features.norm(p=1)
        weight_features = weight_features.flatten().cpu().detach().numpy()
        # tmp = torch.cat((weight_product,bias_product),dim=1)
        # mat = torch.matmul(tmp, tmp.transpose(0,1))
        
        # eigvalue = torch.linalg.eigvals(mat).float().cpu().detach().numpy()
        # print(eigvalue)
        # plt.hist(eigvalue)
        # plt.savefig('eigvalue.png')
        # plt.clf()
        
        
        
        
    else: 
        weight_features = weight_product
        weight_features/=weight_features.norm(p=1)
        weight_features = weight_features.flatten().cpu().detach().numpy()
    
    
    
    
    
    weight_features = weight_features
    
    
    
    

    
    
    
    return weight_features
    

def get_layer_weight_correlation(model,layer_selection):
    
    model = model.cuda()
    model = model.eval()

    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            
            weights = layer.weight.data.transpose(0,1)
            
            cor_mat = torch.zeros(weights.shape[1],weights.shape[1]).cuda()
            
            for i in range(weights.shape[1]):
                for j in range(weights.shape[1]):
                    if i != j:
                        cor_mat[i,j] = torch.cosine_similarity(weights[:,i],weights[:,j],dim=0)
                        
            
            print(cor_mat.max())
            print(cor_mat.min())
            print(cor_mat.mean())
            print(cor_mat.std())
        
            print('=====')
            
    
    exit()
    
            
    
    
    

def get_layer_features(model,feat_list,layer_selection):
    
    #* feat types: min, max, mean, std, skewness, kurtosis, svd, p_norm
    
    
    model = model.cuda()
    model = model.eval()
    
    
    
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
                
                # layer_feature.extend(s[0:2])
                layer_feature.append(s[0])
            
            if 'l2_norm' in feat_list:
                
                layer_feature.append(torch.norm(raw_weight,p=2).item())
            
            if 'l1_norm' in feat_list:
                layer_feature.append(torch.norm(raw_weight,p=1).item())
            
            if 'l_inf_norm' in feat_list:
                layer_feature.append(torch.norm(raw_weight,p=float('inf')).item())
            
            
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
def align_layer_features(feats,padding=True,padding_value=-1,max_len=7):
    
    # concate feats of different layers 
    
    # padding feats to max length
    
    align_feat = None 
    
    
    num_feat_layer = len(feats)
    num_padding_layer = max_len - num_feat_layer
    feat_dim = len(feats[0])
    
    
    

    
    
    if padding == False:
    
    
        for layer_id in range(len(feats)):
            
            
            
            
            if align_feat is None:
                align_feat = feats[layer_id]
                continue
                
            align_feat.extend(feats[layer_id])
            
                
        
        
        
    
    
    else: 
        
        for layer_id in range(len(feats)):
            
            
            
            
            if align_feat is None:
                align_feat = feats[layer_id]
                continue
                
            align_feat.extend(feats[layer_id])
        

        insert_pos = feat_dim * (num_feat_layer - 1)
        
        
        for padding_id in range(num_padding_layer):
            
            
            padding_feat = [padding_value] * feat_dim
            
            
            for i in range(feat_dim):
                align_feat.insert(insert_pos + i,padding_feat[i])
            
            
            # align_feat.extend(padding_feat)
        
        

    
    align_feat = np.array(align_feat)
    
    
    
    
        
            
            
        
        
        
    # print(align_feat.shape)
    
    return align_feat
    
    
    






if __name__ == '__main__':
    
    model_filepath = '/data3/share/trojai/trojai-round12-cyber-v1-dataset/models/id-00000062/model.pt'
    model = torch.load(model_filepath).cuda()
    model = model.eval()
    # feat_list = ['min','max','mean','std','skewness','kurtosis','svd','norm']
    # feat = get_layer_features(model,feat_list=feat_list,layer_selection='all')
    # align_layer_features(feat)
    # get_layer_weight_correlation(model,layer_selection='all')
    weight_features = get_weight_product(model)
    
    plt.hist(weight_features)
    
    plt.savefig('weight_product_62.png')
    

    
    