import torch 
import os 
import matplotlib.pyplot as plt
import numpy as np 
import sys 
from tqdm import tqdm
sys.path.append('../')
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s

# visualize distribution of weight values of a given model in each layer 

def get_weight_distribution(model):
    
    weight_distribution = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            
            weights = layer.weight.data.flatten().cpu().numpy()
            weight_distribution.extend(weights)
            # weight_distribution.append(weights)
            
    return weight_distribution


def visualize_weight_distribution(weights,model_filepath,n_bins=50):
    
    
        
    # weight = weights[layer_id]
    # counts, bins = np.histogram(weight)
    
    # plt.stairs(counts, bins)
    plt.hist(weights, bins=n_bins)
    plt.savefig('../scratch/hist_{}_all.png'.format(model_filepath))
    
    
        
        
    
    
    

if __name__ == '__main__':
    
    model_filepath = '/data3/share/trojai/trojai-round12-cyber-v1-dataset/models/id-00000009/model.pt'
    model = torch.load(model_filepath).cuda()
    model = model.eval()
    
    visualize_weight_distribution(weights=get_weight_distribution(model),model_filepath=model_filepath.split('/')[-2])
    
    
    