import torch 
import os 
import numpy as np 
import json 
import copy 
from tqdm import tqdm
import logging


def model_augmentation(model,scaler,model_dirpath,aug_num,delta_scale,cri_thres=0.95):
    
    
    # logging.info('Model Augmentation via random noises')
    
    
    aug_model_list = []
    
    clean_data_dirpath = os.path.join(model_dirpath, 'clean-example-data')
    poison_data_dirpath = os.path.join(model_dirpath, 'poisoned-example-data')   
    
    
    labels = []
    inputs = []
    



    for clean_data in os.listdir(clean_data_dirpath):
        clean_data_filepath = os.path.join(clean_data_dirpath, clean_data)
        if clean_data_filepath.endswith('.npy'):
            
            
            raw_clean_data = np.load(clean_data_filepath).reshape(1, -1)
            clean_data = torch.from_numpy(scaler.transform(raw_clean_data.astype(float))).float().squeeze()
            
            clean_label_filepath = clean_data_filepath + '.json'
            clean_label = json.load(open(clean_label_filepath))
            
            inputs.append(clean_data)
            labels.append(clean_label)
            
    
    clean_inputs = torch.stack(inputs).cuda()
    clean_labels = torch.tensor(labels).cuda()
    
    
    if os.path.exists(poison_data_dirpath)  == True:
        
        poison_labels = []
        poison_inputs = []

    

        for poison_data in os.listdir(poison_data_dirpath):
            poison_data_filepath = os.path.join(poison_data_dirpath, poison_data)
            if poison_data.endswith('.npy'):
                
                raw_poison_data = np.load(poison_data_filepath).reshape(1, -1)
                
                poison_data = torch.from_numpy(scaler.transform(raw_poison_data.astype(float))).float().squeeze()
                    
                poison_label_filepath = poison_data_filepath + '.json'
                poison_label = json.load(open(poison_label_filepath))
                

                poison_inputs.append(poison_data)
                poison_labels.append(poison_label)
        
        
        poison_inputs = torch.stack(poison_inputs).cuda()
        poison_labels = torch.tensor(poison_labels).cuda()
    
    for aug_id in range(aug_num):
        aug_model = copy.deepcopy(model)

        for layer_id, (name, param) in enumerate(aug_model.named_parameters()):
            if 'weight' in name or 'bias' in name:
                mean = param.mean()
                std = param.std()
                
                delta = torch.rand_like(param) * std + mean
                param.data = param.data + delta * delta_scale
                
        
        
        clean_preds = aug_model(clean_inputs).argmax(dim=1)    
        clean_acc = (clean_preds == clean_labels).float().mean()
        
        if os.path.exists(poison_data_dirpath)  == True:
            poison_preds = aug_model(poison_inputs).argmax(dim=1)
            asr = (poison_preds != poison_labels).float().mean()
        
            
            if asr > cri_thres and clean_acc > cri_thres:
                aug_model_list.append(aug_model.cpu())
                
        
        else:
            
            if clean_acc > cri_thres:
                aug_model_list.append(aug_model.cpu())
    
    
    # logging.info('Number of augmented models: {}'.format(len(aug_model_list)))
    
    return aug_model_list