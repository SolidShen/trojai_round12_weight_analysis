import torch 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score,log_loss
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from utils.feature_extraction import get_weight_product
from utils.augmentation import model_augmentation
import numpy as np 
import logging

def cross_validation(clf_model,scaler,models,labels,model_names,if_bias,augmentation,cv=5):
    
    logging.info('Cross Validation')
    
    avg_acc = 0
    avg_roc = 0
    avg_loss = 0 
    
    kf = KFold(n_splits=cv,shuffle=True)
    
    for i, (train_index, test_index) in enumerate(kf.split(models)):
        
        logging.info('Fold: {}'.format(i))
        

        
        train_models = [models[index] for index in train_index]
        train_model_gt = [labels[index] for index in train_index]
        train_model_name = [model_names[index] for index in train_index]
        
        
        train_data = None 
        train_label = [] 
        
        for idx in range(len(train_models)):
            
            model, model_ground_truth, model_name  = train_models[idx], train_model_gt[idx], train_model_name[idx]
            train_label.append(model_ground_truth) 
            
            model_feat = get_weight_product(model,if_bias)    
            

            if train_data is None:
                train_data = model_feat
                continue
                
            train_data = np.vstack((train_data, model_feat))
            
            
            if augmentation:
                
                aug_model_list = model_augmentation(model,scaler,model_name,aug_num=2,delta_scale=0.3)
                for aug_model in aug_model_list:
                    aug_model_feat = get_weight_product(aug_model,if_bias)
            
        

                    if train_data is None:
                        train_data = aug_model_feat
                        continue
                        
                    train_data = np.vstack((train_data, aug_model_feat))
                    train_label.append(model_ground_truth)        
        
            
        
        logging.info('Training Data Shape: {}'.format(train_data.shape))
        clf_model.fit(train_data, train_label)      
        
        

        test_models = [models[index] for index in test_index]
        test_model_gt = [labels[index] for index in test_index]
        test_model_name = [model_names[index] for index in test_index]
        
        
        test_data = None 
        test_label = [] 
        
        for idx in range(len(test_models)):
            
            model, model_ground_truth, model_name  = test_models[idx], test_model_gt[idx], test_model_name[idx]
            test_label.append(model_ground_truth) 
            
            model_feat = get_weight_product(model,if_bias)    
            

            if test_data is None:
                test_data = model_feat
                continue
                
            test_data = np.vstack((test_data, model_feat))
                
        
        
        test_preds = clf_model.predict(test_data)
        loss = log_loss(test_label, test_preds)
        acc = accuracy_score(test_label, test_preds)
        roc_auc = roc_auc_score(test_label, test_preds)
        
        avg_acc += acc
        avg_roc += roc_auc
        avg_loss += loss
        
        logging.info('Fold: {}  Acc: {:.4f}  ROC_AUC: {:.4f} Log Loss: {:.4f}'.format(i,acc,roc_auc,loss))        
    
    
    logging.info('Average Acc: {:.4f}  Average ROC_AUC: {:.4f} Average Log Loss: {:.4f}'.format(avg_acc/cv,avg_roc/cv,avg_loss/cv))
            

        
    
    