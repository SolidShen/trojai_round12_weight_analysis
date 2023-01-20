import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename
import numpy as np
from tqdm import tqdm
import torch 

# import xgboost as xgb
from scipy.stats import uniform, randint


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score,log_loss
from sklearn.model_selection import cross_validate,RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import importlib
import pickle 
import json 


from utils.abstract import AbstractDetector
from utils.jacobian import get_jacobian
from utils.feature_extraction import get_layer_features, align_layer_features,get_model_features,get_weight_product
from utils.models import load_models_dirpath
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s



class WeightAnalysisDetector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
            
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        

        
        

        
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.scale_parameters_filepath = scale_parameters_filepath
        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")
        

        self.weight_feat_kwargs = {
            
            'if_bias': metaparameters[
                'train_weight_feat_param_if_bias'
            ]
            
        }
        
        self.GB_kwargs = {
            'n_estimators': metaparameters[
                'train_gradient_boosting_param_n_estimators'
                ],
            'learning_rate': metaparameters[
                'train_gradient_boosting_param_learning_rate'
                ],
            'max_depth': metaparameters[
                'train_gradient_boosting_param_max_depth'
                ],
            'subsample': metaparameters[
                'train_gradient_boosting_param_subsample'
                ],
            'max_features': metaparameters[
                'train_gradient_boosting_param_max_features'
                ],
            'loss': metaparameters[
                'train_gradient_boosting_param_loss'
            ]
            
        }
        


    def write_metaparameters(self):
        metaparameters = {
            "train_weight_feat_param_if_bias": self.weight_feat_kwargs["if_bias"],
            "train_gradient_boosting_param_n_estimators": self.GB_kwargs["n_estimators"],
            "train_gradient_boosting_param_learning_rate": self.GB_kwargs["learning_rate"],
            "train_gradient_boosting_param_max_depth": self.GB_kwargs["max_depth"],
            "train_gradient_boosting_param_subsample": self.GB_kwargs["subsample"],
            "train_gradient_boosting_param_max_features": self.GB_kwargs["max_features"],
            "train_gradient_boosting_param_loss": self.GB_kwargs["loss"],
        }

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)
    
    
    
    def write_metrics(self, metrics):
        
        metrics = {
            'train_accuracy': metrics['train_accuracy'],
            'train_log_loss': metrics['train_log_loss'],
            'train_roc_auc': metrics['train_roc_auc'],
            'cv_accuracy': metrics['cv_accuracy'],
            'cv_log_loss': metrics['cv_log_loss'],
            'cv_roc_auc': metrics['cv_roc_auc'],
        }

        with open(join(self.learned_parameters_dirpath, basename('metrics.json')), "w") as fp:
            json.dump(metrics, fp)
            
        

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            # self.weight_table_params["random_seed"] = random_seed
            self.manual_configure(models_dirpath,random_seed)
            
    
    
    def manual_configure(self, models_dirpath,random_seed=42):
        """Manual configuration function.

        Args:
            models_dirpath: str - Path to the directory containing the models to be used for training.
        """
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)
            
        
        
        
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))
        
        models_list, models_ground_truth_list, models_name_list = load_models_dirpath(model_path_list)
        

        
        
        logging.info("Weight statistics feature extraction applied. Creating feature file...")
        
        scaler = StandardScaler()
        scale_params = np.load(self.scale_parameters_filepath)
        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]
        
        
        
        train_data = None 
        train_label = [] 
        
        
        # feat_list = ['min','max','mean','std','skewness','kurtosis','svd','l1_norm','l2_norm','l_inf_norm']
        # layer_selection = 'all'
        
        
        
        for idx in tqdm(range(len(models_list))):
            
            
            
            
            model, model_ground_truth, model_name  = models_list[idx], models_ground_truth_list[idx], models_name_list[idx]
            
            train_label.append(model_ground_truth) 
            
            # clean_data_dirpath = os.path.join(model_name,'clean-example-data')
            
            # model_jacobian = get_jacobian(clean_data_dirpath,model,scaler,**self.jaco_kwargs)
            
            
            # model_feat = align_layer_features(feats=get_layer_features(model,feat_list=feat_list,layer_selection=layer_selection))
            
            model_feat = get_weight_product(model,self.weight_feat_kwargs['if_bias'])
            
            # model_feat = get_model_features(model,feat_list=feat_list)

            
            
            if train_data is None:
                train_data = model_feat
                continue
                
            train_data = np.vstack((train_data, model_feat))
            
        
        

        
        
        
        
        base_model = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.001, max_depth=3, subsample=0.7,
                                      max_features='sqrt', random_state=519, loss='log_loss')
        
        
        
        logging.info("Training GradientBoosting model...")
        
        # base_model = GradientBoostingClassifier(**self.GB_kwargs,random_state=random_seed)
        

        # 'n_estimators': metaparameters[
        #     'train_gradient_boosting_param_n_estimators'
        #     ],
        # 'learning_rate': metaparameters[
        #     'train_gradient_boosting_param_learning_rate'
        #     ],
        # 'max_depth': metaparameters[
        #     'train_gradient_boosting_param_max_depth'
        #     ],
        # 'subsample': metaparameters[
        #     'train_gradient_boosting_param_subsample'
        #     ],
        # 'max_features': metaparameters[
        #     'train_gradient_boosting_param_max_features'
        #     ],
        
        # params = {
        #     "learning_rate": uniform(0.0005, 0.00015), # default 0.1 
        #     "max_depth": randint(2, 6), # default 3
        #     "n_estimators": randint(3000, 5000), # default 100
        #     "subsample": uniform(0.8, 0.4)
        # }
                
        
        # base_model = xgb.XGBClassifier(objective="binary:logistic",eval_metric = "logloss", booster='gbtree',
        #                                colsample_bytree=0.75,
        #                                gamma=0.37,
        #                                learning_rate=0.05,
        #                                max_depth=5,
        #                                n_estimators=470,
        #                                subsample=0.89)
        
        
        
        # params = {
        #     "colsample_bytree": uniform(0.7, 0.3),
        #     "gamma": uniform(0, 0.5),
        #     "learning_rate": uniform(0.03, 0.3), # default 0.1 
        #     "max_depth": randint(2, 6), # default 3
        #     "n_estimators": randint(100, 1000), # default 100
        #     "subsample": uniform(0.8, 0.4)
        # }

        # search = RandomizedSearchCV(base_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1, n_jobs=1, return_train_score=True)

        # search.fit(train_data, train_label)

        # def report_best_scores(results, n_top=3):
        #     for i in range(1, n_top + 1):
        #         candidates = np.flatnonzero(results['rank_test_score'] == i)
        #         for candidate in candidates:
        #             print("Model with rank: {0}".format(i))
        #             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
        #                 results['mean_test_score'][candidate],
        #                 results['std_test_score'][candidate]))
        #             print("Parameters: {0}".format(results['params'][candidate]))
        #             print("")
            
            
        # report_best_scores(search.cv_results_, 1)

        # exit()
        
        
        clf_model = CalibratedClassifierCV(estimator=base_model, cv=5)
        
        
        
        scoring = {'accuracy', 'roc_auc', 'neg_log_loss'}
        
        cv_score = cross_validate(clf_model, train_data, train_label, cv=5,scoring=scoring)
        # mean_score = np.mean(cv_score)
        
        cv_acc = cv_score['test_accuracy']
        cv_roc_auc = cv_score['test_roc_auc']
        cv_log_loss = cv_score['test_neg_log_loss']
        
        mean_acc = np.mean(cv_acc)
        mean_roc_auc = np.mean(cv_roc_auc)
        mean_log_loss = np.mean(cv_log_loss)
                
        # logging.info('cv_score: {} mean_score: {:.4f}'.format(acc,mean_score))
        
        logging.info('Cross Validation Scores:')
        
        logging.info('Accuracy: {} [{:.4f}]'.format(cv_acc,mean_acc))
        logging.info('ROC_AUC: {} [{:.4f}]'.format(cv_roc_auc,mean_roc_auc))
        logging.info('Log_Loss: {} [{:.4f}]'.format(cv_log_loss,mean_log_loss))
        
                
        clf_model.fit(train_data, train_label)
        score_rbf = clf_model.score(train_data, train_label)
        logging.info("The train score of rbf is : %f" % score_rbf)
        
        train_preds = clf_model.predict(train_data)
        train_raw_log = log_loss(train_label, train_preds)
        logging.info("Train log loss: {}".format(train_raw_log))

        train_roc = roc_auc_score(train_label, train_preds)

        logging.info('Train ROC AUC: %.3f' % train_roc)
        
        
        
        logging.info("Saving GradientBoosting model...")
        
        
        with open(self.model_filepath, "wb") as fp:
            pickle.dump(clf_model, fp)
        
        metrics = {}
        metrics['cv_accuracy'] = mean_acc
        metrics['cv_log_loss'] = mean_log_loss
        metrics['cv_roc_auc'] = mean_roc_auc
        metrics['train_accuracy'] = score_rbf
        metrics['train_log_loss'] = train_raw_log
        metrics['train_roc_auc'] = train_roc
        
            
        self.write_metaparameters()
        self.write_metrics(metrics)
        logging.info("Configuration done!")
        
        
            
    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict wether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        scaler = StandardScaler()
        scale_params = np.load(self.scale_parameters_filepath)
        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]
        
        
        with open(self.model_filepath,'rb') as fp:
            clf_model = pickle.load(fp)
        
        
        
        model = torch.load(model_filepath).cuda().eval()
        
        # model_jacobian = get_jacobian(examples_dirpath,model,scaler,**self.jaco_kwargs).reshape(1, -1)
        
        
        model_feat = get_weight_product(model,self.weight_feat_kwargs['if_bias']).reshape(1, -1)
        


        
        
        
        probability = (clf_model.predict_proba(model_feat)[0][1])
        
        # probability = str(clf_model.predict_proba(model_jacobian)[0][1])
        # binarize output 
        # bin_probability = str(clf_model.predict(model_jacobian)[0])
        
        
        
        # if probability > 0.42:
        #     cali_probability = 0.92
        
        # else:
        #     cali_probability = 0
        
        
        cali_probability = probability
        cali_probability = str(cali_probability)
        
        
        
        with open(result_filepath, "w") as fp:
            fp.write(cali_probability)
        
        logging.info("Trojan probability: %s", cali_probability)
        # logging.info("Trojan probability: %s", bin_probability)
        
        
        
    
        
if __name__ == '__main__':
    
    metaparameters_filepath = './metaparameters.json'
    learned_parameters_dirpath = './learned_parameters'
    scale_parameters_filepath = './scale_params.npy'
    models_dirpath = '/data3/share/trojai/trojai-round12-cyber-v1-dataset/models/'
    
    model_filepath = '/data3/share/trojai/trojai-round12-cyber-v1-dataset/models/id-00000001/model.pt'
    result_filepath = './results/result.txt'
    scratch_dirpath = './scratch/'
    examples_dirpath = '/data3/share/trojai/trojai-round12-cyber-v1-dataset/models/id-00000001/clean-example-data/'
    round_training_dataset_dirpath = '/data3/share/trojai/trojai-round12-cyber-v1-dataset/models/'
    
    
    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        )
    
    detector = WeightAnalysisDetector(metaparameters_filepath, learned_parameters_dirpath, scale_parameters_filepath)
    detector.manual_configure(models_dirpath)
    # detector.infer(model_filepath,
    #     result_filepath,
    #     scratch_dirpath,
    #     examples_dirpath,
    #     round_training_dataset_dirpath)
    


