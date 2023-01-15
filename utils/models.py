import re
from collections import OrderedDict
from os.path import join
import os 
import torch
from tqdm import tqdm


def create_layer_map(model_repr_dict):
    model_layer_map = {}
    for (model_class, models) in model_repr_dict.items():
        layers = models[0]
        layer_names = list(layers.keys())
        base_layer_names = list(
            dict.fromkeys(
                [
                    re.sub(
                        "\\.(weight|bias|running_(mean|var)|num_batches_tracked)",
                        "",
                        item,
                    )
                    for item in layer_names
                ]
            )
        )
        layer_map = OrderedDict(
            {
                base_layer_name: [
                    layer_name
                    for layer_name in layer_names
                    if re.match(f"{base_layer_name}.+", layer_name) is not None
                ]
                for base_layer_name in base_layer_names
            }
        )
        model_layer_map[model_class] = layer_map

    return model_layer_map


def load_model(model_filepath: str) -> (dict, str):
    """Load a model given a specific model_path.

    Args:
        model_filepath: str - Path to model.pt file

    Returns:
        model, dict, str - Torch model + dictionary representation of the model + model class name
    """
    model = torch.load(model_filepath).cpu()
    model_class = model.__class__.__name__
    model_repr = OrderedDict(
        {layer: tensor.numpy() for (layer, tensor) in model.state_dict().items()}
    )
    

    

    return model, model_repr, model_class


def load_ground_truth(model_dirpath: str):
    """Returns the ground truth for a given model.

    Args:
        model_dirpath: str -

    Returns:

    """

    with open(join(model_dirpath, "ground_truth.csv"), "r") as fp:
        model_ground_truth = fp.readlines()[0]

    return int(model_ground_truth)



def load_models_dirpath(models_dirpath,if_aug=False,aug_num=None,aug_models_dirpath=None):
    
    models_list = []
    models_ground_truth_list = []
    models_name_list = []
    
    for model_path in tqdm(models_dirpath):
        model = torch.load(join(model_path, "model.pt"))
        
        models_list.append(model)
        models_name_list.append(model_path)
        
        
        if os.path.exists(join(model_path,'poisoned-example-data')):
            models_ground_truth_list.append(1)
        
        else: 
            models_ground_truth_list.append(0)
        
    return models_list,models_ground_truth_list,models_name_list            


# def load_models_dirpath(models_dirpath,aug=False,aug_num=None):
#     model_repr_dict = {}
#     model_ground_truth_dict = {}
#     model_name_dict = {}
    

    

#     for model_path in tqdm(models_dirpath):
#         model, model_repr, model_class = load_model(
#             join(model_path, "model.pt")
#         )
#         model_ground_truth = load_ground_truth(model_path)

#         # Build the list of models
#         if model_class not in model_repr_dict.keys():
#             model_repr_dict[model_class] = []
#             model_ground_truth_dict[model_class] = []
#             model_name_dict[model_class] = []

#         model_repr_dict[model_class].append(model_repr)
#         model_ground_truth_dict[model_class].append(model_ground_truth)
#         model_name_dict[model_class].append('og')
        
        
        
#         if aug:
        
#             for aug_id in range(aug_num):
#                 model_id = model_path.split('/')[-1]
#                 aug_model_path = './augment_models/{}_aug_{}.pt'.format(model_id,aug_id)

#                 aug_model, aug_model_repr, aug_model_class = load_model(aug_model_path)
                
#                 aug_model_ground_truth = model_ground_truth
                
                


#                 model_repr_dict[aug_model_class].append(aug_model_repr)
#                 model_ground_truth_dict[aug_model_class].append(aug_model_ground_truth)
#                 model_name_dict[aug_model_class].append('aug')
        
             
                
            
            

#     return model_repr_dict, model_ground_truth_dict, model_name_dict

