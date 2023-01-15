import torch 
import os 
import numpy as np 
import logging
import json 

def get_editable_feat_ids():

    FileDefined = -519

    _pdfrate_feature_descriptions = {
                    'author_dot'              :{'type':int, 'range':(0, 24), 'edit':'y'},
                    'author_lc'               :{'type':int, 'range':(0, 158), 'edit':'y'},
                    'author_len'              :{'type':int, 'range':(0, 288), 'edit':'m'},
                    'author_mismatch'         :{'type':int, 'range':(FileDefined, 45), 'edit':'m'},
                    'author_num'              :{'type':int, 'range':(0, 63), 'edit':'y'},
                    'author_oth'              :{'type':int, 'range':(0, 79), 'edit':'y'},
                    'author_uc'               :{'type':int, 'range':(0, 108), 'edit':'y'},
                    'box_nonother_types'      :{'type':int, 'range':(FileDefined, 2684), 'edit':'y'},
                    'box_other_only'          :{'type':bool, 'range': (False, True), 'edit':'m'},
                    'company_mismatch'        :{'type':int, 'range':(FileDefined, 2), 'edit':'m'},
                    'count_acroform'          :{'type':int, 'range':(FileDefined, 7), 'edit':'y'},
                    'count_acroform_obs'      :{'type':int, 'range':(FileDefined, 1), 'edit':'y'},
                    'count_action'            :{'type':int, 'range':(FileDefined, 815), 'edit':'y'},
                    'count_action_obs'        :{'type':int, 'range':(FileDefined, 1), 'edit':'y'},
                    'count_box_a4'            :{'type':int, 'range':(FileDefined, 34), 'edit':'y'},
                    'count_box_legal'         :{'type':int, 'range':(FileDefined, 268), 'edit':'y'},
                    'count_box_letter'        :{'type':int, 'range':(FileDefined, 2684), 'edit':'y'},
                    'count_box_other'         :{'type':int, 'range':(FileDefined, 12916), 'edit':'y'},
                    'count_box_overlap'       :{'type':int, 'range':(FileDefined, 10), 'edit':'y'},
                    'count_endobj'            :{'type':int, 'range':(FileDefined, 19632), 'edit':'y'},
                    'count_endstream'         :{'type':int, 'range':(FileDefined, 6668), 'edit':'y'},
                    'count_eof'               :{'type':int, 'range':(FileDefined, 24), 'edit':'y'},
                    'count_font'              :{'type':int, 'range':(FileDefined, 7333), 'edit':'y'},
                    'count_font_obs'          :{'type':int, 'range':(FileDefined, 5), 'edit':'y'},
                    'count_image_large'       :{'type':int, 'range':(FileDefined, 35), 'edit':'y'},
                    'count_image_med'         :{'type':int, 'range':(FileDefined, 278), 'edit':'y'},
                    'count_image_small'       :{'type':int, 'range':(FileDefined, 311), 'edit':'y'},
                    'count_image_total'       :{'type':int, 'range':(FileDefined, 4542), 'edit':'n'},
                    'count_image_xlarge'      :{'type':int, 'range':(FileDefined, 3), 'edit':'y'},
                    'count_image_xsmall'      :{'type':int, 'range':(FileDefined, 4525), 'edit':'y'},
                    'count_javascript'        :{'type':int, 'range':(FileDefined, 404), 'edit':'y'},
                    'count_javascript_obs'    :{'type':int, 'range':(FileDefined, 2), 'edit':'y'},
                    'count_js'                :{'type':int, 'range':(FileDefined, 404), 'edit':'y'},
                    'count_js_obs'            :{'type':int, 'range':(FileDefined, 2), 'edit':'y'},
                    'count_obj'               :{'type':int, 'range':(FileDefined, 19632), 'edit':'y'},
                    'count_objstm'            :{'type':int, 'range':(FileDefined, 1036), 'edit':'y'},
                    'count_objstm_obs'        :{'type':int, 'range':(FileDefined, 1), 'edit':'y'},
                    'count_page'              :{'type':int, 'range':(FileDefined, 1341), 'edit':'y'},
                    'count_page_obs'          :{'type':int, 'range':(FileDefined, 6), 'edit':'y'},
                    'count_startxref'         :{'type':int, 'range':(FileDefined, 24), 'edit':'y'},
                    'count_stream'            :{'type':int, 'range':(FileDefined, 6668), 'edit':'y'},
                    'count_stream_diff'       :{'type':int, 'range':(-2, 69), 'edit':'m'},
                    'count_trailer'           :{'type':int, 'range':(FileDefined, 46), 'edit':'y'},
                    'count_xref'              :{'type':int, 'range':(FileDefined, 46), 'edit':'y'},
                    'createdate_dot'          :{'type':int, 'range':(0, 1), 'edit':'n'},
                    'createdate_mismatch'     :{'type':int, 'range':(FileDefined, 92), 'edit':'m'},
                    'createdate_ts'           :{'type':int, 'range':(0, 2058951292), 'edit':'y'},
                    'createdate_tz'           :{'type':int, 'range':(-36000, 46800), 'edit':'y'},
                    'createdate_version_ratio':{'type':float, 'range':(-1.0, 7326.0), 'edit':'m'},
                    'creator_dot'             :{'type':int, 'range':(0, 5), 'edit':'y'},
                    'creator_lc'              :{'type':int, 'range':(0, 46), 'edit':'y'},
                    'creator_len'             :{'type':int, 'range':(0, 166), 'edit':'m'},
                    'creator_mismatch'        :{'type':int, 'range':(FileDefined, 45), 'edit':'m'},
                    'creator_num'             :{'type':int, 'range':(0, 105), 'edit':'y'},
                    'creator_oth'             :{'type':int, 'range':(0, 64), 'edit':'y'},
                    'creator_uc'              :{'type':int, 'range':(0, 31), 'edit':'y'},
                    'delta_ts'                :{'type':int, 'range':(-1279940312, 1294753248), 'edit':'n'},
                    'delta_tz'                :{'type':int, 'range':(-57600, 46800), 'edit':'n'},
                    'image_mismatch'          :{'type':int, 'range':(FileDefined, 207), 'edit':'m'},
                    'image_totalpx'           :{'type':int, 'range':(FileDefined, 119549960), 'edit':'m'},
                    'keywords_dot'            :{'type':int, 'range':(0, 5), 'edit':'y'},
                    'keywords_lc'             :{'type':int, 'range':(0, 291), 'edit':'y'},
                    'keywords_len'            :{'type':int, 'range':(0, 446), 'edit':'m'},
                    'keywords_mismatch'       :{'type':int, 'range':(FileDefined, 19), 'edit':'m'},
                    'keywords_num'            :{'type':int, 'range':(0, 102), 'edit':'y'},
                    'keywords_oth'            :{'type':int, 'range':(0, 185), 'edit':'y'},
                    'keywords_uc'             :{'type':int, 'range':(0, 164), 'edit':'y'},
                    'len_obj_avg'             :{'type':float, 'range':(0.1, 382922.0), 'edit':'m'},
                    'len_obj_max'             :{'type':int, 'range':(FileDefined, 4161352), 'edit':'m'},
                    'len_obj_min'             :{'type':int, 'range':(6, FileDefined), 'edit':'m'},
                    'len_stream_avg'          :{'type':float, 'range':(0.0, 1531433.0), 'edit':'m'},
                    'len_stream_max'          :{'type':int, 'range':(FileDefined, 4610074), 'edit':'m'},
                    'len_stream_min'          :{'type':int, 'range':(0, FileDefined), 'edit':'m'},
                    'moddate_dot'             :{'type':int, 'range':(0, 1), 'edit':'n'},
                    'moddate_mismatch'        :{'type':int, 'range':(FileDefined, 23), 'edit':'m'},
                    'moddate_ts'              :{'type':int, 'range':(0, 2058951292), 'edit':'y'},
                    'moddate_tz'              :{'type':int, 'range':(-36000, 46800), 'edit':'y'},
                    'moddate_version_ratio'   :{'type':float, 'range':(0.1, 7390.0), 'edit':'m'},
                    'pdfid0_dot'              :{'type':int, 'range':(0, 1), 'edit':'m'},
                    'pdfid0_lc'               :{'type':int, 'range':(0, 21), 'edit':'m'},
                    'pdfid0_len'              :{'type':int, 'range':(0, 46), 'edit':'m'},
                    'pdfid0_mismatch'         :{'type':int, 'range':(FileDefined, 3), 'edit':'m'},
                    'pdfid0_num'              :{'type':int, 'range':(0, 44), 'edit':'m'},
                    'pdfid0_oth'              :{'type':int, 'range':(0, 1), 'edit':'m'},
                    'pdfid0_uc'               :{'type':int, 'range':(0, 21), 'edit':'m'},
                    'pdfid1_dot'              :{'type':int, 'range':(0, 1), 'edit':'m'},
                    'pdfid1_lc'               :{'type':int, 'range':(0, 21), 'edit':'m'},
                    'pdfid1_len'              :{'type':int, 'range':(0, 32), 'edit':'m'},
                    'pdfid1_mismatch'         :{'type':int, 'range':(FileDefined, 23), 'edit':'m'},
                    'pdfid1_num'              :{'type':int, 'range':(0, 32), 'edit':'m'},
                    'pdfid1_oth'              :{'type':int, 'range':(0, 1), 'edit':'m'},
                    'pdfid1_uc'               :{'type':int, 'range':(0, 22), 'edit':'m'},
                    'pdfid_mismatch'          :{'type':bool, 'range': (False, True), 'edit':'m'},
                    'pos_acroform_avg'        :{'type':float, 'range':(FileDefined, 1.0), 'edit':'m'},
                    'pos_acroform_max'        :{'type':float, 'range':(FileDefined, 1.0), 'edit':'m'},
                    'pos_acroform_min'        :{'type':float, 'range':(0.0, 1,0), 'edit':'m'},
                    'pos_box_avg'             :{'type':float, 'range':(FileDefined, 1.0), 'edit':'m'},
                    'pos_box_max'             :{'type':float, 'range':(FileDefined, 1.0), 'edit':'m'},
                    'pos_box_min'             :{'type':float, 'range':(0.0, 1.0), 'edit':'n'},
                    'pos_eof_avg'             :{'type':float, 'range':(FileDefined, 1.0), 'edit':'m'},
                    'pos_eof_max'             :{'type':float, 'range':(FileDefined, 1.0), 'edit':'n'},
                    'pos_eof_min'             :{'type':float, 'range':(0.0, FileDefined), 'edit':'m'},
                    'pos_image_avg'           :{'type':float, 'range':(FileDefined, 1.0), 'edit':'m'},
                    'pos_image_max'           :{'type':float, 'range':(FileDefined, 1.0), 'edit':'m'},
                    'pos_image_min'           :{'type':float, 'range':(0.0, 1.0), 'edit':'m'},
                    'pos_page_avg'            :{'type':float, 'range':(FileDefined, 1.0), 'edit':'m'},
                    'pos_page_max'            :{'type':float, 'range':(FileDefined, 1.0), 'edit':'m'},
                    'pos_page_min'            :{'type':float, 'range':(0.0, 1.0), 'edit':'m'},
                    'producer_dot'            :{'type':int, 'range':(0, 1962), 'edit':'y'},
                    'producer_lc'             :{'type':int, 'range':(0, 7343), 'edit':'y'},
                    'producer_len'            :{'type':int, 'range':(0, 32566), 'edit':'m'},
                    'producer_mismatch'       :{'type':int, 'range':(FileDefined, 9), 'edit':'m'},
                    'producer_num'            :{'type':int, 'range':(0, 25302), 'edit':'y'},
                    'producer_oth'            :{'type':int, 'range':(0, 9233), 'edit':'y'},
                    'producer_uc'             :{'type':int, 'range':(0, 13), 'edit':'y'},
                    'ratio_imagepx_size'      :{'type':float, 'range':(0.0, 559.0), 'edit':'m'},
                    'ratio_size_obj'          :{'type':float, 'range':(24.0, 382971.0), 'edit':'m'},
                    'ratio_size_page'         :{'type':float, 'range':(266.0, 1.15e+13), 'edit':'m'},
                    'ratio_size_stream'       :{'type':float, 'range':(316.0, 39680000000.0), 'edit':'m'},
                    'size'                    :{'type':int, 'range':(FileDefined, 10000000), 'edit':'y'},
                    'subject_dot'             :{'type':int, 'range':(0, 5), 'edit':'y'},
                    'subject_lc'              :{'type':int, 'range':(0, 256), 'edit':'y'},
                    'subject_len'             :{'type':int, 'range':(0, 413), 'edit':'m'},
                    'subject_mismatch'        :{'type':int, 'range':(FileDefined, 15), 'edit':'m'},
                    'subject_num'             :{'type':int, 'range':(0, 113), 'edit':'y'},
                    'subject_oth'             :{'type':int, 'range':(0, 82), 'edit':'y'},
                    'subject_uc'              :{'type':int, 'range':(0, 94), 'edit':'y'},
                    'title_dot'               :{'type':int, 'range':(0, 11), 'edit':'y'},
                    'title_lc'                :{'type':int, 'range':(0, 183), 'edit':'y'},
                    'title_len'               :{'type':int, 'range':(0, 698), 'edit':'m'},
                    'title_mismatch'          :{'type':int, 'range':(FileDefined, 517), 'edit':'m'},
                    'title_num'               :{'type':int, 'range':(0, 420), 'edit':'y'},
                    'title_oth'               :{'type':int, 'range':(0, 160), 'edit':'y'},
                    'title_uc'                :{'type':int, 'range':(0, 74), 'edit':'y'},
                    'version'                 :{'type':int, 'range':(1, 8), 'edit':'y'}}




    editable_feats_ids = []



    total_num = 0
    for feat_id, feat_name in enumerate(_pdfrate_feature_descriptions.keys()):
        feat = _pdfrate_feature_descriptions[feat_name]

        # print(feat_id,feat_name)
        
        if feat['type'] == int and FileDefined not in feat['range'] and feat['edit'] == 'y' and feat['range'][0] >= 0 and feat['range'][1] < 1e+6:

            editable_feats_ids.append(feat_id)
    
    return editable_feats_ids


def get_jacobian(clean_data_dirpath,model,scaler,if_noise=False,noise_scale=None,n_samples=None, sample_classes="1", aggr_method='max', if_reduce_dim=False):
    
    """get model's jacobian matrix based on configs
    """
    
    model = model.cuda()
    model = model.eval()
    
    
    # parse sample classes 
    
    if sample_classes == "1":
        sample_classes = [1]
    
    elif sample_classes == "0":
        sample_classes = [0]
    
    elif sample_classes == 'both':
        sample_classes = [0,1]
    
    
    noise = None 
    
    if if_noise:
        # logging.info("compute jacobian w.r.t random noise")
        raise NotImplementedError("random noise is not implemented yet")
    
    else:
        
        # logging.info('compute jacobian w.r.t clean samples from classes: {}'.format(sample_classes))
        
        
        
        
        for clean_data in os.listdir(clean_data_dirpath):
            clean_data_filepath = os.path.join(clean_data_dirpath, clean_data)
            if clean_data_filepath.endswith('.npy'):
                clean_label_filepath = clean_data_filepath + '.json'
                clean_label = json.load(open(clean_label_filepath))
                
                if clean_label in sample_classes:
                    raw_clean_data = np.load(clean_data_filepath).reshape(1, -1)
                    clean_data = torch.from_numpy(scaler.transform(raw_clean_data.astype(float))).float()
                    
                    if noise is None:
                        noise = clean_data
                        continue 
                    
                    noise = torch.cat((noise,clean_data),dim=0)
        
        
        noise = noise.cuda()
        output = model(noise)
        
        
        
        
        
        jaco = torch.autograd.functional.jacobian(model, noise)
        jaco = torch.diagonal(jaco,dim1=0,dim2=2)
        
        
        if aggr_method == 'max':
            jaco = jaco.max(dim=2)[0]
            output = output.max(dim=0)[0].unsqueeze(1)
            jaco = torch.cat((jaco,output),dim=1)
            

            
            
                    
        elif aggr_method == 'min':
            jaco = jaco.min(dim=2)[0]
            output = output.min(dim=0)[0].unsqueeze(1)
            jaco = torch.cat((jaco,output),dim=1)
        
        elif aggr_method == 'mean':
            jaco = jaco.mean(dim=2)
            output = output.mean(dim=0).unsqueeze(1)
            jaco = torch.cat((jaco,output),dim=1)
        
        else:
            raise NotImplementedError('{} aggregation method not implemented yet, current support methods: [max | min | mean]'.format(aggr_method))
        
        
        
        
        
        if if_reduce_dim:
            editable_feats_ids = get_editable_feat_ids()
            jaco = jaco[:,editable_feats_ids]
        
        jaco = jaco.flatten().detach().cpu().numpy()
        
        
        return jaco 
        
        
        
        
        