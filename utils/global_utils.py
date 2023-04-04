import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
from math import radians, degrees, sin, cos, asin, acos, sqrt
from pathlib import Path
import os
import glob
import h5py

ROOT_PATH = Path(os.path.dirname(__file__))

def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile)

def open_json(fileName):
    try:
        with open(fileName,encoding='utf8') as json_data:
            return json.load(json_data)
    except Exception as s:
        print(s)

# TODO
all_loc_info0 = open_json(f'{ROOT_PATH}/../../mmg_news_dataset/info/geolocations_v1.json')
all_loc_info = {}

cls_to_q = {'city':{}, 'country':{}, 'continent':{}}

for g in all_loc_info0:
    all_loc_info[g] = {}

    for entry in all_loc_info0[g]:
        all_loc_info[g][entry['id']] = entry
        cls_to_q[g][entry['class']] = entry['id']
#



def classify_atk(outputs_in, targets_in, at ):
    
    labels = np.arange( len(outputs_in[0]))
   
    outputs = []
    for o in outputs_in:  outputs.append(o.detach().numpy())

    targets = []
    for t in targets_in:  targets.append(t.detach().numpy())
    
    acc =  top_k_accuracy_score(targets, outputs, k = at, labels=labels) 
    
    return np.round( acc * 100, 1)


def gcd(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    try:
        return 6371 * ( acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))
    except Exception as e:
        # print(lon1, lat1, lon2, lat2)
        return 0


def calc_gcd_accuracy_tk(t_coord, p_coord_list, g):
    n_total = len(t_coord)

    error_levels_ref = {'street': 1, 'city': 25, 'region': 200, 'country': 750, 'continent': 2500}
    error_levels = {'street': 0, 'city': 0, 'region': 0, 'country': 0, 'continent': 0}
    for t, p_all in zip(t_coord, p_coord_list):
        if t[0] == '' :
            continue
        street = False
        city = False
        region = False
        country = False
        continent = False
        for sample_Ps_tk in [p_all]:

            min_gcd_in_tk_Ps = 1000000
            min_p = ''
            for p in sample_Ps_tk:
                if p[0] == '':
                    continue
                gcd_tp = gcd(float(t[0]),float(t[1]),float(p[0]),float(p[1]))
                if gcd_tp < min_gcd_in_tk_Ps:
                    min_gcd_in_tk_Ps = gcd_tp

            if min_gcd_in_tk_Ps <= error_levels_ref['city']and not city:
                    error_levels['city'] +=1
                    city = True
            if min_gcd_in_tk_Ps <= error_levels_ref['region']and not region:
                    error_levels['region'] +=1
                    region = True
            if min_gcd_in_tk_Ps <= error_levels_ref['country']and not country:
                    error_levels['country'] +=1
                    country = True
            if min_gcd_in_tk_Ps <= error_levels_ref['continent']and not continent:
                    error_levels['continent'] +=1
                    continent = True
    g_val = error_levels_ref[g]
    error_levels_output = {f'GCD@{g_val}km': np.round( error_levels[g] / n_total*100, 1 ) }
    return error_levels_output

def return_info_cls(cls_in,g):
    info = {'lat':'', 'lng':''}

    Qloc = cls_to_q[g][str(cls_in)]
    info = {'lat':all_loc_info[g][Qloc]['lat'], 'lng':all_loc_info[g][Qloc]['lng']}          

    return info

def classify_gcd(preds, targets, at, g):


    topk = []

    for p in preds:
        topk.append(p[:at])

    p_coord_tk = []
    for tk_list in topk:
        p_coord = []
        for tk in tk_list:
            p_coord.append((return_info_cls(tk,g)['lng'],return_info_cls(tk,g)['lat']))
        p_coord_tk.append(p_coord)
    
    t_coord = []
    for tg in targets:
        t_coord.append((return_info_cls(tg.item(),g)['lng'], return_info_cls(tg.item(),g)['lat']))

    gcd_acc = calc_gcd_accuracy_tk(t_coord, p_coord_tk, g)

    return gcd_acc


def get_info_locs(data_path, path_data_locs, dir_test_texts):
    if not os.path.exists(path_data_locs):
        files = glob.glob(f'{data_path}/*')
        files.append(f'{dir_test_texts}/test.json')
        data = {}
        data_locs = {'city':{}, 'country':{} }

        for i_f,f in enumerate(files):
            data = open_json(f) 
            print(f'getting loc labels... {i_f}/{len(files)}')
            
            for id_ in data:
                im_label = data[id_]['image_label']
                data_locs['city'][im_label['city']['id']] = {'city':im_label['city']['id'], 'country':im_label['country']['id'], 'continent':im_label['continent']['id']} 
                data_locs['country'][im_label['country']['id']] = {'country':im_label['country']['id'], 'continent':im_label['continent']['id']} 

        save_file(path_data_locs, data_locs)
    else:
        return open_json( path_data_locs)

    return data_locs

def get_valid_test_ids(test_h5_dir):
    valid_ids_test = {}
    for g in ['city','country','continent']:
        valid_ids_test[g] = [id.decode("utf-8") for id in h5py.File(f'{test_h5_dir}/{g}_v1.h5', 'r')['ids']]
    
    return valid_ids_test



def get_pure_accuracy( baseline, partitions, valid_test_ids_per_g, path_test_data, path_acc_meta):
    
    def init_acc_file():
        accuracy_file={}
        for k in [1,2,5,10,20]:
            accuracy_file[f'acc@{k}'] = {}

            for g in ['city', 'country', 'continent']:
                accuracy_file[f'acc@{k}'][g] = {'acc':0, 'correct':0, 'all':0}
        return accuracy_file

    samples = {**open_json(path_test_data)['city'], **open_json(path_test_data)['country'],**open_json(path_test_data)['continent']}
    
    print(f'getting accuracy for {baseline} ... ')
    
    accuracy_file_meta = open_json(path_acc_meta)

    accuracy_file = init_acc_file()
    
    for k in [1,2,5,10,20]:

        for g in partitions:
            counter_correct = 0
            counter_all = 0

            for sample_id in accuracy_file_meta:

                if sample_id in valid_test_ids_per_g[g]:

                    sample =  samples[sample_id]
                    true_class = {'city':sample['image_label']['city']['id'], 'country':sample['image_label']['country']['id'], 'continent':sample['image_label']['continent']['id']}[g]
                    
                    if baseline in [ 't_freq']:
                        pred_classes_topk0 = accuracy_file_meta[sample_id][g][:k]
                    else:
                        pred_classes_topk0 = list(accuracy_file_meta[sample_id][g].keys())[:k]
                    counter_all += 1
                    pred_classes_topk = [str(p) for p in pred_classes_topk0]
                
                    if str(true_class) in pred_classes_topk:
                        counter_correct += 1

            accuracy_file[f'acc@{k}'][g] = {'acc':np.round( counter_correct/counter_all*100, 1), 'correct':counter_correct, 'all':counter_all}

    print(accuracy_file)

    return accuracy_file

def baselines_print_results(dir_results):
    accf = open_json(f'{dir_results}/acc_t_base_fine.json')
    all_res = ''
    for g in ['city', 'country', 'continent']:
        res = ''
        for k in ['acc@1', 'acc@2', 'acc@5', 'acc@10', ]:
            pr = accf[k][g]['acc']
            res += str( pr) + ' & '
        all_res +=res
        print(g, res)
    print(all_res)

def sort_by_value(dic_in):
    return dict(sorted(dic_in.items(), key=lambda item: item[1], reverse=True))

