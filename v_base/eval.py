import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import os
from pathlib import Path
from shutil import copy
import glob
import argparse
from utils.global_utils import *


def get_parser():
    parser = argparse.ArgumentParser(description='V-base')
    parser.add_argument('--dir_val_texts', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/text_splits/val')
    parser.add_argument('--dir_train_texts', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/text_splits/train')
    parser.add_argument('--dir_test_texts', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/text_splits/test')
    parser.add_argument('--dir_test_h5', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/h5_splits/test')

    parser.add_argument('--dir_results', default="/nfs/home/tahmasebzadehg/mmg_ecir/v_base/output")
    parser.add_argument('--path_test_images', default="/nfs/home/tahmasebzadehg/image_splits/test")
    parser.add_argument('--path_data_locs', default="/nfs/home/tahmasebzadehg/mmg_ecir/outputss/data_locs.json")
    parser.add_argument('--path_mapped_Ys_to_cells_fine', default="/nfs/home/tahmasebzadehg/mmg_ecir/outputss/mapped_Ys_to_cells_fine.json")
    
    parser.add_argument('--topk_hierarchical_predictions', default=0)
    parser.add_argument('--save_geolocation_estimation', default=False)
    parser.add_argument('--coords_history', default="/nfs/home/tahmasebzadehg/mmg_news_dataset/info/coords_history.json")
    parser.add_argument('--partition', default = 'fine' )

    return parser

parser = get_parser()
args = parser.parse_args()

root = Path(os.path.dirname(__file__))

def save_geolocation_estimation(topk, path_images):

    geo_results_path = f'{args.dir_results}/predictions.json'
    geo_obj = GeolocationEstimation(path_images)
    rows = geo_obj.geo_esitmation(topk, path_images)
    dic_output = {}
    
    for p in ['coarse', 'middle', 'fine', 'hierarchy']:  dic_output[p] = {}
    
    for i, row in enumerate(rows):
        id = row['img_id']
        lats = row['pred_lat']
        lngs = row['pred_lng']
        cls = row['pred_class'].tolist()
        probability = row['pred_value'].tolist()
        partition = row['p_key']
        emb = row['emb']
        ls_coords = []
        for lat, lng in zip(lats, lngs):  ls_coords.append((str(lat), str(lng)))

        dic_output[partition][id] = {'class':cls, 'probability':probability, 'coords':ls_coords}
            
    save_file(geo_results_path, dic_output)

    return geo_results_path

def sort_dic_by_val(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}

def get_accuracy_meta(image_results,  partition, info_locs):
    
    def find_country_continent_for_city(city):
        return info_locs['city'][city]['country'], info_locs['city'][city]['continent']
    
    def find_continent_for_country(country):
        return info_locs['country'][country]['continent']

    def update_accuracy_file_for_other_granularities(accuracy_file_sample, g, Y, cy, cls_to_prob):
        if g == 'city':
            country, continent = find_country_continent_for_city(Y)
            # a = cls_to_prob[int(cy)]

            if country not in accuracy_file_sample['country']:
                
                accuracy_file_sample['country'][country] = cls_to_prob[int(cy)]
            else:
                accuracy_file_sample['country'][country]   += cls_to_prob[int(cy)]
            
            if country not in accuracy_file_sample['continent']:
                accuracy_file_sample['continent'][continent] =cls_to_prob[int(cy)]
            else:
                accuracy_file_sample['continent'][continent] += cls_to_prob[int(cy)]

        elif g == 'country':
            continent = find_continent_for_country(Y)
            
            if continent not in accuracy_file_sample['continent']:
                accuracy_file_sample['continent'][continent] = cls_to_prob[int(cy)]
            else:
                accuracy_file_sample['continent'][continent] += cls_to_prob[int(cy)]

        return accuracy_file_sample

    print(f'saving accuracy meta method1 ... {partition}')

    [ys_fname] = ['fine' if partition == 'hierarchy' else partition]
    mapped_Ys = open_json(args.path_mapped_Ys_to_cells_fine)
    
    accuracy_file_meta = {}

    err = {'city':{}, 'country':{}, 'continent':{}, 'no_error':{}}

    for i_sample, sample_id in enumerate(image_results):
        
            accuracy_file_meta[sample_id] = {'city':{}, 'country':{}, 'continent':{} }
            print('id', i_sample + 1)
            sample_results = image_results[sample_id]
            probs = sample_results['probability']
            clses = sample_results['class']
            cls_to_prob = {}

            for i_p, p in enumerate(probs):
                cls_to_prob[clses[i_p]] = p
        
            for g in accuracy_file_meta[sample_id]:  # for all city classes get the probabilities
                for Y in mapped_Ys[g]:
                    y_cells = mapped_Ys[g][Y]
                    accuracy_file_meta[sample_id][g][Y] = 0
                    
                    for cy in y_cells:
                        if int(cy) in cls_to_prob:
                            accuracy_file_meta[sample_id][g][Y] += cls_to_prob[int(cy)]
                            
                            try:
                                if g in ['city', 'country']:
                                    accuracy_file_meta[sample_id] = update_accuracy_file_for_other_granularities(accuracy_file_meta[sample_id], g, Y, cy, cls_to_prob)
                                    err['no_error'][Y] = ""
                            except Exception as e:
                                err[g][Y] = e              

                accuracy_file_meta[sample_id][g] = sort_dic_by_val(accuracy_file_meta[sample_id][g])

    save_file(f'{args.dir_results}/accuracy.json', accuracy_file_meta)

def filter_image_results(test_data_path):
    image_results0 = open_json(  f'{args.dir_results}/predictions.json' )
    ids0 = open_json(test_data_path)
    ids = list(set( list(ids0['city'].keys()) + list(ids0['country'].keys()) + list(ids0['continent'].keys()) ))

    image_results = {}

    for id in ids:

        for glevel in ['city', 'country', 'continent']:

            for geo_id in image_results0:
                if id[:-2] in geo_id:

                    if glevel not in image_results:
                        image_results[glevel] = {}
                    image_results[glevel][id] = image_results0[id][glevel]
                   
    return image_results


## save geolocation estimation results 
if args.save_geolocation_estimation:

    save_geolocation_estimation(topk = args.topk_hierarchical_predictions,
                                path_images = args.path_test_images)
    # get info about locations
    info_locs = get_info_locs(data_path = args.dir_train_texts)  
    
    # filter based on the test set scenarios
    image_results =  filter_image_results( test_data_path = f'{args.dir_test_texts}/test.json')  

    get_accuracy_meta(image_results = image_results, 
                    partition = args.partition,
                    info_locs = info_locs)

valid_test_ids_per_g = get_valid_test_ids(test_h5_dir = args.dir_test_h5 )

# get accuracy 
accuracy_file = get_pure_accuracy(baseline='v_base',
                                  partitions = ['city','country','continent'], 
                                  valid_test_ids_per_g=valid_test_ids_per_g, 
                                  path_test_data = f'{args.dir_test_texts}/test.json',
                                  path_acc_meta = f'{args.dir_results}/v_base_meta_{args.partition}.json' )
save_file(f'{args.dir_results}/acc_v_base_{args.partition}.json', accuracy_file)

# get gcd
coords_history = open_json(args.coords_history) 
gcd_file, coords_history = classify_gcd(output_path = args.dir_results,
                                        accuracy_file_meta = open_json(f'{args.dir_results}/v_base_meta_{args.partition}.json'),
                                        dir_test_data = f'{args.dir_test_texts}/test.json',
                                        baseline = 'v_base',
                                        partitions = ['city','country','continent'],
                                        valid_test_ids_per_g = valid_test_ids_per_g,
                                        coords_history = coords_history)

save_file(f'{args.dir_results}/gcd_v_base.json', gcd_file)

# print results
baselines_print_results(args.dir_results)

