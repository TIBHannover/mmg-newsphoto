import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import os
from pathlib import Path
import glob
import argparse
from utils.global_utils import *
root = Path(os.path.dirname(__file__))

def get_parser():
    parser = argparse.ArgumentParser(description='V-base')
    parser.add_argument('--dir_val_texts', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/text_splits/val')
    parser.add_argument('--dir_train_texts', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/text_splits/train')
    parser.add_argument('--dir_test_texts', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/text_splits/test')
    parser.add_argument('--dir_test_h5', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/h5_splits/test')

    parser.add_argument('--dir_results', default="/nfs/home/tahmasebzadehg/mmg_ecir/t_base/output")
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


def save_geolocation_estimation(topk, path_images):

    geo_results_path = f'{args.dir_results}/predictions.json'
    geo_obj = GeolocationEstimation(path_images)
    rows = geo_obj.geo_esitmation(topk, path_images)
    dic_output = {}
    
    for p in ['coarse', 'middle', 'fine', 'hierarchy']:
        dic_output[p] = {}
    
    for i, row in enumerate(rows):
            id = row['img_id']
            lats = row['pred_lat']
            lngs = row['pred_lng']
            cls = row['pred_class'].tolist()
            probability = row['pred_value'].tolist()
            partition = row['p_key']
            emb = row['emb']
            ls_coords = []

            for lat, lng in zip(lats, lngs):
                ls_coords.append((str(lat), str(lng)))

            dic_output[partition][id] = {'class':cls, 'probability':probability, 'coords':ls_coords}
    save_file(geo_results_path, dic_output)

    return geo_results_path

def sort_dic_by_val(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}

def get_accuracy_meta( image_results, samples, partition, info_locs):
    
    def filter_preds_based_on_body(body_locs, sample_preds):
        output_filtered = {'city':{}, 'country':{}, 'continent':{}}
        
        # step1: if country and continent exist direclty in the body, then save their Ps
        for g in sample_preds:
            
            for loc in sample_preds[g]:
                if loc in body_locs:
                    output_filtered[g][loc] = sample_preds[g][loc]

                    # step2: add up the country and continent predictions
                    if g in ['city', 'country']:
                        loc_country = info_locs[g][loc]['country']
                        loc_continent = info_locs[g][loc]['continent']
                        # get the P of continent
                        [p_continent] = [sample_preds['continent'][loc_continent] if loc_continent in sample_preds['continent'] else sample_preds[g][loc] ]

                        if loc_continent in output_filtered['continent']:
                            output_filtered['continent'][loc_continent] += p_continent
                        else:
                            output_filtered['continent'][loc_continent] = p_continent
                    
                    if g == 'city':  # add Ps of country too

                        loc_country = info_locs[g][loc]['country']
                        # get the P of country
                        [p_country] =   [sample_preds['country'][loc_country] if loc_country in sample_preds['country'] else  sample_preds[g][loc] ]
                        
                        if loc_country in output_filtered['country']:
                            output_filtered['country'][loc_country] += p_country
                        else:
                            output_filtered['country'][loc_country] = p_country

        return output_filtered

    def find_country_continent_for_city(city):
        return info_locs['city'][city]['country'], info_locs['city'][city]['continent']
    
    def find_continent_for_country(country):
                return info_locs['country'][country]['continent']

    def update_accuracy_file_for_other_granularities(accuracy_file_sample, g, Y, cy, cls_to_prob):
        if g == 'city':
            country, continent = find_country_continent_for_city(Y)

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

    print('saving accuracy meta method2 ...')
    
    [ys_fname] = ['fine' if partition == 'hierarchy' else partition]
    mapped_Ys = open_json(args.path_mapped_Ys_to_cells_fine)
    
    accuracy_file_meta = { }
    err = {'city':{}, 'country':{}, 'continent':{}, 'no_error':{}}
    

    for i_sample, sample_id in enumerate(image_results[partition]):

        # sample = find_sample(samples, sample_id)
        sample = samples[sample_id]
        
        body_entities = [e for b in sample['body'] for e in b['named_entities']]
       
         
        body_locs = [e['wd_id'] for e in body_entities if e['type_wikidata']=='LOCATION']

        accuracy_file_meta[sample_id] = {'city':{}, 'country':{}, 'continent':{} }
        print('saving accuracy meta method2 ...',i_sample + 1, '/', len(image_results[partition]) )
        sample_results = image_results[partition][sample_id]
        probs = sample_results['probability']
        clses = sample_results['class']
        cls_to_prob = {}

        for i_p, p in enumerate(probs):
            cls_to_prob[clses[i_p]] = p
    
        for g in accuracy_file_meta[sample_id]:
            for Y in mapped_Ys[g]:
                y_cells = mapped_Ys[g][Y]
                accuracy_file_meta[sample_id][g][Y] = 0
                
                for cy in y_cells:
                    if int(cy) in list(cls_to_prob.keys()):
                            accuracy_file_meta[sample_id][g][Y] += cls_to_prob[int(cy)]
                        
                            if g in ['city', 'country']:
                                accuracy_file_meta[sample_id] = update_accuracy_file_for_other_granularities(accuracy_file_meta[sample_id], g, Y, cy, cls_to_prob)
                                err['no_error'][Y] = ""
                        
            accuracy_file_meta[sample_id][g] = sort_dic_by_val(accuracy_file_meta[sample_id][g])

        
        accuracy_file_meta[sample_id] = filter_preds_based_on_body(body_locs, accuracy_file_meta[sample_id]) # all gs

    save_file(f'{args.dir_results}/method2_meta_{partition}.json', accuracy_file_meta)

def filter_image_results(image_results0, test_data_path):
    
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

    info_locs = get_info_locs(data_path = args.dir_train_texts)  # from train or val data

    image_results0 = open_json(f'{args.dir_results}/visual_hierarchical_geolocation_0.json')
    image_results =  filter_image_results(image_results0, f'{args.dir_test_texts}/test.json')  # filter based on the test set scenarios

    get_accuracy_meta(image_results = image_results, 
                      partition = args.partition,
                      info_locs = info_locs)


## get accuracy
valid_test_ids_per_g = get_valid_test_ids(test_h5_dir = args.dir_test_h5 )
accuracy_file = get_pure_accuracy(
                                  baseline='t_base',
                                  partitions = ['city','country','continent'], 
                                  valid_test_ids_per_g = valid_test_ids_per_g, 
                                  path_test_data = f'{args.dir_test_texts}/test.json',
                                  path_acc_meta = f'{args.dir_results}/t_base_meta_{args.partition}.json' )
save_file(f'{args.dir_results}/acc_t_base_{args.partition}.json', accuracy_file)

## get gcd
coords_history = open_json(args.coords_history) 
gcd_file, coords_history = classify_gcd(output_path = args.dir_results,
                                        accuracy_file_meta = open_json(f'{args.dir_results}/t_base_meta_{args.partition}.json'),
                                        dir_test_data = f'{args.dir_test_texts}/test.json',
                                        baseline = 't_base',
                                        partitions = ['city','country','continent'],
                                        valid_test_ids_per_g = valid_test_ids_per_g,
                                        coords_history = coords_history)

save_file(f'{args.dir_results}/gcd_t_base_{args.partition}.json', gcd_file)

# print results

baselines_print_results(args.dir_results)
