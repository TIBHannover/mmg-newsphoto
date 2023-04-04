import os.path
import sys
import glob
import argparse
from utils.global_utils import *
import requests
import time 


def get_parser():
    parser = argparse.ArgumentParser(description='V-base')
    parser.add_argument('--dir_val_texts', default='mmg_news_dataset/text_splits/val')
    parser.add_argument('--dir_train_texts', default='mmg_news_dataset/text_splits/train')
    parser.add_argument('--dir_test_texts', default='mmg_news_dataset/text_splits/test')
    parser.add_argument('--dir_test_h5', default='mmg_news_dataset/h5_splits/test')

    parser.add_argument('--dir_results', default="vt_cm/output")
    parser.add_argument('--path_test_images', default="/nfs/home/tahmasebzadehg/image_splits/test")
    parser.add_argument('--path_data_locs', default="data_locs.json")
    parser.add_argument('--coords_history', default="mmg_news_dataset/info/coords_history.json")

    return parser

parser = get_parser()
args = parser.parse_args()

def fix_ids():
    results = {}

    for id in results0:
            id_new = id.split('_')[0] + id.split('_')[1]
            results[id_new] = results0[id]    
    save_file(f'{args.dir_results}/method4_meta_scenarios_12_per_body.json', results)

def get_country_continent(wikidata_id):
    query =f""" SELECT ?country ?countryLabel ?continent ?continentLabel WHERE{{ wd:{wikidata_id} wdt:P17 ?country. ?country wdt:P30 ?continent. SERVICE wikibase:label {{ bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'.}} }}"""
    url = 'https://query.wikidata.org/sparql'
    while True:
        try:
            r = requests.get(url, params = {'format': 'json', 'query': query})
            res0 = r.json()
            countrys = []
            continents = []
            retreived_results = res0["results"]["bindings"]

            if len(retreived_results) > 0:
                for rr in retreived_results:
                    country_id = rr['country']['value'].split("/")[-1]
                    country_label = rr['countryLabel']['value']
                    cont_id = rr['continent']['value'].split("/")[-1]
                    cont_label = rr['continentLabel']['value']
                    
                    continents.append({"id":cont_id, "label":cont_label})

                    if country_id == wikidata_id:
                        countrys.append( {"id":"self", "label":"self"})
                    else:
                        countrys.append( {"id":country_id, "label":country_label} )
            break
        except Exception as e:
            exc_type, _, exc_tb = sys.exc_info()
            print(wikidata_id , r.status_code, exc_tb.tb_lineno)
            time.sleep(3)
    return countrys, continents

def get_info_locs():
    if not os.path.exists(args.path_data_locs):  # from method1.py
        files = glob.glob(f'{args.dir_train_data}/*')
        files.append(args.dir_test_data)
        data = {}
        data_locs = {'city':{}, 'country':{} }

        for i_f,f in enumerate(files):
            data = open_json(f) 
            print(f'getting loc labels... {i_f}/{len(files)}')
            
            for id_ in data:
                im_label = data[id_]['image_label']
                data_locs['city'][im_label['city']['id']] = {'city':im_label['city']['id'], 'country':im_label['country']['id'], 'continent':im_label['continent']['id']} 
                data_locs['country'][im_label['country']['id']] = {'country':im_label['country']['id'], 'continent':im_label['continent']['id']} 

        save_file(args.path_data_locs, data_locs)
    else:
        return open_json(args.path_data_locs)

    return data_locs

def get_meta_vt_cm(is_per_sentence = False, results = {}, locs_info = []):
    
    def get_from_locs_info(loc):
            country = ''
            continent = ''
            
            try:
                country = locs_info['city'][loc]['country']
            except:

                try:
                    country = locs_info['country'][loc]['country']
                except:
                    country = ''
            try:
                continent = locs_info['city'][loc]['continent']
            except:

                try:
                    continent = locs_info['country'][loc]['continent']
                except:
                    continent = ''

            return country, continent

    def fix_granularity_preds(pred_classes, history_):
        granularity_preds = { 'country': {}, 'continent':{}}

        for loc in pred_classes:
            
            # step 1
            country, continent = get_from_locs_info(loc)
            # step 2
            if country == '' or continent == '':
                try:
                    country = history_[loc]['country']['id']
                    continent = history_[loc]['continent']['id']
                except:
                    country = ''
                    continent = ''
            # step 3
            if country == '' or continent == '':
               history_[loc] = {'country':{}, 'continent':{}}
                
               countrys, continents = get_country_continent(loc)
               
               for country in countrys:
                    granularity_preds['country'][country['id']] = pred_classes[loc]
                    history_[loc]['country'][country['id']] = country['label']
               for continent in continents:
                    granularity_preds['continent'][continent['id']] = pred_classes[loc]
                    history_[loc]['continent'][continent['id']] = continent['label']
            else:
                
                granularity_preds['country'][country] = pred_classes[loc]
                granularity_preds['continent'][continent] = pred_classes[loc]

        for g in granularity_preds:
            granularity_preds[g] = sort_by_value(granularity_preds[g])

        return granularity_preds, history_

    locs_info = get_info_locs()
    history_ = open_json('/nfs/home/tahmasebzadehg/mmg_ecir/vt_cm/output/history.json')
    sample_ids = list( results.keys() )
    accuracy_file_meta = {}

    for i, sample_id in enumerate(sample_ids):

        dic_cms = {}
        print('getting results for vt-cm', i, len(results))

        if is_per_sentence:
            for es_sentence in results[sample_id]:
                for e in es_sentence['entities_cms']:
                    if e["type_wikidata"] == "LOCATION":
                        dic_cms[e["wd_id"]] = e["cms"]
        else:
            for e in results[sample_id]["entities_cms"]:
                if e["type_wikidata"] == "LOCATION":
                    dic_cms[e["wd_id"]] = e["cms"]
                
        sorted_dic_cms = sort_by_value(dic_cms)
        pred_classes, history_ = fix_granularity_preds(sorted_dic_cms, history_)
        accuracy_file_meta[sample_id] = {'city': sorted_dic_cms, 'country':  pred_classes['country'], 'continent': pred_classes['continent']}

    save_file(f'{args.dir_results}/vt-cm_meta.json', accuracy_file_meta)

## accuracy meta
results = open_json(f'{args.dir_results}/vt-cm_meta_per_body.json')  
# get_meta_vt_cm(is_per_sentence = False, results=results)
valid_test_ids_per_g = get_valid_test_ids(test_h5_dir = args.dir_test_h5 )

## get accuracy
accuracy_file = get_pure_accuracy(baseline='vt-cm',
                                  partitions = ['city','country','continent'], 
                                  valid_test_ids_per_g=valid_test_ids_per_g, 
                                  path_test_data = f'{args.dir_test_texts}/test.json',
                                  path_acc_meta = f'{args.dir_results}/vt-cm_meta.json' )
save_file(f'{args.dir_results}/acc_vt-cm.json', accuracy_file)

# get gcd
coords_history = open_json(args.coords_history) 
gcd_file, coords_history = classify_gcd(output_path = args.dir_results,
                                        accuracy_file_meta = open_json(f'{args.dir_results}/vt-cm_meta.json'),
                                        dir_test_data = f'{args.dir_test_texts}/test.json',
                                        baseline = 'vt-cm',
                                        partitions = ['city','country','continent'],
                                        valid_test_ids_per_g = valid_test_ids_per_g,
                                        coords_history = coords_history)

save_file(f'{args.dir_results}/gcd_vt-cm.json', gcd_file)


# print results
baselines_print_results(args.dir_results)
