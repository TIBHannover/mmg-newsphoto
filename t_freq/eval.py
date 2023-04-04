import sys
from pathlib import Path
import os
import glob
import json
import argparse
from utils.global_utils import *

root = Path(os.path.dirname(__file__))

def get_parser():
    parser = argparse.ArgumentParser(description='T-Freq')
    # parser.add_argument('--dir_val_texts', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/text_splits/val')
    parser.add_argument('--dir_train_texts', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/text_splits/train')
    parser.add_argument('--dir_test_texts', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/text_splits/test')
    parser.add_argument('--dir_test_h5', default='/nfs/home/tahmasebzadehg/mmg_news_dataset/h5_splits/test')

    parser.add_argument('--dir_results', default="/nfs/home/tahmasebzadehg/mmg_ecir/t_freq/output")
    # parser.add_argument('--path_test_images', default="/nfs/home/tahmasebzadehg/image_splits/test")
    # parser.add_argument('--path_data_locs', default="/nfs/home/tahmasebzadehg/mmg_ecir/outputss/data_locs.json")
    parser.add_argument('--save_tags_t_freq_wikidata', default=False)
    parser.add_argument('--coords_history', default="/nfs/home/tahmasebzadehg/mmg_news_dataset/info/coords_history.json")
    return parser

parser = get_parser()
args = parser.parse_args()


def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile)

def update_tag(tag, output_tag_count_per_cell):
    
    output_tag_count_per_cell[tag['wd_id']] = {
        'wd_label':tag['wd_label'],
        'wdimage': tag['wdimage'],
        # 'reference_images': tag['reference_images'],
        'type_wikidata': tag['type_wikidata'],
        'locs': {'city':{}, 'country':{}, 'continent':{}}
        }
    return tag

def fix_self(loc):
    try:
        if loc['country']['id'] == 'self':
            loc['country'] = loc['location']
        if loc['continent']['id'] == 'self':
            loc['continent'] = loc['location']
    except:
        return loc
    return loc

# per tag in body (per article), count how many times the image label (city, country, continent) has seen this tag
def update_tag_count_per_loc(output_tag_count_per_loc, loc, body_entities_counts):
    # body_entities_counts: the number of times each entity is mentioned
    for tag_wd_id in body_entities_counts:
        tag = body_entities_counts[tag_wd_id]
        
        if tag['wd_id'] not in output_tag_count_per_loc:
            tag = update_tag(tag, output_tag_count_per_loc)

        tag_cells =  output_tag_count_per_loc[tag_wd_id]['locs']

        for g in ['city', 'country', 'continent']:
            if 'id' in loc[g].keys(): # if there is no location/country(e.g. {})
                if loc[g]['id'] not in tag_cells[g].keys():
                    tag_cells[g][loc[g]['id']] = body_entities_counts[tag_wd_id]['count']
                else:
                    tag_cells[g][loc[g]['id']] += body_entities_counts[tag_wd_id]['count']
    
    return output_tag_count_per_loc

def get_dataset_entities_per_loc():
    if os.path.exists(f'{args.dir_results}/t_freq_wikidata_tags.json'):
        return

    output_tag_count_per_loc = {}
    files = glob.glob(f'{args.dir_train_texts}/*')
    errors = []

    for i_f,f in enumerate(files):
        samples = open_json(f)
          
        for i_a, sample_id in enumerate(samples):
            if i_a % 21 == 0:
                print(f' {i_f+1}/{len(files)} | {i_a+1}/{len(samples)} samples - #tags: {len(output_tag_count_per_loc)}')
    
            sample = samples[sample_id]
            body_entities_counts = get_entities_counts(sample['body'])
            
            label = sample['image_label']
            output_tag_count_per_loc = update_tag_count_per_loc(output_tag_count_per_loc, label, body_entities_counts)
                            
        save_file(f'{args.dir_results}/t_freq_wikidata_tags.json', output_tag_count_per_loc)

def N_t(tag_t_id, tags_per_cell, partition):    # N_t Total number of times tag t is seen
    tag = tags_per_cell[tag_t_id]
    freq = 0

    for geocell_class in tag['geo-cells'][partition]:
        freq += tag['geo-cells'][partition][geocell_class]
    
    return freq

def N_u(tag_t_id, c_i, tags_per_cell, partition):
    tag = tags_per_cell[tag_t_id]
    s = 0

    for geocell_class in tag['geo_cells'][partition]:
        if geocell_class == c_i:
            s += tag['geo_cells'][partition][geocell_class]
    return s

def sort_by_value(p):
    s = dict(sorted(p.items(), key=lambda item: item[1], reverse=True))
    return s

def get_results_per_sample_with_Max(cells, dataset_entities_per_cell, test_data_tags,label):
    
    probs = {"coarse":{}, "middle":{}, "fine": {}}
    try:
        true_clses = {"coarse":cells[label]["coarse"]['cell']['class'],
                "middle":cells[label]["middle"]['cell']['class'],
                "fine": cells[label]["fine"]['cell']['class']}
    except:
        print('no true class !!')
        return None, None

    for p in ["coarse", "middle", "fine"]:

        for t in test_data_tags:
            for cls in dataset_entities_per_cell[t]['geo-cells'][p]:
                
                n_u = dataset_entities_per_cell[t]['geo-cells'][p][cls] # In cell c how many times tag t is seen 
                n_t = N_t(t, dataset_entities_per_cell, p) # N_t Total number of times tag t is seen
                prob_p_c = n_u/n_t # p(t|c)

                if cls in probs[p].keys():   # if another tag from same class exists then take the max n_u/n_t
                    if probs[p][cls] < prob_p_c:
                        probs[p][cls] = prob_p_c  # p(t|c)
                else:
                    probs[p][cls] = prob_p_c  # p(t|c)        
        
    prob_sorted_coarse = sort_by_value(probs["coarse"]) # final sorted freq - top1 is the correct loc geocell
    prob_sorted_middle = sort_by_value(probs["middle"]) 
    prob_sorted_fine = sort_by_value(probs["fine"]) 
    preds = {"coarse":prob_sorted_coarse, "middle":prob_sorted_middle, "fine": prob_sorted_fine}

    return true_clses, preds

def get_body_caption(id, resource):
            body = None
            try:
                s = f"{root}/../data_collection/ccnews/output/{resource}"
                farticles = glob.glob(f'{s}/nel/*')
                found = False
                for farticle in farticles:
                    
                    if found:
                        break
                    articles = open_json(farticle)

                    for id_article in articles:
                        if id == id_article:
                            body = articles[id_article]["body"]
                            found = True
                            break
            except Exception:
                print(f'cant read {farticle}')
            return body

def get_unique_entities(bodys):
    entities = []
   
    for body in bodys:
        for e in body["named_entities"]:
            entities.append(e["wd_id"])
            # if e['type_wikidata'] == "LOCATION":
            #     locs.add(e["wd_id"])
    return set(entities), entities

def get_entities_counts(bodys):
    entities = {}
    for body in bodys:
        for e in body["named_entities"]:
            ewd = e["wd_id"]
            if ewd in entities:
                entities[ewd]['count'] += 1
            else:
                entities[ewd] = e
                entities[ewd]['count'] = 1
    return entities

def get_accuracy_meta(samples, tags_path, acc_meta_path):
    if os.path.exists(acc_meta_path):
        return

    def update_dic(dic_in, dic_main):
        for id in dic_in:
            if id!='':
                dic_main[id] = dic_in[id]
        return dic_main

    results = {}

    tags = open_json(tags_path)

    for c_sample, sample_id in enumerate(samples):

        print(c_sample, len(samples))
        sample = samples[sample_id]

        results[sample_id] = {}

        body_entities = []

        for b in sample['body']:
            body_entities.extend(b['named_entities'])
        
        freqs_granularities = {'city':{},'country':{},'continent':{}}
        for e in body_entities:
            try:
                for g in ['city', 'country', 'continent']:
                    # how many times each tag is repeated per location 
                    freqs_granularities[g] = update_dic(tags[e['wd_id']]['locs'][g] , freqs_granularities[g])  
            except:  # it means this tag doen't exist in train data
                continue

        freq = {}

        for g in ['city', 'country', 'continent']:
            if freqs_granularities[g]=={}:
                continue
            freq_g = {k: v for k, v in sorted(freqs_granularities[g].items(), key=lambda item: item[1], reverse=True)}
            freq[g] = list(freq_g.keys())

        results[sample_id] = freq

    save_file(acc_meta_path, results)

if args.save_tags_t_freq_wikidata:
    get_dataset_entities_per_loc()

    samples_test = {**open_json(f'{args.dir_test_texts}/test.json')['city'],  **open_json(f'{args.dir_test_texts}/test.json')['country'], **open_json(f'{args.dir_test_texts}/test.json')['continent'] }

    get_accuracy_meta(samples = samples_test,   tags_path = f'{args.dir_results}/t_freq_wikidata_tags.json',  acc_meta_path = f'{root}/output/t_freq_meta.json')

valid_test_ids_per_g = get_valid_test_ids(test_h5_dir = args.dir_test_h5 )

# get acc
accuracy_file = get_pure_accuracy(
                                  baseline = 't_freq',
                                  partitions = ['city','country','continent'], 
                                  valid_test_ids_per_g = valid_test_ids_per_g, 
                                  path_test_data = f'{args.dir_test_texts}/test.json',
                                  path_acc_meta = f'{args.dir_results}/t_freq_wikidata_meta.json'
                                 )
save_file(f'{args.dir_results}/acc_t_freq.json', accuracy_file)

## get gcd
coords_history = open_json(args.coords_history) 
gcd_file, coords_history = classify_gcd(output_path = args.dir_results,
                                        accuracy_file_meta = open_json(f'{args.dir_results}/t_freq_meta.json'),
                                        dir_test_data = f'{args.dir_test_texts}/test.json',
                                        baseline = 't_freq',
                                        partitions = ['city','country','continent'],
                                        valid_test_ids_per_g = valid_test_ids_per_g,
                                        coords_history = coords_history)

save_file(f'{args.dir_results}/gcd_t_freq.json', gcd_file)



# print results
baselines_print_results(args.dir_results)