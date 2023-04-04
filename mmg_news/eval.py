import os
import time
import random
import torch
import logging
import numpy as np
import torch.nn as nn
from pathlib import Path
from args import get_parser
from models.v_clip import v_clip
from models.t_1bert import t_1bert
from models.t_2bert import t_2bert
from models.m_1bert_clip import m_1bert_clip
from models.m_2bert_clip import m_2bert_clip
from data_loader import  Data_Loader
from utils.global_utils import *


ROOT_PATH = Path(os.path.dirname(__file__))


# read parser
parser = get_parser()
args = parser.parse_args()
logging_path = 'log'

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{ROOT_PATH}/{logging_path}/test.log', 'w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():  #inputs: args.model_name , args.multimodal_combine, args.train_domain

        # create directories for train experiments
        [mid_folder] = [f'{args.task}_' if args.task != 'hierarchical' else '']

        mid_folder += args.model_name

        if args.model_name in ['m_body_clip', 'm_entity_clip', 'm_2bert_clip', 't_2bert']:
            path_snapshots = f'{args.snapshots}/{mid_folder}/{args.multimodal_combine}'
            results_dir = f'{args.test_results}/{mid_folder}/{args.multimodal_combine}'
        else:
            path_snapshots = f'{args.snapshots}/{mid_folder}'
            results_dir = f'{args.test_results}/{mid_folder}/{args.multimodal_combine}'
            
        model_path = f'{path_snapshots}/{args.test_checkpoint}'

        if args.model_name == 'v_clip':
            model =  v_clip()
        elif args.model_name  in ['t_body', 't_entity']:
            model =  t_1bert()
        elif args.model_name  == 't_2bert':
            model =  t_2bert()
        elif args.model_name  in['m_body_clip', 'm_entity_clip']:
            model =  m_1bert_clip()
        elif args.model_name  == 'm_2bert_clip':
            model =  m_2bert_clip()

        model.to(device)
        
        logger.info(f"=> loading checkpoint '{model_path}'")
                               
        checkpoint = torch.load( model_path, encoding='latin1', map_location=torch.device(device) )

        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"=> loaded checkpoint '{model_path}' (epoch {checkpoint['epoch']})")
        test_vers = {'city':{},'country':{},'continent':{}}
        outputs_with_ids = {'city':{},'country':{},'continent':{}}
        
        
        if args.task == 'hierarchical':
            granularities = ['city','country','continent']
        else:  # city/country/continent
            granularities = [args.task]

        for g in granularities:
            data_loader_func = Data_Loader(data_path=f'{args.data_path}', partition='test', fname = f'{g}') 
            
            print(f'test version is {g}:')
            # prepare test loader
            test_loader = torch.utils.data.DataLoader(
                data_loader_func,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False)

            logger.info('Test loader prepared.')

            test_vers[f'{g}'], outputs_with_ids = test(test_loader, model, g, outputs_with_ids)

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        save_file(f'{results_dir}/{args.test_checkpoint}.json', test_vers)
        Path(f'{results_dir}/ids').mkdir(parents=True, exist_ok=True)
        save_file(f'{results_dir}/ids/ids_{args.test_checkpoint}.json', outputs_with_ids)
        print('saved to',f'{results_dir}/{args.test_checkpoint}.json' )



def get_predictions_by_id(ids, classify_gcds1_, classify_gcds5_, classify_gcds10_):
    def get_loc_name(loc):
        if 'city' in loc:
            return loc['city']
        elif 'country' in loc:
            return loc['country']
        else:
            return loc['continent']
    
    location_to_class = list(open_json('data/location_to_class.json').keys())
    loc_info = open_json('data/all_locations_test.json')
    results = {}
    classify_gcds1 = []
    classify_gcds5 = []
    classify_gcds10 = []
    [classify_gcds1.extend(classify_gcds1__) for classify_gcds1__ in classify_gcds1_]
    [classify_gcds5.extend(classify_gcds5__) for classify_gcds5__ in classify_gcds5_]
    [classify_gcds10.extend(classify_gcds10__) for classify_gcds10__ in classify_gcds10_]

    for i, id in enumerate(ids):
        
        gt, top1 = classify_gcds1[i]
        _, top5 = classify_gcds5[i]
        _, top10 = classify_gcds10[i]
        
        gt_loc = get_loc_name(loc_info[location_to_class[int(gt)]])
        top1_loc = get_loc_name(loc_info[location_to_class[int(top1)]])
        top5_locs = [get_loc_name(loc_info[location_to_class[int(loc)]]) for loc in top5]
        top10_locs = [get_loc_name(loc_info[location_to_class[int(loc)]]) for loc in top10]
        results[id] = {'ground_truth':gt_loc, 'top1':top1_loc, 'top5':top5_locs, 'top10':top10_locs}
    return results


def test(test_loader, model, g, outputs_with_ids):
        # switch to evaluate mode
        model.eval()
              
        logits = {'city':[], 'country':[], 'continent':[]}
        targets = {'city':[], 'country':[], 'continent':[]}
        logits_pure = {'city':[], 'country':[], 'continent':[]}
        preds = {'city':[], 'country':[], 'continent':[]}

        n_classes = {'city':args.n_classes_city, 'country':int(args.n_classes_country), 'continent':int(args.n_classes_continent)}

        track_ids_test = []


        for i_counter, test_input in enumerate(test_loader):
            for iii in test_input['id']:
                track_ids_test.append(iii)

            output = {'city':'', 'country':'', 'continent':''}
            
            # compute output
            if args.model_name == 'v_clip':
                output['city'], output['country'], output['continent']= model(test_input['image_clip'].to(device))
            elif args.model_name == 't_body':
                output['city'], output['country'], output['continent'] =  model(test_input['body'].to(device))
            elif args.model_name == 't_entity':
                output['city'], output['country'], output['continent']  = model(test_input['entity'].to(device))
            elif args.model_name == 't_2bert':
                output['city'], output['country'], output['continent']  = model(test_input['body'].to(device), test_input['entity'].to(device))
            elif args.model_name == 'm_body_clip':
                output['city'], output['country'], output['continent'] ,_,_ = model(test_input['image_clip'].to(device), test_input['body'].to(device))
            elif args.model_name == 'm_entity_clip':
                output['city'], output['country'], output['continent'],_,_  = model(test_input['image_clip'].to(device), test_input['entity'].to(device))
            elif args.model_name == 'm_2bert_clip':
                output['city'], output['country'], output['continent'], _,_  = model(test_input['image_clip'].to(device), test_input['body'].to(device), test_input['entity'].to(device))
            elif args.model_name == 'd_concat_v_clip':
                output['city'], output['country'], output['continent'] = model(test_input['image_clip'].to(device))

            target = {'city':test_input['city'],
                    'country':test_input['country'],
                    'continent': test_input['continent']}

            # topk preds
            for g_counter in ['city', 'country', 'continent']:
                logits[g_counter].extend(output[g_counter].cpu())
                logits_pure[g_counter].extend(output[g_counter])
                preds[g_counter].extend(torch.topk(o, k=n_classes[g_counter])[1].detach().numpy() for o in output[g_counter])
                targets[g_counter].extend(target[g_counter])
          
        for i_, id in enumerate(track_ids_test):
                
                cities = [str(c) for c in list(preds['city'][i_])[:20]]
                countries = [str(c) for c in list(preds['country'][i_])[:20]]
                continents = [str(c) for c in list(preds['continent'][i_])[:20]]

                outputs_with_ids[g][id] = {
                'pred':{'city':cities, 'country': countries, 'continent': continents } ,
                'gt':{'city':str(targets['city'][i_].item()), 'country':str(targets['country'][i_].item()), 'continent':str(targets['continent'][i_].item()) }
                 }     
  
        ## normal classification
        classification_at1 = classify_atk(logits[g], targets[g], 1 ) 
        classification_at2 = classify_atk(logits[g],targets[g], 2)
        classification_at5 = classify_atk(logits[g],targets[g], 5 )
        [classification_at10] = [classify_atk(logits[g],targets[g], 10 ) if g != 'continent' else 1.0 ]
        print(f' {g} accuracy done!', classification_at1, classification_at2, classification_at5, classification_at10 )

        ## gcd
        gcd1 = classify_gcd(preds=preds[g], targets=targets[g], at=1, g=g)
        print('gcd done!', gcd1 )

        result = {
                    'acc@1': classification_at1,
                    'acc@2': classification_at2,  
                    'acc@5': classification_at5,
                    'acc@10': classification_at10,
                    'GCD': gcd1
                 }
       
        return result, outputs_with_ids


if __name__ == '__main__':
    main()

