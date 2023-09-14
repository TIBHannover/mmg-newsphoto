import sys
import os
from pathlib import Path
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, f'{ROOT_PATH}/..')
import random
import torch
import logging
import numpy as np
from mmg_news.mmg_args import get_parser
from mmg_news.models.v_clip import v_clip
from mmg_news.models.t_1bert import t_1bert
from mmg_news.models.t_2bert import t_2bert
from mmg_news.models.m_2bert_clip import m_2bert_clip
from mmg_news.data_loader import  Data_Loader
from utils.global_utils import *

# read parser
parser = get_parser()
args = parser.parse_args()

# create directories for train experiment
args.logging_path = f'{args.logging_path}'
Path(args.logging_path).mkdir(parents=True, exist_ok=True)

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# define device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def main():
    # create directories for train experiments
    [mid_folder] = [f'{args.task}_' if args.task != 'hierarchical' else '']
    mid_folder += args.model_name
    path_snapshots = f'{args.snapshots}/{mid_folder}'
    results_dir = f'{args.test_results}/{mid_folder}'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # load model
    if args.model_name == 'v_clip':
        model =  v_clip()
    elif args.model_name  in ['t_body', 't_entity']:
        model =  t_1bert()
    elif args.model_name  == 't_2bert':
        model =  t_2bert()
    elif args.model_name  == 'm_2bert_clip':
        model =  m_2bert_clip()

    model.to(device)
    model_path = f'{path_snapshots}/{args.test_checkpoint}'

    # set logger
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',  level=logging.INFO,  handlers=[logging.FileHandler(f'{results_dir}/test.log', 'w'), logging.StreamHandler() ])
    logger = logging.getLogger(__name__)

    # load checkpoint
    logger.info(f"=> loading checkpoint '{model_path}'") 
    checkpoint = torch.load( model_path, encoding='latin1', map_location=torch.device(device) )
    sd = {}
    for layer in iter(checkpoint["state_dict"].items()):
        if layer[0].startswith('domain'):continue
        sd[layer[0]] = layer[1]
    model.load_state_dict(sd)
    logger.info(f"=> loaded checkpoint '{model_path}' (epoch {checkpoint['epoch']})")

    # evaluate per granularity
    if args.task == 'hierarchical': granularities = ['city','country','continent']
    else: granularities = [args.task]

    test_vers = {'city':{},'country':{},'continent':{}}
    outputs_with_ids = {'city':{},'country':{},'continent':{}}

    for g in granularities:
        # prepare test data loader
        data_loader_func = Data_Loader(data_path=f'{args.data_path}',  fname = f'test_{g}') 

        test_loader = torch.utils.data.DataLoader(
            data_loader_func,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False)

        logger.info(f'{g} ==> test loader prepared')
        test_vers[g], outputs_with_ids = test(test_loader, model, g, outputs_with_ids)
        logger.info(test_vers[g])

    # store results
    save_file(f'{results_dir}/results.json', test_vers)
    save_file(f'{results_dir}/results_per_sample_id.json', outputs_with_ids)
    logger.info(f'saved to {results_dir}/{args.test_checkpoint}.json' )


def test(test_loader, model, g, outputs_with_ids):
    # switch to eval mode
    model.eval()
    logits = {'city':[], 'country':[], 'continent':[]}
    targets = {'city':[], 'country':[], 'continent':[]}
    logits_pure = {'city':[], 'country':[], 'continent':[]}
    preds = {'city':[], 'country':[], 'continent':[]}

    n_classes = {'city':args.n_classes_city, 'country':int(args.n_classes_country), 'continent':int(args.n_classes_continent)}

    track_ids = []

    for test_input in test_loader:
        track_ids.extend(test_input['id'])

        output = {'city':'', 'country':'', 'continent':''}
        
        # compute output
        if args.model_name == 'v_clip':
            output['city'], output['country'], output['continent'] = model(test_input['image_clip'].to(device))
        elif args.model_name == 't_body':
            output['city'], output['country'], output['continent'] =  model(test_input['body'].to(device))
        elif args.model_name == 't_entity':
            output['city'], output['country'], output['continent'] = model(test_input['entity'].to(device))
        elif args.model_name == 't_2bert':
            output['city'], output['country'], output['continent'] = model(test_input['body'].to(device), test_input['entity'].to(device))
        elif args.model_name == 'm_body_clip':
            output['city'], output['country'], output['continent'] = model(test_input['image_clip'].to(device), test_input['body'].to(device))
        elif args.model_name == 'm_entity_clip':
            output['city'], output['country'], output['continent'] = model(test_input['image_clip'].to(device), test_input['entity'].to(device))
        elif args.model_name == 'm_2bert_clip':
            output['city'], output['country'], output['continent'] = model(test_input['image_clip'].to(device), test_input['body'].to(device), test_input['entity'].to(device))

        target = {'city':test_input['city'], 'country':test_input['country'], 'continent': test_input['continent']}

        # topk predictions
        for g_counter in ['city', 'country', 'continent']:
            logits[g_counter].extend(output[g_counter].cpu())
            logits_pure[g_counter].extend(output[g_counter])
            preds[g_counter].extend(torch.topk(o, k=n_classes[g_counter])[1].detach().numpy() for o in output[g_counter])
            targets[g_counter].extend(target[g_counter])
    
    # topk predictions per sample id
    for i_, id in enumerate(track_ids):

        cities = [str(c) for c in list(preds['city'][i_])[:20]]
        countries = [str(c) for c in list(preds['country'][i_])[:20]]
        continents = [str(c) for c in list(preds['continent'][i_])[:20]]

        outputs_with_ids[g][id] = {
                                'pred':{'city':cities, 'country': countries, 'continent': continents } ,
                                'gt':{'city':str(targets['city'][i_].item()), 'country':str(targets['country'][i_].item()), 'continent':str(targets['continent'][i_].item()) }}     

    # compute classification accuraycy
    classification_at1 = classify_atk(logits[g], targets[g], 1 ) 
    classification_at2 = classify_atk(logits[g],targets[g], 2)
    classification_at5 = classify_atk(logits[g],targets[g], 5 )
    [classification_at10] = [classify_atk(logits[g],targets[g], 10 ) if g != 'continent' else 1.0 ]

    # compute gcd
    gcd1 = classify_gcd(preds=preds[g], targets=targets[g], at=1, g=g)

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

