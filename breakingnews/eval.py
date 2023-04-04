import os
import sys
# from bn_args import get_parser
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import random
import torch
import logging
import numpy as np
import torch.nn as nn
from pathlib import Path
from mmg_news.models.v_clip import v_clip
from mmg_news.models.t_1bert import t_1bert
from mmg_news.models.t_2bert import t_2bert
from mmg_news.models.m_1bert_clip import m_1bert_clip
from mmg_news.models.m_2bert_clip import m_2bert_clip
import torch.utils.data as data
import h5py
from utils.global_utils import *
import glob
import argparse
ROOT_PATH = Path(os.path.dirname(__file__))

def get_parser():
    parser = argparse.ArgumentParser(description='BreakingNews')
    # paths
    parser.add_argument('--data_path', default=f'/nfs/home/tahmasebzadehg/mmg_news_dataset/h5_splits')
    parser.add_argument('--dataset_version', default='v1')
    # parser.add_argument('--snapshots', default=f'{root}/../mmg_news/experiments/snapshots', type=str)
    parser.add_argument('--snapshots', default=f'{ROOT_PATH}/experiments/snapshots', type=str)
    parser.add_argument('--test_results', default=f'{ROOT_PATH}/evaluation', type=str)
    parser.add_argument('--images_dir', default=f'/nfs/home/tahmasebzadehg/mmg_news_dataset/image_splits', type=str)
    parser.add_argument('--logging_path', default=f'{ROOT_PATH}/log', type=str)
    parser.add_argument('--all_locations', default=f'/nfs/home/tahmasebzadehg/mmg_news_dataset/info/labels.json', type=str)
    parser.add_argument('--freeze_first_layers_image',default=True, type=bool)
    parser.add_argument('--freeze_all_image',default=False, type=bool)
    parser.add_argument('--bn_mmg_to_coord', default= '/nfs/home/tahmasebzadehg/mmg_news_dataset/info/bn_mmg_to_coord.json')
        
    # general
    parser.add_argument('--workers', default=6, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--batch_size', default=256, type=int)

    # model
    parser.add_argument('--emb_dim', default=1024, type=int)
    parser.add_argument('--text_dim', default=768, type=int)
    parser.add_argument('--attention_dim', default=6144, type=int)
    parser.add_argument('--heads', default=4, type=int)
    parser.add_argument('--multi_head', default=6144, type=int)
    parser.add_argument('--n_classes_city', default=14331, type=int)  
    parser.add_argument('--n_classes_country', default=241, type=int)
    parser.add_argument('--n_classes_continent', default=6, type=int) 
    parser.add_argument('--n_classes_domain', default=10, type=int)  
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--test_checkpoint', default='epoch_120_loss_1.95.pth.tar', type=str)

    # train
    parser.add_argument('--lr', default=0.0000001, type=float) #default=0.00001
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--no_base', default=True, type=bool)
    
    # train & test
    parser.add_argument('--task', default = 'hierarchical', type=str, help= '(hierarchical/ city/ country/ continent) all, only city, only country or only continent')
    parser.add_argument('--train_domain', default = False, type=bool) 
    parser.add_argument('--cross_modal', default  = False, type=bool) 
    parser.add_argument('--multimodal_combine', default='concat', help='concat or max', type=str)
    parser.add_argument('--model_name', default='v_clip', type=str, help='v_clip, [t_body, t_entity ,t_2bert], [m_2bert_clip] ' ) 
    parser.add_argument('--tensorboard', default = True, type=bool )

    return parser

class Data_Loader(data.Dataset):
    def __init__(self, data_path, partition, fname):

        if data_path == None:
            raise Exception('No data path specified.')

        self.h5f = None
        self.partition = partition
        self.data_path = data_path
        self.fname = fname
        # self.h5_path = f'{self.data_path}/{self.partition}/{self.fname}.h5'
        self.h5_path = f'{self.data_path}/{self.partition}/{self.partition}_bn.h5'
        
        with h5py.File(name=self.h5_path, mode='r') as  file:
            self.ids = [id for id in file['ids']]
            self.len_ids = len(self.ids)

        p = 0
        

    def __getitem__(self, index):
        if self.h5f == None:
            self.h5f = h5py.File(self.h5_path, mode='r')

        instanceId = self.ids[index]
        # print(instanceId)
        grp = self.h5f[instanceId]
        clip = grp[f'image_clip'][()]
        all_clip = grp[f'all_images_clip'][()]
        avg_clip = grp[f'avg_images_clip'][()]
        bert_body = grp['bert_entity'][()]
        bert_entity = grp['bert_entity'][()]


        loc = grp['loc'][()]


        if len(loc.shape) == 1:
            loc = np.array([loc])

        d = 50 - loc.shape[0]
        
        for _ in range(d):
            t = np.array([[1000, 1000]])
            loc = np.append(loc, t, 0)

        domain = int(grp['domain'][()])

        item =  {
            'id': instanceId,
            # 'image_clip': clip,
            'image_clip': avg_clip,
            'body': bert_body,
            'entity': bert_entity,
            'domain':domain,
            'loc': loc
        }
        
        return item

    def __len__(self):
        return self.len_ids


def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile)

# read parser
parser = get_parser()
bn_args = parser.parse_args()

# create directories for train experiments
logging_path = f'log'
# Path(logging_path).mkdir(parents=True, exist_ok=True)

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
random.seed(bn_args.seed)
np.random.seed(bn_args.seed)
if torch.cuda.is_available():
    torch.manual_seed(bn_args.seed)
    torch.cuda.manual_seed(bn_args.seed)
    torch.cuda.manual_seed_all(bn_args.seed)

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def main():  #inputs: bn_args.model_name , bn_args.multimodal_combine, bn_args.train_domain
    # set model
        
        # create directories for train experiments
        [mid_folder] = [f'{bn_args.task}_' if bn_args.task != 'hierarchical' else '']

        # mid_folder +=  bn_args.model_name
        mid_folder += 'BN_' + bn_args.model_name

        if bn_args.model_name in ['m_body_clip', 'm_entity_clip', 'm_2bert_clip', 't_2bert']:
            path_snapshots = f'{bn_args.snapshots}/{mid_folder}/{bn_args.multimodal_combine}'
            results_dir = f'{bn_args.test_results}/{mid_folder}/{bn_args.multimodal_combine}'
        else:
            path_snapshots = f'{bn_args.snapshots}/{mid_folder}'
            results_dir = f'{bn_args.test_results}/{mid_folder}/{bn_args.multimodal_combine}'

        # model_path = f'/nfs/home/tahmasebzadehg/mmg/train1/experiments/snapshots/BN_m_2bert_clip/concat/epoch_110_loss_2.09.pth.tar'
        # model_path = f'/nfs/home/tahmasebzadehg/mmg/train1/experiments/snapshots/BN_t_2bert/concat/epoch_108_loss_2.31.pth.tar'   
        model_path = f'/nfs/home/tahmasebzadehg/mmg/train1/experiments/snapshots/BN_v_clip/epoch_107_loss_2.87.pth.tar'
        # model_path = f'{path_snapshots}/{bn_args.test_checkpoint}'
        

        if bn_args.model_name == 'v_clip':
            model =  v_clip()
        elif bn_args.model_name  in ['t_body', 't_entity']:
            model =  t_1bert()
        elif bn_args.model_name  == 't_2bert':
            model =  t_2bert()
        elif bn_args.model_name  in['m_body_clip', 'm_entity_clip']:
            model =  m_1bert_clip()
        elif bn_args.model_name  == 'm_2bert_clip':
            model =  m_2bert_clip()


        model.to(device)
        
        logger.info(f"=> loading checkpoint '{model_path}'")

        checkpoint = torch.load(model_path, encoding='latin1',map_location=torch.device(device))

        bn_args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"=> loaded checkpoint '{model_path}' (epoch {checkpoint['epoch']})")
        test_vers = {'city':{},'country':{},'continent':{}}
        
        
        if bn_args.task == 'hierarchical':
            granularities = ['city']
        else:  # city/country/continent
            granularities = [bn_args.task]    
        
        for g in granularities:

            data_loader_func = Data_Loader(data_path=f'{bn_args.data_path}', partition='test', fname =f'test_bn')  
            
            print(f'test version is {g}:')
            # prepare test loader
            test_loader = torch.utils.data.DataLoader(
                data_loader_func,
                batch_size=bn_args.batch_size,

                shuffle=False,
                num_workers=bn_args.workers,
                pin_memory=False)

            logger.info('Test loader prepared.')

            test_vers[f'{g}'] = test(test_loader, model, g, at = 1)

            
            print(f'********{g} mean_gcd:', np.round(test_vers[f'{g}']['mean']/1000, 2),  '-  median:', np.round(test_vers[f'{g}']['median']/1000, 2) )

            # save_file(f'/nfs/home/tahmasebzadehg/acmm22/train1/evaluation/bn/{bn_args.model_name}_list_ids_invalid.json', test_vers['city']['n_unk'])
        


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


def test(test_loader, model, g, at):
        # switch to evaluate mode
        file_maps = open_json(bn_args.bn_mmg_to_coord)
        model.eval()
        top1_preds = {'city':[], 'country':[], 'continent':[]}
        test_loader_ids = []

        for i_counter, test_input in enumerate(test_loader):
            test_loader_ids.extend(test_input['id'])

            output = {'city':'', 'country':'', 'continent':''}
            
            # compute output
            if bn_args.model_name == 'v_clip':
                output['city'], output['country'], output['continent']= model(test_input['image_clip'].to(device))
            elif bn_args.model_name == 't_body':
                output['city'], output['country'], output['continent'] =  model(test_input['body'].to(device))
            elif bn_args.model_name == 't_entity':
                output['city'], output['country'], output['continent']  = model(test_input['entity'].to(device))
            elif bn_args.model_name == 't_2bert':
                output['city'], output['country'], output['continent'] = model(test_input['body'].to(device), test_input['entity'].to(device))
            elif bn_args.model_name == 'm_body_clip':
                output['city'], output['country'], output['continent'] ,_,_ = model(test_input['image_clip'].to(device), test_input['body'].to(device))
            elif bn_args.model_name == 'm_entity_clip':
                output['city'], output['country'], output['continent'],_,_  = model(test_input['image_clip'].to(device), test_input['entity'].to(device))
            elif bn_args.model_name == 'm_2bert_clip':
                output['city'], output['country'], output['continent'] ,_,_  = model(test_input['image_clip'].to(device), test_input['body'].to(device), test_input['entity'].to(device))

            target = {'loc':test_input['loc']}

            # topk preds
            for g_counter in ['city', 'country', 'continent']:
                top1_preds[g_counter].append([[t, torch.topk(o, k=at)[1]] for o, t in zip(output[g_counter], target['loc'])])

        gcd_sum, gcd_list = classify_gcd(top1_preds[g], at, g, test_loader_ids,  file_maps = file_maps)
        
        result = { 'gcd_sum': gcd_sum, 'gcd_list':gcd_list,  'mean':gcd_sum/len(gcd_list), 'median': statistics.median(gcd_list)}
 
        
        return result


if __name__ == '__main__':
    main()

