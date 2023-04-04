import torch.utils.data as data
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import h5py
from PIL import Image
import torchvision
from pathlib import Path
from args import get_parser
from utils.global_utils import return_info_cls
parser = get_parser()
args = parser.parse_args()
root = Path(os.path.dirname(__file__))


class Data_Loader(data.Dataset):
    def __init__(self, data_path, partition, fname):

        if data_path == None:
            raise Exception('No data path specified.')
        
        self.h5f = None
        self.partition = partition
        self.data_path = data_path
        self.fname = fname
        with h5py.File(name=f'{self.data_path}/{self.partition}/{self.fname}_{args.dataset_version}.h5', mode='r') as  file:
            self.ids = [id for id in file['ids']]
            self.len_ids = len(self.ids)
        

    def __getitem__(self, index):

        if self.h5f == None:  self.h5f = h5py.File(f'{self.data_path}/{self.partition}/{self.fname}_{args.dataset_version}.h5', 'r')

        instanceId = self.ids[index]
        if isinstance(instanceId, bytes):  instanceId = instanceId.decode()

        grp = self.h5f[instanceId]

        clip = grp[f'image_clip'][()]

        bert_body = grp['bert_body'][()]
        bert_entity = grp['bert_entity'][()]

        country = int(grp['country'][()])
        city = int(grp['city'][()])
        continent = int(grp['continent'][()])
        domain = int(grp['domain'][()])
        city_info = return_info_cls(city, 'city')
        country_info = return_info_cls(country, 'country')
        continent_info = return_info_cls(continent, 'continent')

        item =  {
            'id': instanceId,
            'image_clip': clip,
            'body': bert_body,
            'entity': bert_entity,
            'domain':domain,
            'city': city,
            'country': country,
            'continent': continent,
            'city_info': city_info,
            'country_info': country_info,
            'continent_info': continent_info
        }
        
        return item

    def __len__(self):
        return self.len_ids
        