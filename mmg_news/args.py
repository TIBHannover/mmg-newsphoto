import argparse
from pathlib import Path
import os
root = Path(os.path.dirname(__file__))

def get_parser():
    parser = argparse.ArgumentParser(description='MMG-News')
    # paths
    parser.add_argument('--data_path', default=f'mmg_news_dataset/h5_splits')
    parser.add_argument('--snapshots', default=f'{root}/experiments/snapshots', type=str)
    parser.add_argument('--test_results', default=f'{root}/evaluation', type=str)
    parser.add_argument('--images_dir', default=f'mmg_news_dataset/image_splits', type=str)
    parser.add_argument('--logging_path', default=f'{root}/log', type=str)
    parser.add_argument('--all_locations', default=f'mmg_news_dataset/info/labels.json', type=str)
    parser.add_argument('--freeze_first_layers_image',default=True, type=bool)
    parser.add_argument('--freeze_all_image',default=False, type=bool)
        
    # general
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--batch_size', default=256, type=int)

    # model
    parser.add_argument('--emb_dim', default=1024, type=int)
    parser.add_argument('--text_dim', default=768, type=int)
    parser.add_argument('--n_classes_city', default=14331, type=int)  
    parser.add_argument('--n_classes_country', default=241, type=int)
    parser.add_argument('--n_classes_continent', default=6, type=int) 
    parser.add_argument('--n_classes_domain', default=10, type=int)  
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--test_checkpoint', default='', type=str)

    # train
    parser.add_argument('--lr', default=0.00001, type=float) 
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--no_base', default=True, type=bool)
    
    # train & test
    parser.add_argument('--task', default = 'continent', type=str, help= '(hierarchical/ city/ country/ continent) all, only city, only country or only continent')
    parser.add_argument('--cross_modal', default  = False, type=bool) 
    parser.add_argument('--multimodal_combine', default='concat', help='concat or max', type=str)
    parser.add_argument('--model_name', default='m_2bert_clip', type=str, help='v_clip, [t_body, t_entity ,t_2bert], [m_2bert_clip] ' ) 
    parser.add_argument('--tensorboard', default = True, type=bool )

    return parser


