import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
from args import get_parser
# from base_model_geo_vision import get_base_model

parser = get_parser()
args = parser.parse_args()


class Norm(nn.Module):
    def forward(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class LearnImages(nn.Module):
    def __init__(self,dropout=args.dropout):
        super(LearnImages, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(512, args.emb_dim), 
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.Tanh(),
            Norm()

        )

    def forward(self, x):
        r = self.embedding(x)
        return r


# combine network
class Classify_city(nn.Module):
    def __init__(self, tags=args.n_classes_city, dropout=args.dropout):
        super(Classify_city, self).__init__()
        self.combine_net = nn.Sequential(
            nn.Linear(args.emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, tags)
        )

    def forward(self, x):
        cn = self.combine_net(x)
        return cn

class Classify_country(nn.Module):
    def __init__(self, tags=args.n_classes_country, dropout=args.dropout):
        super(Classify_country, self).__init__()
        self.combine_net = nn.Sequential(
            nn.Linear(args.emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, tags)
        )

    def forward(self, x):
        cn = self.combine_net(x)
        return cn

class Classify_continent(nn.Module):
    def __init__(self, tags=args.n_classes_continent, dropout=args.dropout):
        super(Classify_continent, self).__init__()
        self.combine_net = nn.Sequential(
            nn.Linear(args.emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, tags)
        )

    def forward(self, x):
        cn = self.combine_net(x)
        return cn

class Classify_domain(nn.Module):
    def __init__(self, tags=10, dropout=args.dropout):
        super(Classify_domain, self).__init__()
        self.combine_net = nn.Sequential(
            nn.Linear(args.emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, tags)
        )

    def forward(self, x):
        cn = self.combine_net(x)
        return cn

class v_clip(nn.Module):
    def __init__(self):
        super(v_clip, self).__init__()
        self.learn_image = LearnImages()
       
        self.city_classifier = Classify_city()
        self.country_classifier = Classify_country()
        self.continent_classifier = Classify_continent()

    def forward(self, image_input):

        image_emb = self.learn_image(image_input)
            
        output_city = self.city_classifier(image_emb)
        output_country = self.country_classifier(image_emb)
        output_continent = self.continent_classifier(image_emb)

        return output_city, output_country, output_continent