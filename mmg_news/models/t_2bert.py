import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
from args import get_parser
import torch

parser = get_parser()
args = parser.parse_args()


class Norm(nn.Module):
    def forward(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


# embed text
class LearnText(nn.Module):
    def __init__(self,dropout=args.dropout):
        super(LearnText, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(2*args.text_dim, args.emb_dim),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.Tanh(),
            Norm()
        )

    def forward(self, x):
        return self.embedding(x)


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



class t_2bert(nn.Module):
    def __init__(self):
        super(t_2bert, self).__init__()
        self.learn_text = LearnText()
        self.city_classifier = Classify_city()
        self.country_classifier = Classify_country()
        self.continent_classifier = Classify_continent()

    def forward(self, text_input1, text_input2):
        text_input = torch.cat((text_input1, text_input2),1)
        text_emb = self.learn_text(text_input)
        
        output_city = self.city_classifier(text_emb)
        output_country = self.country_classifier(text_emb)
        output_continent = self.continent_classifier(text_emb)

        return output_city, output_country, output_continent