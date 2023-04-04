import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
import torch
from args import get_parser
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
            nn.Linear(args.text_dim, args.emb_dim),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.Tanh(),
            Norm()
        )

    def forward(self, x):
        return self.embedding(x)


class LearnImages(nn.Module):
    def __init__(self,dropout=args.dropout):
        super(LearnImages, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(512, args.emb_dim), #2048
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.Tanh(),
            Norm()

        )

    def forward(self, x):
        r = self.embedding(x)
        return r

class Fix_dim(nn.Module):
    def __init__(self, tags=args.n_classes_city, dropout=args.dropout):
        super(Fix_dim, self).__init__()
        self.combine_net = nn.Sequential(
            nn.Linear(2*args.emb_dim, args.emb_dim)
        )

    def forward(self, x):
        cn = self.combine_net(x)
        return cn

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


class m_1bert_clip(nn.Module):
    def __init__(self):
        super(m_1bert_clip, self).__init__()
        self.learn_image = LearnImages()
        self.learn_text = LearnText()
        self.city_classifier = Classify_city()
        self.country_classifier = Classify_country()
        self.continent_classifier = Classify_continent()
        self.fix_dim = Fix_dim()

    def forward(self, image_input, text_input):

        if args.no_base == False:
            image_input = self.base_model(image_input)
            
        image_emb = self.learn_image(image_input)
        text_emb = self.learn_text(text_input)

        if args.multimodal_combine == 'max':
            image_output_city = self.city_classifier(image_emb)
            image_output_country = self.country_classifier(image_emb)
            image_output_continent = self.continent_classifier(image_emb)

            text_output_city = self.city_classifier(text_emb)
            text_output_country = self.country_classifier(text_emb)
            text_output_continent = self.continent_classifier(text_emb)

            output_city =  torch.max(image_output_city, text_output_city)
            output_country = torch.max(image_output_country, text_output_country)
            output_continent = torch.max(image_output_continent, text_output_continent)

        elif args.multimodal_combine == 'concat':
            multimodal_emb1 = torch.cat((image_emb, text_emb),1)

            multimodal_emb = self.fix_dim(multimodal_emb1)   

            output_city = self.city_classifier(multimodal_emb)
            output_country = self.country_classifier(multimodal_emb)
            output_continent = self.continent_classifier(multimodal_emb)

        return output_city, output_country, output_continent, image_emb, text_emb