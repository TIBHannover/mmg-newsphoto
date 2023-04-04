import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
from args import get_parser
import torch
parser = get_parser()
args = parser.parse_args()

if args.multimodal_combine == 'concat':
    classifier_emb = 3*args.emb_dim
    classifier_emb_domain = 3*args.emb_dim
else:
    classifier_emb = args.emb_dim
    classifier_emb_domain = args.emb_dim

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
            nn.Linear(classifier_emb, 512),
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
            nn.Linear(classifier_emb, 512),
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
            nn.Linear(classifier_emb, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, tags)
        )

    def forward(self, x):
        cn = self.combine_net(x)
        return cn
 

class m_2bert_clip(nn.Module):
    def __init__(self):
        super(m_2bert_clip, self).__init__()

        # if args.no_base == False:
        #     self.base_model = get_base_model()
        self.learn_image = LearnImages()
        self.learn_text = LearnText()
        [nclasses_city] = [ args.n_classes_city]
        [nclasses_country] = [ args.n_classes_country]
        [nclasses_continent] = [ args.n_classes_continent]
        self.city_classifier = Classify_city(tags=nclasses_city)
        self.country_classifier = Classify_country(tags=nclasses_country)
        self.continent_classifier = Classify_continent(tags=nclasses_continent)


    def forward(self, image_input, text_input_body, text_input_entity):

        if args.no_base == False:
            image_input = self.base_model(image_input)
            
        image_emb = self.learn_image(image_input)
        text_emb_body = self.learn_text(text_input_body)
        text_emb_entity = self.learn_text(text_input_entity)

        if args.multimodal_combine == 'max':
            image_output_city = self.city_classifier(image_emb)
            image_output_country = self.country_classifier(image_emb)
            image_output_continent = self.continent_classifier(image_emb)

            body_output_city = self.city_classifier(text_emb_body)
            body_output_country = self.country_classifier(text_emb_body)
            body_output_continent = self.continent_classifier(text_emb_body)

            entity_output_city = self.city_classifier(text_emb_entity)
            entity_output_country = self.country_classifier(text_emb_entity)
            entity_output_continent = self.continent_classifier(text_emb_entity)

            output_city =  torch.max(image_output_city, torch.max(body_output_city, entity_output_city ))
            output_country = torch.max(image_output_country, torch.max(body_output_country, entity_output_country))
            output_continent = torch.max(image_output_continent, torch.max(body_output_continent, entity_output_continent))

        else:
            text_emb = torch.cat((text_emb_body, text_emb_entity), 1)
            
            multimodal_emb = torch.cat((text_emb, image_emb), 1)

            output_city = self.city_classifier(multimodal_emb)
            output_country = self.country_classifier(multimodal_emb)
            output_continent = self.continent_classifier(multimodal_emb)

        return output_city, output_country, output_continent, image_emb, text_emb_entity