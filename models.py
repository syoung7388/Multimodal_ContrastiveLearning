import torch.nn as nn
from transformers import AutoModel, ViTModel
import torch
def get_classification_layer(hidden_size, class_num):
    return nn.Sequential(
        nn.LayerNorm(hidden_size), 
        nn.Dropout(p=0.1),
        nn.ReLU(),
        nn.Linear(hidden_size, class_num)
    )

class TextModel(nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(args.text_model_name)
        if args.classification_layer:
            self.text_linear = get_classification_layer(self.text_model.config.hidden_size, args.class_num)
        else:
            self.text_linear = nn.Linear(self.text_model.config.hidden_size, args.class_num)




    def forward(self, input_ids, attention_mask):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True)
        cls_outputs = text_outputs.last_hidden_state[:, 0]
        last_outputs = self.text_linear(cls_outputs)
        return last_outputs
        




class ImageModel(nn.Module):
    def __init__(self, args):
        super(ImageModel, self).__init__()
        self.image_model = ViTModel.from_pretrained(args.image_model_name)
        if args.classification_layer:
            self.image_linear = get_classification_layer(self.image_model.config.hidden_size, args.class_num)
        else:
            self.image_linear = nn.Linear(self.image_model.config.hidden_size, args.class_num)


    def forward(self, pixel_values):
        image_outputs = self.image_model(pixel_values=pixel_values)
        cls_outputs = image_outputs.last_hidden_state[:, 0]
        last_outputs = self.image_linear(cls_outputs)
        return last_outputs


class ImgContrastiveModel(nn.Module):
    def __init__(self, args):
        super(ImgContrastiveModel, self).__init__()
        self.image_model = ViTModel.from_pretrained(args.image_model_name)
        self.contrastive_linear = nn.Linear(args.hidden_size, args.hidden_size)
    def forward(self, pixel_values):
        image_outputs = self.image_model(pixel_values=pixel_values)
        cls_outputs = image_outputs.last_hidden_state[:, 0]
        contrastive_outputs = self.contrastive_linear(cls_outputs)
        return contrastive_outputs

class TextContrastiveModel(nn.Module):
    def __init__(self, args):
        super(TextContrastiveModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(args.text_model_name)
        self.contrastive_linear = nn.Linear(args.hidden_size, args.hidden_size)


    def forward(self, input_ids, attention_mask):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True)
        cls_outputs = text_outputs.last_hidden_state[:, 0]
        contrastive_outputs = self.contrastive_linear(cls_outputs)
        return contrastive_outputs

class MultimodalModel(nn.Module):
    def __init__(self, args, path):
        super(MultimodalModel, self).__init__()
        self.text_model =  TextContrastiveModel(args)
        self.text_model.load_state_dict(torch.load(path+f'/models/contrastive/text_{args.contrastive_epochs}.pt'))
        self.image_model =  ImgContrastiveModel(args)
        self.image_model.load_state_dict(torch.load(path+f'/models/contrastive/image_{args.contrastive_epochs}.pt'))
        if args.classification_layer:
            self.classification_linear = get_classification_layer(args.hidden_size*2, args.class_num)
        else:
            self.classification_linear = nn.Linear(args.hidden_size*2, args.class_num)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_output = self.text_model(input_ids, attention_mask)
        image_output = self.image_model(pixel_values)
        multimodal_input = torch.cat([text_output, image_output], dim=-1)
        multimodal_output = self.classification_linear(multimodal_input)
        return multimodal_output



class MultimodalTestModel(nn.Module):
    def __init__(self, args, path):
        super(MultimodalTestModel, self).__init__()
        self.text_model =  TextContrastiveModel(args)
        self.image_model =  ImgContrastiveModel(args)
        if args.classification_layer:
            self.classification_linear = get_classification_layer(args.hidden_size*2, args.class_num)
        else:
            self.classification_linear = nn.Linear(args.hidden_size*2, args.class_num)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_output = self.text_model(input_ids, attention_mask)
        image_output = self.image_model(pixel_values)
        multimodal_input = torch.cat([text_output, image_output], dim=-1)
        multimodal_output = self.classification_linear(multimodal_input)
        return multimodal_output
