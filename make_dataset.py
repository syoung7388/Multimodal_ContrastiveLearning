

from torch.utils.data import Dataset, DataLoader
import torch 
import cv2 
from PIL import Image
import requests

def get_classes(text_class):
    text_class = set(text_class)
    text_class = sorted(text_class)
    dict_class = {k:v for v, k in enumerate(text_class)}
    return dict_class

class CategoryDataset(Dataset):
    def __init__(self, text, img_path, labels,  tokenizer, feature_extractor, args):
        self.text = text
        self.img_path = img_path
        self.labels = labels
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor #vit feature extractor
        self.max_len = args.max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text =str(self.text[item])
        image_path = './datasets'+ str(self.img_path[item])[1:]
        image = cv2.imread(image_path)
        label = self.labels[item]
        tok_text = self.tokenizer(text=text,  max_length=self.max_len, return_token_type_ids=False, padding = 'max_length', truncation = True, return_attention_mask=True, return_tensors='pt')
        image_feature = self.feature_extractor(images= image, return_tensors='pt')
        return {
            'input_ids': tok_text['input_ids'].squeeze(0),
            'attention_mask': tok_text['attention_mask'].squeeze(0),
            'pixel_values': image_feature['pixel_values'][0],
            'label': torch.tensor(label, dtype = torch.long)
        }

def create_dataloader(train_data, tokenizer, feature_extractor, args):
    dataset = CategoryDataset(
        text = train_data.overview.to_numpy(),
        img_path = train_data.img_path.to_numpy(),
        labels= train_data.cat3.to_numpy(),
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        args = args
    )
    return DataLoader(dataset, batch_size = args.batch_size, shuffle=True)


def create_test_dataloader(test_data, tokenizer, feature_extractor, args):
    dataset = CategoryDataset(
        text = test_data.overview.to_numpy(),
        img_path = test_data.img_path.to_numpy(),
        labels = test_data.cat3.to_numpy(), 
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        args = args
    )

    return DataLoader(dataset, batch_size = 1, shuffle=False)