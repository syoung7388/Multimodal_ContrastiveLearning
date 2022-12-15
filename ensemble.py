
import argparse
import os 
import torch 
import pandas as pd
from make_dataset import get_classes, create_dataloader, create_test_dataloader
from transformers import AutoTokenizer, ViTFeatureExtractor, AdamW, get_linear_schedule_with_warmup
from models import TextModel, ImageModel
from tqdm import tqdm
import torch.nn as nn
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default='0', type=str, help='gpus')
    parser.add_argument("--epochs", default=10, type=int, help='epochs')
    parser.add_argument("--batch_size", default=64, type=int, help='batch_size')
    parser.add_argument("--max_len", default=256, type=int, help='max_len')
    parser.add_argument("--lr", default=1e-5, type=float, help='lr')
    parser.add_argument("--class_num", default=128, type=int, help='class_num')
    parser.add_argument("--text_model_name", default='klue/roberta-base', type=str, help='text_model_name')
    parser.add_argument("--image_model_name", default='google/vit-base-patch32-384', type=str, help='image_model_name')
    parser.add_argument("--ckpt", default='ensemble', type=str, help='ckpt')
    parser.add_argument("--seed", default=1234, type=int, help='seed')
    parser.add_argument("--ensemble", default=1, type=int, help='ensemble')
    parser.add_argument("--classification_layer", default=1, type=int, help='classification_layer')
    args =  parser.parse_args()
    return args

def preprocess(s):
    #print(s)
    #print("[HTML]")
    s=BeautifulSoup(s, "html5lib").get_text()
    #print(s)
    
    #print("[URL]")
    s = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' ', s)
    #print(s)
    
    #print("[HASHTAG]")
    #s = re.sub('[#]+[0-9a-zA-Z_]+', ' ', s)
    #print(s)
    
    #print("[특수 문자 제거]")
    #s = re.sub('[^0-9a-zA-Zㄱ-ㅎ가-힣]', ' ', s)
    #print(s)

    #print("[띄어쓰기 제거]")
    #s = s.replace('\n',' ')
    #print(s)


    return s

if __name__ == "__main__":


    #args
    args = parse_args()

    if args.ensemble:
        print("[ENSEMBLE]")
    else:
        print("[NOT ENSEMBLE]")


    #seed 
    if args.seed is not None:
        import random
        import numpy as np
        import torch.backends.cudnn as cudnn
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    #gpus 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #save path 
    path = f'./result/{args.ckpt}'
    if not os.path.exists(path):
        os.makedirs(path+'/models')

    #data 
    tot_data = pd.read_csv('./datasets/train.csv')

    #class 
    classes = get_classes(tot_data['cat3'])
    print('[class]')
    print(len(classes))
    print(classes)
   
    for i in range(len(tot_data)):
        tot_data['cat3'][i] = classes[tot_data['cat3'][i]]
        tot_data['overview'][i] = preprocess(tot_data['overview'][i])

    #tot_data = tot_data[:100]



    #data split
    train_num = int(len(tot_data)*0.8)
    train_data = tot_data[:train_num]
    test_data = tot_data[train_num:]
    print(test_data)
    print(f'[data num] tot: {len(tot_data)}, train: {len(train_data)}, test: {len(test_data)}')

    #model 
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.image_model_name)
    text_model = TextModel(args).to(device)
    image_model = ImageModel(args).to(device)
    #dataloader 
    train_dataloader = create_dataloader(train_data, tokenizer, feature_extractor, args)
    test_dataloader = create_test_dataloader(test_data,  tokenizer, feature_extractor, args)
    print(f'[dataloader] train: {len(train_dataloader)}, test: {len(test_dataloader)}')

    #train settings 
    total_steps = len(train_dataloader)*args.epochs
    num_warmup_steps = int(total_steps*(1/3))
    print(f'[step] total: {total_steps}, warmup: {num_warmup_steps}')
    text_optimizer = AdamW(text_model.parameters(), lr = args.lr , betas = (0.9, 0.999), eps = 1e-8)
    text_scheduler = get_linear_schedule_with_warmup(text_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    img_optimizer = AdamW(image_model.parameters(), lr = args.lr , betas = (0.9, 0.999), eps = 1e-8)
    img_scheduler = get_linear_schedule_with_warmup(img_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    cel_fnc = nn.CrossEntropyLoss()

    #train 
    text_model.train()
    image_model.train()
    result_loss = {'text':[], 'image':[]}
    for e in range(1, args.epochs+1):
        print(f"==========[{e}/{args.epochs}]==========")
        e_text_loss = 0.0
        e_img_loss = 0.0 
        with tqdm(total=len(train_dataloader), desc ='Training') as pbar:
            for data in train_dataloader:
                
                input_ids, attention_mask, pixel_values = data["input_ids"].to(device),  data["attention_mask"].to(device), data["pixel_values"].to(device)
                label = data["label"].to(device)

                #text model train
                text_optimizer.zero_grad()
                text_output = text_model(input_ids, attention_mask)
                text_loss = cel_fnc(text_output, label)
                text_loss.backward()
                text_optimizer.step()
                text_scheduler.step()
                e_text_loss += text_loss.item()

                #image model train
                img_optimizer.zero_grad()
                img_output = image_model(pixel_values)
                img_loss = cel_fnc(img_output, label)
                img_loss.backward()
                img_optimizer.step()
                img_scheduler.step()
                e_img_loss += img_loss.item()
                pbar.update(1)
        
        e_text_loss /= len(train_dataloader)
        e_img_loss /= len(train_dataloader)
        result_loss['text'].append(e_text_loss)
        result_loss['image'].append(e_img_loss)
        torch.save(text_model.state_dict(), path+f'/models/text_{e}.pt')
        torch.save(image_model.state_dict(), path+f'/models/image_{e}.pt')
        print(f'[train loss] text:{e_text_loss:.5f}, image: {e_img_loss:.5f}')


    result_loss = pd.DataFrame(result_loss)
    result_loss.to_csv(path+'/loss.csv')

    
    image_model.load_state_dict(torch.load("./result/ensemble_each_linearch/models/image_10.pt"))
    text_model.load_state_dict(torch.load("./result/ensemble_each_linearch/models/text_10.pt"))

    text_model.eval()
    image_model.eval()
    if args.ensemble:
        test_acc = 0.0
    else:
        text_acc = 0.0 
        img_acc = 0.0 
    with torch.no_grad():
        with tqdm(total=len(test_dataloader), desc ='Testing') as pbar:
            for data in test_dataloader:
                input_ids, attention_mask, pixel_values = data["input_ids"].to(device),  data["attention_mask"].to(device), data["pixel_values"].to(device)
                label = data["label"].to(device)

                text_output = text_model(input_ids, attention_mask)
                img_output = image_model(pixel_values)               
                if args.ensemble:
                    tot_output = (text_output+img_output)/2
                    preds = torch.argmax(tot_output)
                    test_acc += torch.sum(preds == label).item()
                else:                    
                    text_preds = torch.argmax(text_output)
                    text_acc += torch.sum(text_preds == label).item()
                    img_preds = torch.argmax(img_output)
                    img_acc += torch.sum(img_preds == label).item()
               
                pbar.update(1)
    if args.ensemble:
        test_acc = (test_acc/len(test_dataloader))*100
        print(f"[test] acc: {test_acc:.5f}")
    else:
        img_acc = (img_acc/len(test_dataloader))*100
        text_acc = (text_acc/len(test_dataloader))*100
        print(f"[test] text: {text_acc:.5f}, img: {img_acc:.5f}")
      


    info = open(path+'/info.txt', 'w')
    if args.ensemble:
        info.write(f'[test acc] {test_acc:.5f}%\n')
    else:
        info.write(f"[test acc] text: {text_acc:.5f}, img: {img_acc:.5f}\n")
    info.write(f'cuda version: {torch.version.cuda}\n')
    info.write(f'torch version: {torch.__version__}\n')
    info.write(f'[dataset size] train: {len(train_data)}, test: {len(test_data)}\n')
    info.write(str(args)+'\n')