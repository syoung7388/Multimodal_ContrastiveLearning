
import argparse
import os 
import torch 
import pandas as pd
from make_dataset import get_classes, create_dataloader, create_test_dataloader
from transformers import AutoTokenizer, ViTFeatureExtractor, AdamW, get_linear_schedule_with_warmup
from models import TextContrastiveModel, ImgContrastiveModel, MultimodalModel
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from bs4 import BeautifulSoup
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default='0', type=str, help='gpus')
    parser.add_argument("--contrastive_epochs", default=3, type=int, help='contrastive_epochs')
    parser.add_argument("--multimodal_epochs", default=10, type=int, help='multimodal_epochs')
    parser.add_argument("--batch_size", default=64, type=int, help='batch_size')
    parser.add_argument("--hidden_size", default=768, type=int, help='batch_size')
    parser.add_argument("--seed", default=1234, type=int, help='seed')
    parser.add_argument("--max_len", default=256, type=int, help='max_len')
    parser.add_argument("--lr", default=1e-5, type=float, help='lr')
    parser.add_argument("--class_num", default=128, type=int, help='class_num')
    parser.add_argument("--text_model_name", default='klue/roberta-base', type=str, help='text_model_name')
    parser.add_argument("--image_model_name", default='google/vit-base-patch32-384', type=str, help='image_model_name')
    parser.add_argument("--ckpt", default='contrastive', type=str, help='ckpt')
    parser.add_argument("--temp", default=0.1, type=float, help='temp')
    parser.add_argument("--classification_layer", default=0, type=int, help='classification_layer')
    args =  parser.parse_args()
    return args


def contrastive_training():

    #train settings 
    total_steps = len(train_dataloader)*args.contrastive_epochs
    num_warmup_steps = int(total_steps*(1/3))
    print(f'[step] total: {total_steps}, warmup: {num_warmup_steps}')
    text_optimizer = AdamW(text_model.parameters(), lr = args.lr , betas = (0.9, 0.999), eps = 1e-8)
    text_scheduler = get_linear_schedule_with_warmup(text_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    img_optimizer = AdamW(image_model.parameters(), lr = args.lr , betas = (0.9, 0.999), eps = 1e-8)
    img_scheduler = get_linear_schedule_with_warmup(img_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)


    #train 
    text_model.train()
    image_model.train()
    contrastive_loss = {'text':[], 'image':[]}
    for e in range(1, args.contrastive_epochs+1):
        print(f"==========[{e}/{args.contrastive_epochs}]==========")
        e_text_contrastive_loss = 0.0 
        e_img_contrastive_loss = 0.0 
        with tqdm(total=len(train_dataloader), desc ='Training') as pbar:
            for data in train_dataloader:
                input_ids, attention_mask, pixel_values = data["input_ids"].to(device),  data["attention_mask"].to(device), data["pixel_values"].to(device)
                text_output = text_model(input_ids, attention_mask)
                img_output = image_model(pixel_values)
     
                text_contrastive_loss = get_contrastive_loss(img_output, text_output)
                text_optimizer.zero_grad()
                text_contrastive_loss.backward(retain_graph=True)
                text_optimizer.step()
                text_scheduler.step()

                e_text_contrastive_loss += text_contrastive_loss.item()
                img_contrastive_loss = get_contrastive_loss(text_output, img_output)
                img_optimizer.zero_grad()
                img_contrastive_loss.backward()
                img_optimizer.step()
                img_scheduler.step()
                e_img_contrastive_loss += img_contrastive_loss.item() 

                del input_ids
                del attention_mask
                del pixel_values               
                
                pbar.update(1)
        e_text_contrastive_loss /= len(train_dataloader)
        e_img_contrastive_loss /= len(train_dataloader)
        contrastive_loss['text'].append(e_text_contrastive_loss)
        contrastive_loss['image'].append(e_img_contrastive_loss)
        torch.save(text_model.module.state_dict(), path+f'/models/contrastive/text_{e}.pt')
        torch.save(image_model.module.state_dict(), path+f'/models/contrastive/image_{e}.pt')
        print(f'[train loss] text:{e_text_contrastive_loss:.5f}, image: {e_img_contrastive_loss:.5f}')
    
    contrastive_loss = pd.DataFrame(contrastive_loss)
    contrastive_loss.to_csv(path+'/contrastive_loss.csv')




def contrastive_testing(save_name):
    text_model.eval()
    image_model.eval()
    text_outputs = []
    img_outputs = []
    with torch.no_grad():
        with tqdm(total=len(test_dataloader), desc ='Testing') as pbar:
            for data in test_dataloader:  
                input_ids, attention_mask, pixel_values = data["input_ids"].to(device),  data["attention_mask"].to(device), data["pixel_values"].to(device)
                text_output = text_model(input_ids, attention_mask)
                img_output = image_model(pixel_values)
                text_outputs.append(text_output)
                img_outputs.append(img_output)
                pbar.update(1)
                del input_ids
                del attention_mask
                del pixel_values
    text_outputs = torch.concat(text_outputs, dim=0)
    img_outputs = torch.concat(img_outputs, dim=0)
    torch.save(text_outputs, path+f'/outputs/{save_name}_text.pt')
    torch.save(img_outputs, path+f'/outputs/{save_name}_image.pt')
    del text_outputs
    del img_outputs


def get_contrastive_loss(z1, z2):
    N = z1.size(0)
    z2_z1_sim = torch.exp(F.cosine_similarity(z2.unsqueeze(1), z1.unsqueeze(0), dim=2)/args.temp)
    z2_z2_sim = torch.exp(F.cosine_similarity(z2.unsqueeze(1), z2.unsqueeze(0), dim=2)/args.temp)
    mask = torch.eq(torch.eye(N), 0).to(z2_z2_sim.device)
    z2_z2_sim = z2_z2_sim.masked_select(mask).view(N, -1)
    pos = torch.diag(z2_z1_sim, 0)
    pos_neg = torch.sum(z2_z1_sim, dim=-1) + torch.sum(z2_z2_sim, dim = -1)
    loss = (-torch.log(pos/pos_neg)).mean()
    return loss


def multimodal_training():
    
    #train settings 
    total_steps = len(train_dataloader)*args.multimodal_epochs
    num_warmup_steps = int(total_steps*(1/3))
    print(f'[step] total: {total_steps}, warmup: {num_warmup_steps}')
    multi_optimizer = AdamW(multimodal_model.parameters(), lr = args.lr , betas = (0.9, 0.999), eps = 1e-8)
    multi_scheduler = get_linear_schedule_with_warmup(multi_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    cel_fnc = nn.CrossEntropyLoss()    

    #train 
    multimodal_model.train()
    multimodal_loss = {'loss':[]}
    for e in range(1, args.multimodal_epochs+1):
        print(f"==========[{e}/{args.multimodal_epochs}]==========")
        e_multi_loss = 0.0
        with tqdm(total=len(train_dataloader), desc ='Training') as pbar:
            for data in train_dataloader:
                
                input_ids, attention_mask, pixel_values = data["input_ids"].to(device),  data["attention_mask"].to(device), data["pixel_values"].to(device)
                label = data["label"].to(device)

                #text model train
                multi_optimizer.zero_grad()
                multi_output = multimodal_model(input_ids, attention_mask, pixel_values)
                multi_loss = cel_fnc(multi_output, label)
                multi_loss.backward()
                multi_optimizer.step()
                multi_scheduler.step()
                e_multi_loss += multi_loss.item()

                del input_ids
                del attention_mask
                del pixel_values
                pbar.update(1)
        
        e_multi_loss /= len(train_dataloader)       
        multimodal_loss['loss'].append(e_multi_loss)
        torch.save(multimodal_model.state_dict(), path+f'/models/multimodal/e={e}.pt')
        print(f'[train loss] {e_multi_loss:.5f}')

    multimodal_loss = pd.DataFrame(multimodal_loss)
    multimodal_loss.to_csv(path+'/multimodal_loss.csv')


def multimodal_testing():

    multimodal_model.eval()
    test_acc = 0.0
    with tqdm(total=len(test_dataloader), desc ='Testing') as pbar:
        for data in test_dataloader:  
            input_ids, attention_mask, pixel_values = data["input_ids"].to(device),  data["attention_mask"].to(device), data["pixel_values"].to(device)
            label = data["label"].to(device)

            #text model train
            outputs = multimodal_model(input_ids, attention_mask, pixel_values)
            preds = torch.argmax(outputs)
            test_acc += torch.sum(preds == label).item()
            pbar.update(1)
    test_acc = (test_acc/len(test_dataloader))*100
    print(f"[multimodal test] acc: {test_acc:.5f}")
    return test_acc   

def preprocess(s):
    #print(s)
    #print("[HTML]")
    s=BeautifulSoup(s, "html5lib").get_text()
    #print(s)
    
    #print("[URL]")
    s = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' ', s)
    #print(s)
    
    #print("[HASHTAG]")
    # s = re.sub('[#]+[0-9a-zA-Z_]+', ' ', s)
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
        os.makedirs(path+'/models/contrastive')
        os.makedirs(path+'/models/multimodal')
        os.makedirs(path+'/outputs')
    #data 
    tot_data = pd.read_csv('./datasets/train.csv')

    #tot_data = tot_data[:100]

    #class 
    classes = get_classes(tot_data['cat3'])
    print('[class]')
    print(len(classes))
    print(classes)

    #label change (from text to num)
    for i in range(len(tot_data)):
        tot_data['cat3'][i] = classes[tot_data['cat3'][i]]
        tot_data['overview'][i] = preprocess(tot_data['overview'][i])


    #data split
    train_num = int(len(tot_data)*0.8)
    train_data = tot_data[:train_num]
    test_data = tot_data[train_num:]
    print(f'[data num] tot: {len(tot_data)}, train: {len(train_data)}, test: {len(test_data)}')

    
    #contrastive model 
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.image_model_name)
    text_model = TextContrastiveModel(args)
    text_model = nn.DataParallel(text_model).to(device)

    image_model = ImgContrastiveModel(args).to(device)
    image_model = nn.DataParallel(image_model).to(device)

    #dataloader 
    train_dataloader = create_dataloader(train_data, tokenizer, feature_extractor, args)
    test_dataloader = create_test_dataloader(test_data,  tokenizer, feature_extractor, args)
    print(f'[dataloader] train: {len(train_dataloader)}, test: {len(test_dataloader)}')


    print("CONTRASTIVE") 
    contrastive_testing('scratch')
    contrastive_training()
    contrastive_testing('contrastive')

    del text_model
    del image_model
    

    print("MULTIMODAL") 
    multimodal_model = MultimodalModel(args, path)
    multimodal_model = nn.DataParallel(multimodal_model).to(device)
    multimodal_training()
    test_acc = multimodal_testing()


    info = open(path+'/info.txt', 'w')
    info.write(f'[test acc] {test_acc:.5f}%\n')
    info.write(f'cuda version: {torch.version.cuda}\n')
    info.write(f'torch version: {torch.__version__}\n')
    info.write(f'[dataset size] train: {len(train_data)}, test: {len(test_data)}\n')
    info.write(str(classes)+'\n')
    info.write(str(args)+'\n')






