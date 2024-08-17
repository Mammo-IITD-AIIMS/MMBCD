# Importing the Libraries:
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import numpy as np

from args import get_args
from model import MMBCD
from data import all_mammo
from test import test_code, load_model_again
from transformers import RobertaTokenizer

from torch.utils.data.sampler import WeightedRandomSampler

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

seed_value = 42
np.random.seed(seed_value)

def make_weights(train_targets, prob_malignant):
    class_sample_counts = [train_targets.count(class_idx) for class_idx in range(2)]
    weights = np.array(class_sample_counts, dtype=np.float32)

    class_weights = [1-prob_malignant, prob_malignant] / weights
    # import pdb; pdb.set_trace()
    train_targets = np.array(train_targets, dtype=np.float64)
    train_targets[np.where(np.array(train_targets)==0)[0]] = class_weights[0]
    train_targets[np.where(np.array(train_targets)==1)[0]] = class_weights[1]
    # Define sampler for weighted sampling on the training set
    train_targets = torch.tensor(train_targets) 

    return train_targets

def load_data(CSV, IMG_BASE, TEXT_BASE, prob_malignant=0.5, type=1, workers=8, batch_size=32, topk=5, img_size=224):
    # 1 for train 0 for test
     
    if type == 1:
        dataset = all_mammo(CSV, IMG_BASE, TEXT_BASE, topk=topk, img_size=img_size, mask_ratio=0.2, enable_mask=True)
        print(f'Malignancy Count: {(sum(dataset.label) / len(dataset.label)) * 100 if dataset.label else 0}')
        train_targets = dataset.label
        train_targets = make_weights(train_targets, prob_malignant)
        train_targets = train_targets.to("cuda")
        sampler = WeightedRandomSampler(train_targets, train_targets.shape[0]*2, replacement=True)
        print("Made Train Dataloader")
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, drop_last=True) 
    else: 
        dataset = all_mammo(CSV, IMG_BASE, TEXT_BASE, topk=topk, img_size=img_size, enable_mask=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, drop_last=True) 
        print("Made Test Dataloader")

    return dataset, dataloader

def load_model(checkpoint_path_vit, vit_layers_freeze, r50_img_size, rob_checkpoint_path, rob_layers_unfreeze):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    # model = R50_RoBERTa(checkpoint_path_vit, vit_layers_freeze, r50_img_size, rob_checkpoint_path, rob_layers_unfreeze)
    model = MMBCD(checkpoint_path_vit, vit_layers_freeze, r50_img_size, rob_checkpoint_path, rob_layers_unfreeze)
    # model = VIT_RoBERTa(checkpoint_path_vit, vit_layers_freeze, r50_img_size, rob_checkpoint_path, rob_layers_unfreeze)
    model.to(device)

    model = torch.nn.DataParallel(model)
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    return model, tokenizer

def train_code(model, train_dataloader, val_dataloader,train_dataset, file_path, checkpoint_path, plot_path, tokenizer, num_epochs=50, learning_rate=5e-3):
    file = open(file_path, "w")
    file.close()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9,0.98),eps=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 if epoch == 19 else 1)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_max = sys.maxsize
    loss_list_val = []
    loss_list_train = []
    exit_cnt = 0

    for epoch in range(num_epochs):
        train_dataset.select_random_words()
        file = open(file_path, "a")
        print(f'Started Epoch #{epoch+1}')

        pbar_train = tqdm(train_dataloader, total=len(train_dataloader), desc='train', position=0, leave=True)

        model.train()
        avg_loss_train = 0
        batch_num_train = 0
        for batch in pbar_train:
            optimizer.zero_grad()
            crops, texts, labels = batch 

            texts = tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt', max_length=90)
            inputids = texts['input_ids']
            attmask = texts['attention_mask']

            # import pdb; pdb.set_trace()
            crops = crops.to(device)
            labels = labels.to(device)
            inputids = inputids.to(device)
            attmask = attmask.to(device)

            logits, _ = model(crops, inputids, attmask)
            loss = loss_criterion(logits, labels)

            avg_loss_train += loss.item()
            batch_num_train += 1

            loss.backward()
            optimizer.step()

            pbar_train.set_description(f"\tEpoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        avg_loss_train /= batch_num_train 
        tqdm.write(f'Epoch {epoch+1}: Average loss TRAIN = {avg_loss_train:.4f}')
        file.write(f'Epoch {epoch+1}: Average loss TRAIN = {avg_loss_train:.4f}\n')

        # Validation
        model.eval()
        pbar_test = tqdm(val_dataloader, total=len(val_dataloader), desc='val', position=0, leave=True)
        avg_loss_val = 0
        batch_num_val = 0
        with torch.no_grad():
            for batch in pbar_test:
                crops, texts, labels = batch 

                texts = tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt', max_length=90)
                inputids = texts['input_ids']
                attmask = texts['attention_mask']

                crops = crops.to(device)
                labels = labels.to(device)
                inputids = inputids.to(device)
                attmask = attmask.to(device)

                logits, _ = model(crops, inputids, attmask)
                loss = loss_criterion(logits, labels)

                avg_loss_val += loss.item()
                batch_num_val += 1
                pbar_test.set_description(f"\tEpoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
            avg_loss_val /= batch_num_val

        tqdm.write(f'Epoch {epoch+1}: Average loss VAL = {avg_loss_val:.4f}')
        file.write(f'Epoch {epoch+1}: Average loss VAL = {avg_loss_val:.4f}\n')
        # scheduler.step()
        print(f'Epoch {epoch + 1}: Learning Rate: {optimizer.param_groups[0]["lr"]}')

        loss_list_val.append(avg_loss_val)
        loss_list_train.append(avg_loss_train) 
        plt.plot([num+1 for num in range(len(loss_list_val))], loss_list_val, label = "VAL_LOSS")
        plt.plot([num+1 for num in range(len(loss_list_val))], loss_list_train, label = "TRAIN_LOSS")
        plt.xlabel('Epoch #')
        plt.ylabel('Validation & Train Loss')
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()
        tqdm.write('\n\n')

        # scheduler.step(avg_loss_val)

        if avg_loss_val < val_max:
            val_max = avg_loss_val
            exit_cnt = 0
            torch.save(model.state_dict(), checkpoint_path)

            tqdm.write(f'\tEpoch #{epoch+1} - Model checkpoint saved.')
            file.write(f'\tEpoch #{epoch+1} - Model checkpoint saved.\n')
        else: 
            exit_cnt += 1
            if exit_cnt >= 50:
                tqdm.write(f'\Exiting training loop due to overfitting.')
                file.write(f'\Exiting training loop due to overfitting.\n')
                break

        file.close()
    
    best_lr_used = optimizer.param_groups[0]['lr']
    print(f'Best Learning Rate Used: {best_lr_used}')


if __name__=="__main__":
    TRAIN_CSV = "focalnet_dino/clip/r50_rob_multi_label/data/train_correct.csv"
    TRAIN_IMG_BASE = "data/Train_Cropped"
    TRAIN_TEXT_BASE = "focalnet_dino/cropped_data/Train_focalnet"

    EVAL_CSV = "focalnet_dino/clip/r50_rob_multi_label/data/test_correct.csv"
    EVAL_IMG_BASE = "data/Test_Cropped"
    EVAL_TEXT_BASE = "focalnet_dino/cropped_data/Test_focalnet"

    checkpoint_path_vit = None
    rob_checkpoint_path = None


    args = get_args()

    os.makedirs(args.checkpoint_model_save, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_model_save + "model_best.pt")
    file_path = os.path.join(args.checkpoint_model_save + "training_stats.txt")
    plot_path = os.path.join(args.checkpoint_model_save + "loss_plot.png")
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    # DataLoader Params
    topk = args.topk
    num_workers = args.num_workers
    batch_size = args.batch_size
    prob_malignant = 0.3

    # Model Params
    vit_layers_freeze = args.vit_layers_freeze
    r50_img_size = args.img_size
    rob_layers_unfreeze = args.rob_layers_unfreeze

    print(f'topk={topk}\nnum_workers={num_workers}\nbatch_size={batch_size}\n50_layers_freeze={vit_layers_freeze}\nrob_layers_unfreeze={rob_layers_unfreeze}\nimagesize={r50_img_size}')

    # import pdb; pdb.set_trace()

    model, tokenizer = load_model(checkpoint_path_vit, vit_layers_freeze, r50_img_size, rob_checkpoint_path, rob_layers_unfreeze)
    print("Loading training DataLoader: ")
    train_dataset, train_dataloader = load_data(TRAIN_CSV, TRAIN_IMG_BASE, TRAIN_TEXT_BASE, prob_malignant, 0, num_workers, batch_size, topk, r50_img_size)  
    print("Loading validation DataLoader: ")
    val_dataset, val_dataloader = load_data(EVAL_CSV, EVAL_IMG_BASE, EVAL_TEXT_BASE, prob_malignant, 0, num_workers, batch_size, topk, r50_img_size)
    val_dataset.word_mask_ratio = 0
    print("Now training: \n\n")
    train_code(model, train_dataloader, val_dataloader, train_dataset, file_path, checkpoint_path, plot_path, tokenizer, num_epochs=num_epochs, learning_rate=learning_rate)

    checkpoint_path_test = os.path.join(args.checkpoint_model_save + "model_best.pt")
    plot_path_test = os.path.join(args.checkpoint_model_save + "result_auc_plot.png")
    score_file = os.path.join(args.checkpoint_model_save + "result_scores.txt")

    test_model = load_model_again(checkpoint_path_test, checkpoint_path_vit, vit_layers_freeze, r50_img_size, rob_checkpoint_path, rob_layers_unfreeze)
    test_code(test_model, tokenizer, val_dataloader, plot_path_test, score_file)


