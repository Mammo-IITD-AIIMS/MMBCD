import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import MMBCD
from transformers import RobertaTokenizer
import shutil
import numpy as np
import os
from data import all_mammo
    
def load_data(CSV, IMG_BASE, TEXT_BASE, workers=8, batch_size=32, topk=5, img_size=224):
    dataset = all_mammo(CSV, IMG_BASE, TEXT_BASE, topk=topk, img_size=img_size, mask_ratio=0, enable_mask=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers) 

    return dataloader

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def load_model_again(checkpoint_path, checkpoint_path_r50, r50_layers_freeze, r50_img_size, rob_checkpoint_path, rob_layers_unfreeze):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model = MMBCD(checkpoint_path_r50, r50_layers_freeze, r50_img_size, rob_checkpoint_path, rob_layers_unfreeze)
    model.load_state_dict(remove_module_prefix(torch.load(checkpoint_path)))
    model.to(device)

    model = torch.nn.DataParallel(model)
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    return model, tokenizer


def recall2FPR(logits, labels, fpr_value = 0.3):
    neg_idx = np.where(np.array(labels)==0)[0]
    neg_logits  = sorted([logits[idx] for idx in neg_idx], reverse=True)
    thresh =  neg_logits[int(fpr_value*len(neg_logits))]
    logits = np.array(logits)
    preds = np.zeros(len(labels))
    preds[np.where(logits>=thresh)[0]] = 1
    
    true_positives = sum(1 for pred, gold in zip(preds, labels) if pred == 1 and gold == 1)
    actual_positives = sum(labels)
    recall = true_positives / actual_positives if actual_positives != 0 else 0
    
    # Find indices of false negatives and true positives
    false_negatives_indices = [i for i, (pred, gold) in enumerate(zip(preds, labels)) if pred == 0 and gold == 1]
    true_positives_indices = [i for i, (pred, gold) in enumerate(zip(preds, labels)) if pred == 1 and gold == 1]
    print(recall, len(false_negatives_indices), len(true_positives_indices))
    return recall, false_negatives_indices, true_positives_indices

def save_images_recall2fpr(images, fn_idx, tp_idx):
    fn_folder = "fn"; os.makedirs(fn_folder, exist_ok=True)
    tp_folder = "tp"; os.makedirs(tp_folder, exist_ok=True)
    for i,img_path in enumerate(images):
        image_path = os.path.join(TEST_IMG_BASE, img_path)
        if(i in fn_idx):
            trgt_path = os.path.join(fn_folder, image_path.split("/")[-1])
            shutil.copy(image_path, trgt_path)
        if(i in tp_idx):
            trgt_path = os.path.join(tp_folder, image_path.split("/")[-1])
            shutil.copy(image_path, trgt_path)
     
     
        
def fpr_r1(y_true, y_logits, im_paths):
    y_true = np.array(y_true)
    y_logits = np.array(y_logits)
    true_positive_indices = np.where(y_true == 1)[0]
    logits_true_positives = [y_logits[i] for i in true_positive_indices]
    min_logit = np.min(logits_true_positives)
    predicted_positives = np.where(y_logits > min_logit)[0]
    final_indices = np.setdiff1d(predicted_positives, true_positive_indices)

    fpr = len(final_indices) / len(np.where(y_true == 0)[0])
    fp_imgs = [im_paths[i] for i in final_indices]
    print("Total FP images @ Recall=1", len(fp_imgs))
    np.save("fp_images", fp_imgs)
    # return fpr, final_indices
    

def test_code(model, tokenizer, test_dataloader, plot_path, file_path):
    file = open(file_path, "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    predictions = []
    true_labels = []
    prob_val = []
    # images= []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            crops, texts, labels = batch 
            # images.extend(img_path)
            texts = tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt', max_length=90)
            inputids = texts['input_ids']
            attmask = texts['attention_mask']

            crops = crops.to(device)
            labels = labels.to(device)
            inputids = inputids.to(device)
            attmask = attmask.to(device)

            logits, _ = model(crops, inputids, attmask)
            # import pdb; pdb.set_trace()
            probabilities = F.softmax(logits, dim=-1)
            pred = probabilities.max(1, keepdim=True)[1]
            greater_prob = [x[1] for x in probabilities.tolist()]
            
            predictions.extend([x[0] for x in pred.tolist()])
            true_labels.extend(labels.tolist())
            
            prob_val.extend(greater_prob)
        
        recall, fn_idx, tp_ipx = recall2FPR(prob_val, true_labels)
        # fpr_r1(true_labels, prob_val, images)
        # save_images_recall2fpr(images, fn_idx, tp_ipx)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        print(f'Accuracy: {accuracy:.2f}')
        print(f'F1 Score: {f1:.2f}')
        file.write(f'Accuracy: {accuracy:.2f}\n')
        file.write(f'F1 Score: {f1:.2f}\n')

        print(classification_report(true_labels, predictions, labels=[0, 1]))
        file.write(classification_report(true_labels, predictions, labels=[0, 1]))

        auc_score = roc_auc_score(true_labels, prob_val)
        print('Logistic: ROC AUC=%.3f' % (auc_score))
        file.write('Logistic: ROC AUC=%.3f\n' % (auc_score))
        # calculate roc curves
        lr_fpr, lr_tpr, _ = roc_curve(true_labels, prob_val) 
        plt.plot(lr_fpr, lr_tpr, marker='.', label='text')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(plot_path)
        plt.show()

if __name__=="__main__":
    TEST_CSV = "inhouse2_DATA/inhouse2_data.csv"
    TEST_IMG_BASE = "inhouse2_DATA/Mammo_PNG"
    TEST_TEXT_BASE = "inhouse2_DATA/Mammo_PNG_focalnet"

    plot_path = './models/mmbcd/result_auc_plot_inhouse.png'
    score_file = './models/mmbcd/result_scores_inhouse.txt'

    checkpoint_path = "./models/mmbcd/model_best.pt"
    num_workers = 8
    batch_size = 32
    topk = 8
    img_size = 224

    layers_freeze = 2

    print(f'topk = {topk}\nnum_workers = {num_workers}\nbatch_size = {batch_size}\nimage = {img_size}\nlayers_freeze = {layers_freeze}')

    # model = load_model_again(checkpoint_path, layers_freeze, img_size)
    model, tokenizer = load_model_again(checkpoint_path, None, 0, img_size, None, 0)
    print("Loading validation DataLoader: ")
    val_dataloader = load_data(TEST_CSV, TEST_IMG_BASE, TEST_TEXT_BASE, num_workers, batch_size, topk, img_size)    

    print("Now Testing: ")
    # test_code(model, val_dataloader, plot_path, score_file)
    test_code(model, tokenizer, val_dataloader, plot_path, score_file)

    
