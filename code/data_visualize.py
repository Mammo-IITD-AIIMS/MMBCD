import os
import numpy as np 
import pandas as pd
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
from tqdm.auto import tqdm;
from PIL import Image
import copy
import cv2
import random
from torchvision import transforms
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class all_mammo():
    def __init__(self, csv_path, img_base, text_base, iou_threshold=0.1, topk=5, img_size=224, mask_ratio=0.2):
        self.img_base = img_base
        self.word_mask_ratio = mask_ratio
        self.text_base = text_base
        self.img_size = img_size
        self.image_path_list, self.text, self.label, self.both_label = self.csv_to_list(csv_path) 
        self.prompt_list = self.create_valid_prompt(self.text, self.label, self.both_label)

        self.box_text_path = self.generate_file_path(self.image_path_list)
        self.all_proposals = self.create_proposals(iou_threshold, topk)
        self.word_freq_list, self.word_list = self.get_tfidf_values(self.text)
        # import pdb; pdb.set_trace()
        self.words2mask = self.select_random_words()
        

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # Text -> 
        title = self.prompt_list[index]
        # title = self.update_report(title)        

        # Label -> 
        label = self.label[index]

        # Proposals -> 
        proposals = self.all_proposals[index]

        # Image Paths ->
        image_paths = self.image_path_list[index]

        # Crops -> 
        crops = self.create_crops(image_paths, proposals, self.img_size)
        
        # return torch.stack(crops), title, label   
        return torch.stack(crops), title, label, proposals, image_paths
    
    def get_tfidf_values(self, text_data):
        texts_c = []
        for text in text_data:
            if type(text) == float: 
                pass
            else:
                texts_c.append(text)

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts_c)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        top_words = tfidf_df.sum().sort_values(ascending=False).head(100)
        return top_words.to_list(), top_words.keys().to_list()
    
    def select_random_words(self):
        word_list, word_freq = self.word_list, self.word_freq_list
        total_freq = sum(word_freq)
        probabilities = [freq / total_freq for freq in word_freq]
        # probabilities = [1 / len(word_list) for freq in word_freq]
        sampled_words = random.choices(word_list, weights=probabilities, k=int(len(word_list)*self.word_mask_ratio))
        return sampled_words

    def update_report(self, report):
        words = re.findall(r'\S+|\s+', report)
        masked_words = []
        for word in words:
            if word not in self.words2mask: masked_words.append(word)
        new_report = " ".join(masked_words)
        return new_report
    
    def csv_to_list(self, csv_path):
        df = pd.read_csv(csv_path)

        return (df['im_path'].tolist(), df['text'].tolist(), df['cancer'].tolist(), df['all_views_cancer'].tolist())
        # return (df['im_path'].tolist()[:201], df['text'].tolist()[:201], df['cancer'].tolist()[:201], df['all_views_cancer'].tolist()[:201])

    def create_valid_prompt(self, text, label, both_label):
        lds = len(text)

        prompt = []
        for x in range(lds):
            if both_label[x]==1:
                if label[x]==1:
                    prompt.append(f'Indication: {text[x]}')
            elif both_label[x] == 0:
                prompt.append(f'Indication: {text[x]}')

            if both_label[x]==1: 
                if label[x]==0: 
                    prompt.append(f'Indication:')
            
        return prompt

    def generate_file_path(self, im_paths):
        text_paths = [(im_path.rstrip(".png")+"_preds.txt") for im_path in im_paths]

        return text_paths
    
    def create_proposals(self, iou_threshold, topk): 
        all_proposals = []

        for index, img_path in enumerate(tqdm(self.image_path_list, total=len(self.image_path_list), desc='generating proposals', position=0, leave=True)):
        # for index, img_path in enumerate(self.image_path_list):
            proposal_path = os.path.join(self.text_base, self.box_text_path[index])            
            img_path = os.path.join(self.img_base, img_path)            

            assert os.path.isfile(proposal_path)
            
            # for each im_path, get NMS boxes
            boxes = np.loadtxt(proposal_path, dtype=np.float32)
            proposals = self.non_max_suppression(boxes, iou_threshold=iou_threshold)

            # sample if number of proposals is less than topk
            if len(proposals) < topk:
                # Randomly sample from the existing proposals and append to the end
                additional_proposals = random.choices(proposals, k=topk - len(proposals))
                proposals = np.concatenate([proposals, additional_proposals])

            # select top k out of those boxes over connf_threshold
            proposals = proposals[:topk]
            all_proposals.append(proposals)

        return all_proposals
    
    def create_crops(self, img_path, proposals, img_size):
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # crop the images for the top k boxes 
        crop_lis = []
        img = Image.open(os.path.join(self.img_base, img_path)).convert('RGB')
        for j,box in enumerate(proposals):
            pascal_box = self.convert_yolo_pascal(box[:4], img)
            pil_crop = img.crop(pascal_box)
            # resizing image to 244x244
            pil_crop = transform(pil_crop)

            crop_lis.append(pil_crop)
        
        # return the crops
        return crop_lis
            
    def convert_yolo_pascal(self, box, image):
        W,H = image.size
        cx, cy, w, h = box
        x1 = int((cx-w/2)*W); x2 = int((cx+w/2)*W)
        y1 = int((cy-h/2)*H); y2 = int((cy+h/2)*H)
        bbox = [x1, y1, x2, y2]
        return bbox
    
    def non_max_suppression(self, boxes, iou_threshold):
        # import pdb; pdb.set_trace()
        boxes_copy = copy.deepcopy(boxes)
        while(True):
            selected_indices = []
            removed_indices = []
            for i in range(len(boxes_copy)):
                if i in selected_indices or i in removed_indices:
                   continue    
                current_box = boxes_copy[i]
                selected_indices.append(i)
                for j in range(i + 1, len(boxes_copy)):
                    if j in selected_indices or j in removed_indices:
                        continue                    
                    if self.calculate_iou(current_box, boxes_copy[j]) > iou_threshold:
                        removed_indices.append(j)
            
            selected_indices = sorted(selected_indices)
            # print(len(selected_indices),len(boxes_copy))
            if(len(selected_indices)==len(boxes_copy)):
                break
            else:
                boxes_copy = boxes_copy[selected_indices]
        return boxes_copy
                
    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1, c1 = box1
        x2, y2, w2, h2, c2 = box2
        
        # Convert to absolute coordinates
        x1 = x1 - w1/2; y1 = y1 - h1/2
        x2 = x2 - w2/2; y2 = y2 - h2/2
        
        # Calculate intersection coordinates
        x_intersection = max(x1, x2)
        y_intersection = max(y1, y2)
        w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
        h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)
        
        # Calculate intersection area
        intersection_area = w_intersection * h_intersection
        
        # Calculate areas of the bounding boxes
        area1 = w1 * h1
        area2 = w2 * h2
        
        iou = intersection_area / float(area1 + area2 - intersection_area)
        return iou

    def combine_and_save_images(self, all_cropped_images, output_path):
        to_pil_transform = transforms.ToPILImage()
        for index, cropped_images in enumerate(all_cropped_images):
            total_width = 0
            max_height = 0

            for img in cropped_images:
                img = to_pil_transform(img)
                total_width += img.width
                max_height = max(max_height, img.height)

            combined_image = Image.new("RGB", (total_width, max_height))

            current_width = 0
            for img in cropped_images:
                img = to_pil_transform(img)
                combined_image.paste(img, (current_width, 0))
                current_width += img.width

            combined_image.save(os.path.join(output_path, f'image_{index}.png'))

    def save_images_batch_wise(self, all_cropped_images, save_folder="focalnet_dino/output_check/combined_images"):
        batch_size, num_images, channels, height, width = all_cropped_images.size()

        os.makedirs(save_folder, exist_ok=True)

        for batch_idx in range(batch_size):
            batch_tensor = all_cropped_images[batch_idx]
            reshaped_tensor = batch_tensor.view(num_images, channels, height, width)
            image_array = reshaped_tensor.cpu().numpy().transpose(0, 2, 3, 1)
            stacked_image = Image.new('RGB', (width * num_images, height))

            for i in range(num_images):
                col = i
                image = Image.fromarray((image_array[i] * 255).astype('uint8'))
                stacked_image.paste(image, (col * width, 0))

            save_path = os.path.join(save_folder, f'batch_{batch_idx + 1}.jpg')
            stacked_image.save(save_path)

if __name__=='__main__':
    TEST_CSV = "focalnet_dino/text_resnet50/data/test_final.csv"
    IMG_BASE = "data/Test_Cropped"
    TEXT_BASE = "focalnet_dino/cropped_data/Test_focalnet"
    
    dataset = all_mammo(TEST_CSV, IMG_BASE, TEXT_BASE, topk=5)    
    print(len(dataset))

    batch_size = 3  
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    crops, titles, labels = next(iter(data_loader))

    # dataset.draw_boxes(image_paths=image_paths, bounding_boxes=proposals)
    # dataset.save_images_batch_wise(all_cropped_images=crops)
    print(crops.shape)
    print(labels)
    print(titles)
    print(len(crops))

    

