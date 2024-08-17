import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification

class vit_dino(nn.Module):
    def __init__(self, layers=11, img_size=224):
        super(vit_dino, self).__init__()
        self.img_size = img_size
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')        
                        
        in_features = 384
        self.backbone = model
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 2),
        )


    def forward(self, input_tensor):
        x = input_tensor.view(-1, 3, self.img_size, self.img_size)
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)

        individual_tensors = features.view(input_tensor.shape[0], input_tensor.shape[1], -1)
        output_tensor, _ = torch.max(individual_tensors, dim=1)
        output_tensor = self.mlp(output_tensor)

        return output_tensor

class MMBCD(nn.Module):
    def __init__(self, checkpoint_path_vit, vit_layers_freeze, vit_img_size, rob_checkpoint_path, rob_layers_unfreeze):
        super(MMBCD, self).__init__()

        self.img_size = vit_img_size

        ## Loading image model 
        model_image = vit_dino(vit_layers_freeze, vit_img_size)
        if(checkpoint_path_vit!=None):
            print("Loading Pretrained model")
            state_dict = torch.load(checkpoint_path_vit)

            state_dict = self.remove_module_prefix(state_dict)
            state_dict = self.change_clip_prefix(state_dict)
            missing_keys_pretrained, missing_keys_new_model = self.load_common_weights(state_dict, model_image)

        self.image_encoder = model_image.backbone
        # import pdb; pdb.set_trace()
        for i,child in enumerate(self.image_encoder.children()):
            if(i<2):
                for param in child.parameters():
                    param.requires_grad = False
            if(i==2):
                for j,child2 in enumerate(child.children()):
                    if(j<vit_layers_freeze):
                        for param in child2.parameters():
                            param.requires_grad = False

        self.img_fc1 = nn.Linear(384, 256)
        self.img_fc_layer = nn.Sequential(
            nn.BatchNorm1d(384),
            self.img_fc1, 
            nn.GELU()
        )
        
        for i, child in enumerate(self.image_encoder.children()):
            if(i==2):
                for gchild in child.children():
                    for param in gchild.parameters():
                        print(i, param.requires_grad)

        print("----------------------- Loaded VIT successfully -----------------------")



        ## Loading text model
        model_name = 'roberta-base'

        model_text = RobertaForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
        if(rob_checkpoint_path!=None):
            print("Loading Pretrained model")
            state_dict = self.remove_module_prefix_text(torch.load(rob_checkpoint_path))
            missing_keys_pretrained, missing_keys_new_model = self.load_common_weights(state_dict, model_text)
        self.text_encoder = model_text

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.roberta.encoder.layer[-rob_layers_unfreeze:].parameters():
            param.requires_grad = True

        self.txt_fc1 = nn.Linear(768, 256)
        self.txt_fc_layer = nn.Sequential(
            nn.BatchNorm1d(768),
            self.txt_fc1, 
            nn.GELU()
        )

        print("----------------------- Loaded RoBERTa successfully -----------------------")

        in_features = 256
        self.attention = nn.MultiheadAttention(embed_dim=in_features, num_heads=1, batch_first = True, dropout=0.3)
        self.model_fc2 = nn.Linear(in_features*3, 2)
        
    def forward(self, image_tensor, inputids, attmask):
        x = image_tensor.view(-1, 3, self.img_size, self.img_size)
        features = self.image_encoder(x)
        features = features.squeeze(-1).squeeze(-1)

        image_embeddings = self.img_fc_layer(features)
        image_embeddings = image_embeddings.view(image_tensor.shape[0], image_tensor.shape[1], -1)
        maxpool_img_embedd, _ = torch.max(image_embeddings, dim=1)
        tokenized_sentences = {'input_ids': inputids, 'attention_mask': attmask}
        text_embedd = self.text_encoder(**tokenized_sentences)

        sentence_embeddings = text_embedd.hidden_states[-1][:,0,:]
        text_embeddings = self.txt_fc_layer(sentence_embeddings)

        attn_features, attn_weights = self.attention(text_embeddings.unsqueeze(1), image_embeddings, image_embeddings)        
        embeddings_org = torch.cat((attn_features.squeeze(1), text_embeddings, maxpool_img_embedd), dim=1)
        embeddings = self.model_fc2(embeddings_org.squeeze(1))

        return embeddings, embeddings_org
    
    def remove_module_prefix(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove the 'module.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key]
        
        return new_state_dict

    def change_clip_prefix(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('image_encoder.'):
                new_key = key.replace("image_encoder", "backbone")  # Remove the 'module.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    def load_common_weights(self, pretrained_state_dict, new_model):
        common_keys = set(pretrained_state_dict.keys()) & set(new_model.state_dict().keys())
        missing_keys_pretrained = set(new_model.state_dict().keys()) - common_keys
        missing_keys_new_model = set(pretrained_state_dict.keys()) - common_keys

        for key in common_keys:
            new_model.state_dict()[key].copy_(pretrained_state_dict[key])

        return missing_keys_pretrained, missing_keys_new_model

    def remove_module_prefix_text(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.text_encoder.'):
                new_key = key.replace("module.text_encoder.", "")   # Remove the 'module.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        return new_state_dict




if __name__=='__main__':
    device = 'cuda'

    resnet_pretrained_path = None
    roberta_pretrained_path = None

    model = MMBCD(resnet_pretrained_path, 9, 224, roberta_pretrained_path, 2)
    model.to('cuda')
    # model()

    images = torch.rand((2, 5, 3, 224, 224))
    from transformers import RobertaTokenizer
    texts = ['Hello', 'Hows it going']

    model_name = 'roberta-base' 
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    texts = tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt', max_length=90)
    inputids = texts['input_ids']
    attmask = texts['attention_mask']

    images = images.to(device)
    inputids = inputids.to(device)
    attmask = attmask.to(device)

    logits = model(images, inputids, attmask)
    print(logits)
    

