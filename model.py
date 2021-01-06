import torch
import random
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
from transformers import T5Model, T5Tokenizer, T5ForConditionalGeneration
from torch import nn
import numpy as np

from metrics import *

class ImageCaptioningOCR(pl.LightningModule):

    def __init__(self, train_dataloader, val_dataloader, test_dataloader, dict_parameters):
        super(ImageCaptioningOCR, self).__init__()
        self.dict_parameters = dict_parameters

        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')

        self.decoder = T5ForConditionalGeneration.from_pretrained(self.dict_parameters.model_type)

        self.features2T5 = nn.Conv2d(in_channels=1280, out_channels=self.decoder.config.d_model, kernel_size=1)
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.dict_parameters.model_type)


        # EfficientNet parameters will not be freeze
        '''
        for param in self.encoder.parameters():
            param.requires_grad = False
        '''
        
        self._train_dataloader = train_dataloader
        self._val_dataloader   = val_dataloader
        self._test_dataloader  = test_dataloader

    def extract_features(self, images):
        #images [batch_size, channels, height, weight]

        #features = model.encoder.extract_endpoints(images)["reduction_4"]
        features = self.encoder.extract_features(images)

        #features [batch_size, 112, 8, 8]
        features = self.features2T5(features)

        #features [batch_size, 512, 8, 8]
        features = features \
            .permute(0, 2, 3, 1) \
            .reshape(features.shape[0], -1, self.decoder.config.d_model)

        #features [batch_size, 64, 512]
        return features

    
    def generate_text(self, features):

        decoded_ids = torch.full(
            (features.shape[0], 1),
            self.decoder.config.decoder_start_token_id,
            dtype=torch.long).to(features.device)

        encoder_hidden_states = self.decoder.get_encoder()(inputs_embeds=features)

        for step in range(self.dict_parameters.max_seq_length-1):
            logits = self.decoder(
                decoder_input_ids=decoded_ids,
                encoder_outputs=encoder_hidden_states)[0]
            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
            decoded_ids = torch.cat([decoded_ids, next_token_id], dim=-1)

            if torch.eq(next_token_id[:, -1], self.tokenizer.eos_token_id).all():
                break

        return decoded_ids

    def forward(self, batch):
        images, labels, tokens = batch 
        features = self.extract_features(images)

        if self.training:
            loss = self.decoder(
                inputs_embeds=features,
                decoder_input_ids=None, 
                labels=tokens,
                return_dict=True).loss
            return loss
        else:
            return self.generate_text(features)

    def training_step(self, batch, batch_idx): 
        loss = self(batch)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        with open('training_loss.txt', 'a') as output_file:
            output_file.write(str(loss.item()) + '\n')
        return loss

    def validation_step(self, batch, batch_idx):
        pred_tokens = self(batch)
        decoded_pred = [self.tokenizer.decode(
            tokens, 
            clean_up_tokenization_spaces=True) for tokens in pred_tokens]
        return {"Predicted": decoded_pred, "True": batch[1]}

    def test_step(self, batch, batch_idx):
        pred_tokens = self(batch)
        # Tokens -> String
        decoded_pred = [self.tokenizer.decode(tokens) for tokens in pred_tokens]
        return {"Predicted": decoded_pred, "True": batch[1]}

    def validation_epoch_end(self, outputs):
        trues = sum([list(x['True']) for x in outputs], [])
        preds = sum([list(x['Predicted']) for x in outputs], [])

        n = random.choice(range(len(trues)))
        print(f"\nSample Target: {trues[n]}\nPrediction: {preds[n]}\n")

        f1 = []
        exact = []
        for true, pred in zip(trues, preds):
            f1_score = compute_f1(a_gold=true, a_pred=pred)
            em       = compute_exact(a_gold=true, a_pred=pred)
            f1.append(f1_score)
            exact.append(em)
        f1 = np.mean(f1)
        exact = np.mean(exact)

        self.log("val_f1", f1, prog_bar=True)
        self.log("val_exact", exact, prog_bar=True)

    def test_epoch_end(self, outputs):

        trues = sum([list(x['True']) for x in outputs], [])
        preds = sum([list(x['Predicted']) for x in outputs], [])

        n = random.choice(range(len(trues)))
        print(f"\nSample Target: {trues[n]}\nPrediction: {preds[n]}\n")

        f1 = []
        exact = []
        for true, pred in zip(trues, preds):
            f1_score = compute_f1(a_gold=true, a_pred=pred)
            em       = compute_exact(a_gold=true, a_pred=pred)
            f1.append(f1_score)
            exact.append(em)

        f1 = np.mean(f1)
        exact = np.mean(exact)

        self.log("test_f1", f1, prog_bar=True)
        self.log("test_exact", exact, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.dict_parameters.learning_rate, eps=1e-08)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader