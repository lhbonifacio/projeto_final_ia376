import os
import re
import sys
import glob
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import pytorch_lightning as pl
from transformers import T5Tokenizer
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from model import ImageCaptioningOCR
from pytorch_lightning.callbacks import ModelCheckpoint


class SyntheticWordDataset(Dataset):
    """
    Class wrapper for Synthetic Word Dataset
    """
    def __init__(self, file_path, transforms = None, args=None):
            
        # Get the images and labels for each file in file_path
        self.images, self.labels = self.get_files(file_path)
        
        # Transformation to be applied to data
        self.transforms = transforms

        # Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_type)

        self.args = args


    def get_files(self, file_path):
        images = []
        labels = []

        with open(file_path, 'r') as reader:
            for line in tqdm(reader, desc="Loading Dataset"):
                line = line[1:]
                images.append(line.split(' ')[0])
                label = re.search(r'\_(.*)\_', line.split(' ')[0])
                labels.append(label.group(1).strip())

        return images, labels

    def __len__(self):      
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        
        # Get image and label according to index
        label = self.labels[idx]
        img = self.images[idx]
        
        try:
            img = np.array(Image.open('data/' + img).convert("RGB"))
        except:
            img = self.images[idx+1]
            img = np.array(Image.open('data/' + img).convert("RGB"))

        img = torch.from_numpy(img.transpose(2, 0, 1)) 

        # Normalize image between 0 and 1.
        image = (img - img.min()) / np.max([img.max() - img.min(), 1])
        
        tokenized_label = self.tokenizer.encode(
            label,
            padding='max_length',
            truncation=True,
            max_length=self.args.max_seq_length,
            return_tensors='pt')[0]

        # Applyt transformation, if required
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, label, tokenized_label


class SROIEDataset(Dataset):
    """
    Class wrapper for SROIE  Dataset
    """
    def __init__(self, files_path, args, transform=None):
        
        #get receipts images and its captions
        self.images = self.get_images(files_path)
        self.captions = self.get_captions(files_path)

        # Transformation to be applied to data
        self.transform = transform

        # Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_type)

        # Args
        self.args = args

        

    def get_images(self, path_to_images):
        return sorted(glob.glob('{}/*.jpg'.format(path_to_images)))

    def get_captions(self, path_to_labels):
        return [json.load(open(label, 'r')) \
        for label in sorted(glob.glob('{}/*.txt'.format(path_to_labels)))]

    def __getitem__(self, index):

        image = self.images[index]
        image = np.array(Image.open(image).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image)

        caption = self.captions[index]

        company_field = caption.get("company", "NA")
        address_field = caption.get("address", "NA")

        original_caption = 'company: {}, address: {}'.format(company_field, address_field)

        tokenized_caption = self.tokenizer.encode(
            original_caption,
            padding='max_length',
            truncation=True,
            max_length=self.args.max_seq_length,
            return_tensors='pt')[0]
        
        return image, original_caption, tokenized_caption

    def __len__(self):
        return len(self.captions)

    def close(self):
        self.data.close()

font = ImageFont.truetype("arial.ttf", 18)

class Wiki_Dataset(Dataset):
    def __init__(self, dataset, mode, train_size, test_size, transforms = None):
            
        # Data mode (train, test)
        self.mode = mode

        # Get text from Wikipedia
        self.dataset = self.get_dataset(dataset)
      
        # Transformation to be applied to data
        self.transforms = transforms

        self.train_size = train_size
        self.test_size = test_size

    def get_dataset(self, dataset):
        """As Wikipedia size is huge, it is necessary
        to control the amount of data to be used for training.
        """
        total = 0
        documents = []
        if self.mode == 'train':
            for item in tqdm(dataset['train'], desc="Reading dataset"):
                documents.append(re.sub(r'(\n)+', '', item['text']))
                total+=1
                if total == self.train_size:
                    break
        else:
            for item in tqdm(dataset['train'], desc="Reading dataset"):
                if total > 1000000:
                    documents.append(re.sub(r'(\n)+', '', item['text']))
                    if total == self.train_size+self.test_size:
                        break
                total+=1
        
        return documents

    def create_image(self, text, max_width, max_height):
        font = ImageFont.truetype("arial.ttf", 26)
        image = Image.new('RGB', (max_width, max_height), (255, 255, 255))  # White background.
        d = ImageDraw.Draw(image)
        short_words = ' '.join([w for w in text.split() if len(w)<=17])
        text_words = short_words.split(' ')
        target_words = ' '.join(text_words)
        img_words = ' \n'.join(text_words)
        d.text((10, 0), img_words, font=font, fill=(0, 0, 0)) 
      
        return np.array(image), target_words
   
    def __len__(self):      
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        label = " ".join(self.dataset[idx].split(' ')[:15])
        img, label = self.create_image(label, 300, 500)
        img = torch.from_numpy(img.transpose(2, 0, 1)) 

        # Normalize image between 0 and 1.
        image = (img - img.min()) / np.max([img.max() - img.min(), 1])
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, label

tokenizer = T5Tokenizer.from_pretrained('t5-small')

def collate_one(batch):
    """
    Tokenizing labels.
    """
    imgs = torch.as_tensor(torch.stack([sample[0] for sample in batch]))    
    labels = [sample[1] for sample in batch]

    encoded = tokenizer.batch_encode_plus(
        labels, 
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='pt')['input_ids']

    return imgs, labels, encoded

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help=".txt describing the path for training images")
    parser.add_argument("--val_file", default=None, type=str, required=True,
                        help=".txt describing the path for validation images")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help=".txt describing the path for test images")
    parser.add_argument("--model_type", default='t5-smal', type=str, required=True,
                        help="T5 type model")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--dataset", default='sroie', type=str, required=True,
                        help="Dataset for pre-training/training")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=False,
                        help="Path to pre-trained model")
    parser.add_argument("--batch_size", default=32, type=int, required=False,
                        help="Batch size per GPU/CPU.")
    parser.add_argument("--num_workers", default=4, type=int, required=False,
                        help="Workers for Dataloader")
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    args = parser.parse_args()
    
    if args.dataset == 'sroie':
        # SROIE ICDAR Dataset
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.Resize((256, 512)),
            transforms.ToTensor()])

        #Datasets for SROIE
        train_ds = SROIEDataset(
            'train',
            args=args,
            transform=transform)
    
        test_ds = SROIEDataset(
            'test', 
            args=args,
            transform=transform)

        #Dataloaders for SROIE
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers)

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers)

        # Callback to save the pretrained model
        checkpoint_callback = ModelCheckpoint(
            prefix="img2text-sroie-",
            filepath="{epoch}-{val_exact:.2f}",
            save_top_k=-1)  # -1 = Keeps all checkpoints, 1 = save best

    elif args.dataset == 'synthetic':
        # Synthetic Word Dataset
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.Resize((256, 512)),
            transforms.ToTensor()])

        # Datasets for Synthetic Word Dataset
        train_ds = SyntheticWordDataset(
            file_path = args.train_file,
            transforms=transform,
            args=args)

        val_ds = SyntheticWordDataset(
            file_path = args.val_file,
            transforms=transform,
            args=args)

        test_ds = SyntheticWordDataset(
            file_path = args.test_file,
            transforms=transform,
            args=args)

        # Dataloaders for Synthetic Word Dataset
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size, 
            num_workers=args.num_workers)

        val_loader = DataLoader(
            val_ds, 
            batch_size=args.batch_size,
            num_workers=args.num_workers)

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers)

        # Callback to save the pretrained model
        checkpoint_callback = ModelCheckpoint(
            prefix="img2text-synteticworddataset-",
            filepath="{epoch}-{val_exact:.2f}",
            save_top_k=-1)  # -1 = Keeps all checkpoints, 1 = save best
        
    elif args.dataset == 'wikipedia':
        # Wikipedia Synthetic Dataset
        dataset = load_dataset('wikipedia', '20200501.en')

        test_augmentations = transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize((512, 256)),
            transforms.ToTensor()])

        # Dataset for Wikipedia Synthetic
        train_ds = Wiki_Dataset(
            dataset, 
            'train', 
            test_augmentations)

        test_ds = Wiki_Dataset(
            dataset, 
            'test', 
            test_augmentations)

        # Dataloaders for Wikipedia Synthetic
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=collate_one)

        val_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=collate_one)

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=collate_one)

        # Callback to save the pretrained model
        checkpoint_callback = ModelCheckpoint(
            prefix="img2text-syntheticwiki-",
            filepath="{epoch}-{val_exact:.2f}",
            save_top_k=-1)  # -1 = Keeps all checkpoints, 1 = save best

    resume_from_checkpoint = None
    if os.path.exists(args.checkpoint_path):
        print(f'Restoring checkpoint: {checkpoint_path}')
        resume_from_checkpoint = checkpoint_path
    
    # PytorchLightning Trainer
    trainer = pl.Trainer(
        gpus=1,
        precision=32, 
        log_gpu_memory=True,
        max_epochs=2,
        val_check_interval=10000,
        check_val_every_n_epoch=1,
        profiler=True,
        callbacks=None,
        checkpoint_callback= checkpoint_callback,
        progress_bar_refresh_rate=100,
        resume_from_checkpoint=resume_from_checkpoint)
    
    # Model for training
    model = ImageCaptioningOCR(
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        test_dataloader=test_loader,
        dict_parameters=args)

    # Model training
    trainer.fit(model)

    # Model testing
    trainer.test(model)

if __name__ == '__main__':
    main()
