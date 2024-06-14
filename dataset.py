from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset, Features
from transformers import GPT2Tokenizer, GPT2TokenizerFast
import torch
from tqdm import tqdm
import pickle
from tokenizer import TiktokenTokenizer
from configs import get_configs
from gpt import GPTActor

from tokenizer import TiktokenTokenizer
from datasets import load_dataset
from evaluate import generate_gpt2

from sentence_transformers import SentenceTransformer
import torch
import tqdm

class Emb_Model:
    '''Embedding model, to be loaded and used for finding similar embeddings in the prompts.'''

    def __init__(self, model_name='sentence-transformers/all-distilroberta-v1', sim_threshold=0.8, dis_threshold=0.2):
        self.model = SentenceTransformer(model_name)
        self.sim_threshold = sim_threshold
        self.dis_threshold = dis_threshold

    def similar_dissimilar(self, col1, col2, col3):
        embs1 = self.model.encode(col1, convert_to_tensor=True)
        embs2 = self.model.encode(col2, convert_to_tensor=True)
        embs3 = self.model.encode(col3, convert_to_tensor=True)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        similarity1 = cos(embs1, embs2)
        similarity2 = cos(embs1, embs3)
        condition1  = torch.logical_or(similarity1 > self.sim_threshold, similarity1 < self.dis_threshold)
        condition2 = torch.logical_or(similarity2 > self.sim_threshold, similarity2 < self.dis_threshold)

        return torch.unique(torch.concat((torch.nonzero(condition1), torch.nonzero(condition2)))), similarity1, similarity2
        
        
class DahoasRMStaticDataset_Filtered(Dataset):
    """
    https://huggingface.co/datasets/Dahoas/rm-static
    """

    def __init__(self,
                 block_size,
                 current_model_path,
                 embedding_model_name="AnnaWegmann/Style-Embedding",
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Dahoas/rm-static", split=split)
        self.pairs = []
        self.masks = []
        #embeddings' check
        embedding_model = Emb_Model(embedding_model_name, 0.5, 0.5)

        cfg = get_configs('gpt2-medium')
        current_generation_model = GPTActor.from_checkpoint(cfg, current_model_path)

        tokenizer = TiktokenTokenizer('gpt2')


        cnt = 0
        print(f"Filtering DahoasRMStaticDataset {split} split")

        for data in dataset:
            prompt = data['prompt']

            generation = generate_gpt2(current_generation_model, prompt, 'cuda')
            
            

            _, pos_sem, neg_sem = embedding_model.similar_dissimilar([generation], [data['chosen']], [data['rejected']])

            if pos_sem >= 0.8 or neg_sem >= 0.8:
                cnt += 1

                positive_text = prompt + data['chosen'] + "<|endoftext|>"
                positive = tokenizer(positive_text,
                                    max_length=block_size,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt")

                negative_text = prompt + data['rejected'] + "<|endoftext|>"
                negative = tokenizer(negative_text,
                                    max_length=block_size,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt")

                self.pairs.append(
                    torch.stack((positive['input_ids'], negative['input_ids']),
                                dim=0))

                self.masks.append(
                    torch.stack(
                        (positive['attention_mask'], negative['attention_mask']),
                        dim=0))
                if max_examples and cnt >= max_examples:
                    break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("Dahoas/rm-static", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["prompt"] + data["chosen"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)

class Read_RM_Filtered_Data(Dataset):
    """
    Load into format from the pickled files
    """

    def __init__(self,
                 path) -> None:
        super().__init__()
        
        with open(path, "rb") as f:
            dataset = pickle.load(f)
        self.pairs = []
        self.masks = []

        cnt = 0
        print(f"Loading RM Filtered split")

        for data in dataset:
            prompt = data['prompt']

            self.pairs.append(data['pairs'])
            self.masks.append(data['masks'])

    @classmethod
    def save(self, cls, split, fp):
        examples = []
        for i in range(len(self.pairs)):
            examples.append((self.pairs[i], self.masks[i]))
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]


class DahoasSFTStaticPromptsDataset(Dataset):

    def __init__(self,
                 block_size,
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Dahoas/rm-static", split="train")
        self.prompts = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading DahoasSFTStaticPromptsDataset")
        for data in dataset:
            cnt += 1
            prompt = data['prompt']
            tokens = tokenizer(prompt,
                               max_length=block_size,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")

            self.prompts.append(
                [tokens['input_ids'], tokens['attention_mask'], torch.sum(tokens['attention_mask'])])

            if max_examples and cnt >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("fka/awesome-chatgpt-prompts", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["prompt"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx][0], self.prompts[idx][1], self.prompts[idx][2]  # (1, T), (1, T)


class DahoasSFTStaticPromptsDataset_ITER2(Dataset):

    def __init__(self,
                 block_size,
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Dahoas/rm-static", split="train")
        self.prompts = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading DahoasSFTStaticPromptsDataset")
        for i,data in enumerate(dataset):
            if i < 2000: continue
            cnt += 1
            prompt = data['prompt']
            tokens = tokenizer(prompt,
                               max_length=block_size,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")

            self.prompts.append(
                [tokens['input_ids'], tokens['attention_mask'], torch.sum(tokens['attention_mask'])])

            if max_examples and cnt >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("fka/awesome-chatgpt-prompts", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["prompt"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx][0], self.prompts[idx][1], self.prompts[idx][2]  # (1, T), (1, T)
