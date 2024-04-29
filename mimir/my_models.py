"""
    Model definitions, with basic helper functions. Supports any model as long as it supports the functions specified in Model.
"""
import torch
import torch.nn as nn
# import openai
from typing import List
import numpy as np
import transformers
import time
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
# from hf_olmo import *
import math
from mimir.config import ExperimentConfig
from mimir.custom_datasets import SEPARATOR
from mimir.data_utils import drop_last_word


class Model(nn.Module):
    """
        Base class (for LLMs).
    """
    def __init__(self, config: ExperimentConfig, **kwargs):
        super().__init__()
        self.model = None # Set by child class
        self.tokenizer = None # Set by child class
        self.config = config
        self.device = None
        self.device_map = None
        self.name = None
        self.kwargs = kwargs
        self.cache_dir = self.config.env_config.cache_dir

    def to(self, device):
        """
            Shift model to a particular device.
        """
        self.model.to(device, non_blocking=True)

    def load(self):
        """
            Load model onto GPU (and compile, if requested) if not already loaded with device map.
        """
        if not self.device_map:
            start = time.time()
            try:
                self.model.cpu()
            except NameError:
                pass
            if self.config.openai_config is None:
                self.model.to(self.device, non_blocking=True)
            if self.config.env_config.compile:
                torch.compile(self.model)
            print(f'DONE ({time.time() - start:.2f}s)')

    def unload(self):
        """
            Unload model from GPU
        """
        start = time.time()
        try:
            self.model.cpu()
        except NameError:
            pass
        print(f'DONE ({time.time() - start:.2f}s)')

    def get_token_logprob(self,
                          text: str,
                          tokens: np.ndarray = None,
                          no_grads: bool = True,
                          return_all_probs: bool = False):
        """
            Get the probabilities or log-softmaxed logits for a text under the current model.
            Args:
                text (str): The input text for which to calculate probabilities.
                tokens (numpy.ndarray, optional): An optional array of token ids. If provided, these tokens
                are used instead of tokenizing the input text. Defaults to None. tokens are unnecessary if text and tokenizer is provided.

            Raises:
                ValueError: If the device or name attributes of the instance are not set.

            Returns:
                list: A list of probabilities.
        """
        with torch.set_grad_enabled(not no_grads):
            if self.device is None:
                raise ValueError("Please set self.device and self.name in child class")

            if tokens is not None:
                input_ids = torch.from_numpy(tokens.astype(np.int64)).type(torch.LongTensor)
                if input_ids.shape[0] != 1:
                    # expand first dimension
                    input_ids = input_ids.unsqueeze(0)
            else:
                tokenized = self.tokenizer(
                    text, return_tensors="pt")
                input_ids = tokenized.input_ids

            target_token_log_prob = []
            all_token_log_prob = []
            for i in range(0, input_ids.size(1), self.stride):
                begin_loc = max(i + self.stride - self.max_length, 0)
                end_loc = min(i + self.stride, input_ids.size(1))
                trg_len = end_loc - i  # may be different from stride on last loop
                segment_input_ids = input_ids[:, begin_loc:end_loc].to(self.device)
                #target_ids = input_ids.clone()
                #target_ids[:, :-trg_len] = -100

                logits = self.model(segment_input_ids).logits
                if no_grads:
                    logits = logits.cpu()
                logits = logits[0,:-1,:].contiguous()
                logits = logits[-trg_len:,:]
                
                #(bs=1, vocab_size)
                #shift_logits = logits[..., :-1, :].contiguous()
                #(bs=1, vocab_size)
                log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
                # shift_labels = target_ids[..., 1:]
                # if no_grads:
                #     shift_labels = shift_labels.cpu()
                # shift_labels = shift_labels.contiguous()
                # labels_processed = shift_labels[0]

                target_ids = input_ids[0, begin_loc:end_loc]
                target_ids = target_ids[1:]
                target_ids = target_ids[-trg_len:]
                assert target_ids.shape[0] == logits.shape[0]

                segment_target_log_prob = torch.gather(log_prob, 1, target_ids.unsqueeze(1)).squeeze(1)
                target_token_log_prob.append(segment_target_log_prob)
                all_token_log_prob.append(log_prob)

            target_token_log_prob = torch.cat(target_token_log_prob, dim=0).to(torch.float32).numpy()
            all_token_log_prob = torch.cat(all_token_log_prob, dim=0).to(torch.float32).numpy()


            if not return_all_probs:
                return target_token_log_prob 
            return target_token_log_prob, all_token_log_prob


                # del input_ids
                # del target_ids

            #     for i, token_id in enumerate(labels_processed):
            #         if token_id != -100:
            #             log_probability = log_probabilities[0, i, token_id]
            #             if no_grads:
            #                 log_probability = log_probability.item()
            #             target_token_log_prob.append(log_probability)
            #             all_token_log_prob.append(log_probabilities[0, i])
            
            # # Should be equal to # of tokens - 1 to account for shift
            # assert len(target_token_log_prob) == labels.size(1) - 1
            # all_token_log_prob = torch.stack(all_token_log_prob, dim=0)
            # assert len(target_token_log_prob) == len(all_token_log_prob)

        # if not no_grads:
        #     target_token_log_prob = torch.stack(target_token_log_prob)

        # if not return_all_probs:
        #     return target_token_log_prob
        # return target_token_log_prob, all_token_log_prob

    @torch.no_grad()
    def get_loss(self,
               text: str,
               tokens: np.ndarray=None,
               probs = None):
        """
            Get the log likelihood of each text under the base_model.

            Args:
                text (str): The input text for which to calculate the log likelihood.
                tokens (numpy.ndarray, optional): An optional array of token ids. If provided, these tokens
                are used instead of tokenizing the input text. Defaults to None.
                probs (list, optional): An optional list of probabilities. If provided, these probabilities
                are used instead of calling the `get_probabilities` method. Defaults to None.
        """
        all_prob = probs if probs is not None else self.get_token_logprob(text, tokens=tokens)
        # return -np.mean(all_prob)
        if isinstance(all_prob,torch.Tensor):
            return -torch.mean(all_prob).item()
        else: # numpy or list
            return float(-np.mean(all_prob))

        

    # def load_base_model_and_tokenizer(self, model_kwargs):
    #     """
    #         Load the base model and tokenizer for a given model name.
    #     """
    #     if self.device is None or self.name is None:
    #         raise ValueError("Please set self.device and self.name in child class")

    #     if self.config.openai_config is None:
    #         print(f'Loading BASE model {self.name}...')
    #         device_map = self.device_map # if self.device_map else 'cpu'
    #         if "silo" in self.name or "balanced" in self.name:
    #             raise NotImplementedError
    #         #     from utils.transformers.model import OpenLMforCausalLM
    #         #     model = OpenLMforCausalLM.from_pretrained(
    #         #         self.name, **model_kwargs, device_map=self.device, cache_dir=self.cache_dir)
    #         #     # Extract the model from the model wrapper so we dont need to call model.model
    #         elif "llama" in self.name or "alpaca" in self.name:
    #             # TODO: This should be smth specified in config in case user has
    #             # llama is too big, gotta use device map
    #             model = transformers.AutoModelForCausalLM.from_pretrained(self.name, **model_kwargs, cache_dir=self.cache_dir)
    #             self.device = 'cuda:1'
    #         elif "stablelm" in self.name.lower():  # models requiring custom code
    #             model = transformers.AutoModelForCausalLM.from_pretrained(
    #                 self.name, **model_kwargs, trust_remote_code=True, device_map=device_map, cache_dir=self.cache_dir)
    #         elif "olmo" in self.name.lower():
    #             model = transformers.AutoModelForCausalLM.from_pretrained(
    #                 self.name, **model_kwargs, trust_remote_code=True, cache_dir=self.cache_dir)
    #         else:
    #             model = transformers.AutoModelForCausalLM.from_pretrained(
    #                 self.name, **model_kwargs, device_map=device_map, cache_dir=self.cache_dir)
    #     else:
    #         model = None

    #     optional_tok_kwargs = {}
    #     if "facebook/opt-" in self.name:
    #         print("Using non-fast tokenizer for OPT")
    #         optional_tok_kwargs['fast'] = False
    #     if self.config.dataset_member in ['pubmed'] or self.config.dataset_nonmember in ['pubmed']:
    #         optional_tok_kwargs['padding_side'] = 'left'
    #         self.pad_token = self.tokenizer.eos_token_id
    #     if "silo" in self.name or "balanced" in self.name:
    #         tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(
    #             "EleutherAI/gpt-neox-20b", **optional_tok_kwargs, cache_dir=self.cache_dir)
    #     elif "datablations" in self.name:
    #         tokenizer = transformers.AutoTokenizer.from_pretrained(
    #             "gpt2", **optional_tok_kwargs, cache_dir=self.cache_dir)
    #     elif "llama" in self.name or "alpaca" in self.name:
    #         tokenizer = transformers.LlamaTokenizer.from_pretrained(
    #             self.name, **optional_tok_kwargs, cache_dir=self.cache_dir)
    #     elif "pubmedgpt" in self.name:
    #         tokenizer = transformers.AutoTokenizer.from_pretrained(
    #             "stanford-crfm/BioMedLM", **optional_tok_kwargs, cache_dir=self.cache_dir)
    #     else:
    #         tokenizer = transformers.AutoTokenizer.from_pretrained(
    #             self.name, **optional_tok_kwargs, cache_dir=self.cache_dir,
    #             trust_remote_code=True if "olmo" in self.name.lower() else False)
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    #     return model, tokenizer

    # def load_model_properties(self):
    #     """
    #         Load model properties, such as max length and stride.
    #     """
    #     # TODO: getting max_length of input could be more generic
    #     if "silo" in self.name or "balanced" in self.name:
    #         self.max_length = self.model.model.seq_len
    #     elif hasattr(self.model.config, 'max_position_embeddings'):
    #         self.max_length = self.model.config.max_position_embeddings
    #     elif hasattr(self.model.config, 'n_positions'):
    #         self.max_length = self.model.config.n_positions
    #     else:
    #         # Default window size
    #         self.max_length = 1024
    #     self.stride = self.max_length // 2


# why do we need reference model?
# class ReferenceModel(Model):
#     """
#         Wrapper for reference model
#     """
#     def __init__(self, config: ExperimentConfig, name: str):
#         super().__init__(config)
#         self.device = self.config.env_config.device_aux
#         self.name = name
#         base_model_kwargs = {'revision': 'main'}
#         if 'gpt-j' in self.name or 'neox' in self.name or 'llama' in self.name or 'alpaca' in self.name:
#             base_model_kwargs.update(dict(torch_dtype=torch.float16))
#         if 'gpt-j' in self.name:
#             base_model_kwargs.update(dict(revision='float16'))
#         if ':' in self.name:
#             print("Applying ref model revision")
#             # Allow them to provide revisions as part of model name, then parse accordingly
#             split = self.name.split(':')
#             self.name = split[0]
#             base_model_kwargs.update(dict(revision=split[-1]))
#         self.model, self.tokenizer = self.load_base_model_and_tokenizer(
#             model_kwargs=base_model_kwargs)
#         self.load_model_properties()

#     def load(self):
#         """
#         Load reference model noto GPU(s)
#         """
#         if "llama" not in self.name and "alpaca" not in self.name:
#             super().load()

#     def unload(self):
#         """
#         Unload reference model from GPU(s)
#         """
#         if "llama" not in self.name and "alpaca" not in self.name:
#             super().unload()


class QuantileReferenceModel(Model):
    """
        Wrapper for referenc model, specifically used for quantile regression
    """
    def __init__(self, config: ExperimentConfig, name: str):
        super().__init__(config)
        self.device = self.config.env_config.device_aux
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name,
            num_labels=2,
            max_position_embeddings=1024)
        # Modify model's last linear layer to have only 1 output
        self.model.classifier.linear_out = nn.Linear(self.model.classifier.linear_out.in_features, 1)
        self.load_model_properties()


class LanguageModel(Model):
    """
        Generic LM- used most often for target model
    """
    def __init__(self, config: ExperimentConfig, model_name_or_path: str, device: str, device_map = None,  **kwargs):
        super().__init__(config, **kwargs)
        self.device = device
        self.device_map = device_map

        # Use provided name (if provided)
        # Relevant for scoring-model scenario
        # self.name = self.kwargs.get('name', self.config.base_model)

        # base_model_kwargs = {}
        # if config.revision:
        #     base_model_kwargs.update(dict(revision=config.revision))
        # if 'gpt-j' in self.name or 'neox' in self.name:
        #     base_model_kwargs.update(dict(torch_dtype=torch.float16))
        # if 'gpt-j' in self.name:
        #     base_model_kwargs.update(dict(revision='float16'))
        # self.model, self.tokenizer = self.load_base_model_and_tokenizer(
        #     model_kwargs=base_model_kwargs)
        # self.load_model_properties()

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                        torch_dtype=torch.bfloat16,
                                                        trust_remote_code=True,
                                                        quantization_config=None,
                                                    )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.max_length = self.model.config.max_position_embeddings
        self.stride = self.max_length // 2

    @torch.no_grad()
    def get_ref(self, text: str, ref_model: Model, tokens=None, probs=None):
        """
            Compute the loss of a given text calibrated against the text's loss under a reference model -- MIA baseline
        """
        lls = self.get_ll(text, tokens=tokens, probs=probs)
        lls_ref = ref_model.get_ll(text)

        return lls - lls_ref

    @torch.no_grad()
    def get_rank(self, text: str, log: bool=False):
        """
            Get the average rank of each observed token sorted by model likelihood
        """
        openai_config = self.config.openai_config
        assert openai_config is None, "get_rank not implemented for OpenAI models"

        tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
        logits = self.model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()

    # TODO extend for longer sequences
    @torch.no_grad()
    def get_batch_token_logprob(self, texts: List[str], batch_size: int = 6):
        #return [self.get_ll(text) for text in texts] # -np.mean([self.get_ll(text) for text in texts])
        # tokenized = self.tokenizer(texts, return_tensors="pt", padding=True)
        # labels = tokenized.input_ids
        total_size = len(texts)
        target_logprob = defaultdict(list)
        from tqdm import tqdm

        for batch_idx in range(math.ceil(total_size / batch_size)):
        #for batch_idx in range(0, total_size, batch_size):
            # Delegate batches and tokenize
            batch = texts[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            # no max_leng th no truncation
            tokenized = self.tokenizer(batch, return_tensors="pt", padding=True, return_attention_mask=True)
            batch_input_id = tokenized.input_ids
            batch_attention_mask = tokenized.attention_mask
            # # mask out padding tokens
            #attention_mask = tokenized.attention_mask
            # assert attention_mask.size() == label_batch.size()

            # needs_sliding = label_batch.size(1) > self.max_length // 2
            # if not needs_sliding:
            #     label_batch = label_batch.to(self.device)
            #     attention_mask = attention_mask.to(self.device)

            # Collect token probabilities per sample in batch
            for segment_idx in range(0, batch_input_id.size(1), self.stride):
                begin_loc = max(segment_idx + self.stride - self.max_length, 0)
                end_loc = min(segment_idx + self.stride, batch_input_id.size(1))
                trg_len = end_loc - segment_idx  # may be different from stride on last loop
                segment_input_ids = batch_input_id[:, begin_loc:end_loc]
                segment_attention_mask = batch_attention_mask[:, begin_loc:end_loc]
                # if needs_sliding:
                #     input_ids = input_ids.to(self.device)
                #     mask = mask.to(self.device)

                #target_ids = input_ids.clone()
                # Don't count padded tokens or tokens that already have computed probabilities
                #target_ids[:, :-trg_len] = -100
                # target_ids[attention_mask == 0] = -100
                logits = self.model(input_ids = segment_input_ids.to(self.device), attention_mask = segment_attention_mask.to(self.device)).logits.cpu()
                # process every case in the batch
                for i in range(logits.size(0)):
                    # begin to deal each instance in the current batch and current segment
                    rel_len = torch.sum(segment_attention_mask[i],dim=0).item()
                    if  segment_idx > 0 and rel_len <= self.stride + 1   :
                        continue
                    shift_logits = logits[i,:rel_len,:]
                    shift_logits = shift_logits[:-1, :].contiguous()
                    shift_logits = shift_logits[-trg_len:,:]
                    probabilities = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                    
                    labels = segment_input_ids[i, :rel_len]
                    shift_labels = labels[1:]
                    shift_labels = shift_labels[-trg_len:].contiguous()
                    assert shift_labels.shape[0] == probabilities.shape[0], "shift_label {} != probabilities {}, rel len {} actual len {} ".format(shift_labels.shape[0], probabilities.shape[0], rel_len, len(segment_input_ids[i]))

                    segment_target_logprob = torch.gather(probabilities, 1, shift_labels.unsqueeze(1)).squeeze(1).tolist()
                    target_logprob[batch_idx * batch_size + i].extend(segment_target_logprob)
                    # print("update target logprob {}, batch idx {}  i{}   ".format(batch_idx * batch_size + i, batch_idx, i))
                    # for i, sample in enumerate(shift_labels):
                    #     for j, token_id in enumerate(sample):
                    #         if token_id != -100 and token_id != self.tokenizer.pad_token_id:
                    #             probability = probabilities[i, j, token_id].item()
                    #             all_prob[i].append(probability)
                    # del input_ids
                    # del mask
        assert len(target_logprob) == len(texts), "the logprob size {} != text size {}".format(len(target_logprob), len(texts))
        return target_logprob


            #     shift_logits = logits[..., :-1, :].contiguous()
            #     probabilities = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                
            #     shift_labels = target_ids[..., 1:].contiguous()

            #     shift_labels = shift_labels[:, -trg_len:]
            #     probabilities = probabilities[:, -trg_len:,:]

            #     assert shift_labels.shape[1] == probabilities.shape[1]

            #     all_prob[i] = torch.gather(probabilities, 2, shift_labels.unsqueeze(2)).squeeze(2).tolist()
            #     # for i, sample in enumerate(shift_labels):
            #     #     for j, token_id in enumerate(sample):
            #     #         if token_id != -100 and token_id != self.tokenizer.pad_token_id:
            #     #             probability = probabilities[i, j, token_id].item()
            #     #             all_prob[i].append(probability)

            #     # del input_ids
            #     # del mask
                
            
            # # average over each sample to get losses
            # batch_losses = [-np.mean(all_prob[idx]) for idx in range(label_batch.size(0))]
            # # print(batch_losses)
            # losses.extend(batch_losses)
            # del label_batch
            # del attention_mask
        #return losses #np.mean(losses)

    def sample_from_model(self, texts: List[str], **kwargs):
        """
            Sample from base_model using ****only**** the first 30 tokens in each example as context
        """
        min_words = kwargs.get('min_words', 55)
        max_words = kwargs.get('max_words', 200)
        prompt_tokens = kwargs.get('prompt_tokens', 30)

        # encode each text as a list of token ids
        if self.config.dataset_member == 'pubmed':
            texts = [t[:t.index(SEPARATOR)] for t in texts]
            all_encoded = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device, non_blocking=True)
        else:
            all_encoded = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device, non_blocking=True)
            all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words and tries <  self.config.neighborhood_config.top_p:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {}
            if self.config.do_top_p:
                sampling_kwargs['top_p'] = self.config.top_p
            elif self.config.do_top_k:
                sampling_kwargs['top_k'] = self.config.top_k
            #min_length = 50 if config.dataset_member in ['pubmed'] else 150

            #outputs = base_model.generate(**all_encoded, min_length=min_length, max_length=max_length, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
            #removed minlen and attention mask min_length=min_length, max_length=200, do_sample=True,pad_token_id=base_tokenizer.eos_token_id,
            outputs = self.model.generate(**all_encoded, min_length=min_words*2, max_length=max_words*3,  **sampling_kwargs,  eos_token_id=self.tokenizer.eos_token_id)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

        return decoded

    @torch.no_grad()
    def get_entropy(self, text: str):
        """
            Get average entropy of each token in the text
        """
        # raise NotImplementedError("get_entropy not implemented for OpenAI models")
        
        tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
        logits = self.model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()
    
    @torch.no_grad()
    def get_max_norm(self, text: str, context_len=None, tk_freq_map=None):
        # TODO: update like other attacks
        tokenized = self.tokenizer(
            text, return_tensors="pt").to(self.device)
        labels = tokenized.input_ids

        max_length = context_len if context_len is not None else self.max_length
        stride = max_length // 2 #self.stride
        all_prob = []
        for i in range(0, labels.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, labels.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = labels[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = self.model(input_ids, labels=target_ids)
            logits = outputs.logits
            # Shift so that tokens < n predict n
            # print(logits.shape)
            shift_logits = logits[..., :-1, :].contiguous()
            # shift_logits = torch.transpose(shift_logits, 1, 2)
            probabilities = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            shift_labels = target_ids[..., 1:].contiguous()
            labels_processed = shift_labels[0]

            for i, token_id in enumerate(labels_processed):
                if token_id != -100:
                    probability = probabilities[0, i, token_id].item()
                    max_tk_prob = torch.max(probabilities[0, i]).item()
                    tk_weight = max(tk_freq_map[token_id.item()], 1) / sum(tk_freq_map.values()) if tk_freq_map is not None else 1
                    if tk_weight == 0:
                        print("0 count token", token_id.item())
                    tk_norm = tk_weight
                    all_prob.append((1 - (max_tk_prob - probability)) / tk_norm)

        # Should be equal to # of tokens - 1 to account for shift
        assert len(all_prob) == labels.size(1) - 1
        return -np.mean(all_prob)

