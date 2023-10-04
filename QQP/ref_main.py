# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import glob
import logging
import os
from args import get_args, num_labels
from train_eval import train, evaluate, set_seed
from prettytable import PrettyTable
from torch.utils.data import TensorDataset
from model import *
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AutoConfig,
    AutoTokenizer,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from utils import load_and_cache_examples
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm, trange
from collections import Counter

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
logger = logging.getLogger(__name__)

class Ref_Bert(BertPreTrainedModel):
    def __init__(self, config, args, num_train):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_train = num_train
        self.args = args
        self.bert = BertModel(config)
        for p in self.bert.parameters():
            p.requires_grad = False

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier.weight.requires_grad = False
        self.classifier.bias.requires_grad = False

        ###########
        self.env_logits = nn.Parameter(torch.randn(num_train, args.num_envs))
        self.env_feature = nn.Parameter(torch.randn(num_train, 5), requires_grad=False)
        self.env_classifier = nn.Sequential(nn.Linear(5, args.num_envs),
                                            nn.Tanh(),
                                            nn.Linear(args.num_envs, args.num_envs))
        self.init_weights()

    def forward(
        self,
        stage=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        idx=None,
    ):
        if stage == '1':  # Extract the confidence firstly
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            pooled_output = outputs[1]
            outputs = self.classifier(pooled_output)
            loss = F.cross_entropy(outputs, labels, reduction='none')
            return loss, outputs

        elif stage == '2': # Train the env classifier then
            outputs = self.env_feature[:, 2:]
            # IRM Regularizer
            loss_value = F.cross_entropy(outputs, labels, reduction='none')
            scale = torch.ones((1, outputs.size(-1))).type_as(outputs).requires_grad_()
            penalty = F.cross_entropy(outputs * scale, labels, reduction='none')

            # env = self.env_classifier(self.env_feature)
            env = self.env_logits
            split_logits = F.log_softmax(env, dim=-1)
            if self.args.ref_hardsplit:
                split = F.gumbel_softmax(split_logits, tau=1, hard=True)
            else:
                split = F.softmax(split_logits, dim=-1)
            penalty = (split * penalty.unsqueeze(-1) / (split.sum(0) + 1e-20)).sum(0)
            erm_risk = (split * loss_value.unsqueeze(-1) / (split.sum(0)+1e-20)).sum(0)
            irm_risk_list = []
            for index in range(penalty.size(0)):
                irm_risk = torch.autograd.grad(penalty[index], [scale], create_graph=True)[0]
                irm_risk_list.append(torch.sum(irm_risk ** 2))
            # irm_risk_all = - erm_risk.mean() - 1e6 * torch.stack(irm_risk_list).mean()
            irm_risk_all = - 1e6 * torch.stack(irm_risk_list).mean()
            return irm_risk_all, self.env_logits

def ref_main(args, tb_writer):
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce model loading logs
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)  # Reduce model loading logs
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)  # Reduce model loading logs
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    config.num_labels = num_labels[args.task_name]
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate='train', output_examples=False)
    num_train = len(train_dataset)

    if args.model_type == 'bert':
        Model = Ref_Bert

    model = Model.from_pretrained(
        args.ref_dir,
        args=args,
        num_train=num_train,
    )
    model.to(args.device)

    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size*4)

    # Stage one
    import numpy as np
    feature = np.zeros((num_train, 2+2))
    for i, batch in tqdm(enumerate(train_dataloader)):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "stage": '1',
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
                "idx": batch[4]
            }
            idx = batch[4].detach().cpu().numpy().tolist()
            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]
            loss, logits = model(**inputs)
            loss = loss.detach().cpu().numpy()
            prob = F.softmax(logits, dim=-1)
            gt_confidence = (F.one_hot(batch[3], num_classes=2).type_as(prob) * prob).sum(-1)
            gt_confidence = gt_confidence.detach().cpu().numpy()
            hi_confidence = torch.max(prob, dim=-1)[0]
            hi_confidence = hi_confidence.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            item = np.append(gt_confidence.reshape(-1,1), hi_confidence.reshape(-1,1), axis=1)
            item = np.append(item, logits, axis=1)
            feature[idx] = item
    logger.info("Stage one: Extract feature done!")


    # Stage two
    model.env_feature = nn.Parameter(torch.from_numpy(feature).type_as(model.classifier.weight), requires_grad=False)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.ref_learning_rate, eps=args.adam_epsilon)
    # optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.ref_learning_rate, momentum=0.9, weight_decay=0)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level, verbosity=0)

    epochs_trained = 0
    global_step = 0
    set_seed(args)
    best_loss = 1e5
    update_scale = True
    best_model = None
    from copy import deepcopy
    epoch_iterator = tqdm(range(args.ref_train_steps), desc="Iteration")
    for step in enumerate(epoch_iterator):
        model.train()
        labels = train_dataset[:][3].to(args.device)
        inputs = {
            "stage": '2',
            "labels": labels
        }

        if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
            del inputs["token_type_ids"]

        loss, env_logits = model(**inputs)
        if update_scale:
            scale_multi = irm_scale(loss, -50)
            update_scale = False  # every time just update once
        loss *= scale_multi

        if loss.detach().item() < best_loss:
            best_env = copy.deepcopy(env_logits)
            best_loss = loss.detach().item()
            logger.info("Best loss: {}".format(best_loss))

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        # scheduler.step()  # Update learning rate schedule
        global_step += 1
        tb_writer.add_scalar("train/ref_training_loss", loss, global_step)
        model.zero_grad()
    logger.info("Stage two: Train env classifier done!")

    best_env = best_env.detach().cpu().numpy()
    return best_env

def irm_scale(irm_loss, default_scale=-50):
    with torch.no_grad():
        scale = default_scale / irm_loss.clone().detach()
    return scale