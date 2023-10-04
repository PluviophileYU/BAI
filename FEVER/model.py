import torch
import torch.nn as nn
import numpy as np
import copy
import math
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, KLDivLoss, MSELoss
from transformers.modeling_bert import (
    BertPreTrainedModel, BertModel)

class IRM_Bert(BertPreTrainedModel):
    def __init__(self, config, args, envs):
        super().__init__(config, args)
        self.num_labels = config.num_labels
        self.num_envs = args.num_envs
        if args.recur_dir[-1] == 'N.A':
            self.envs = torch.nn.Parameter(torch.from_numpy(envs), requires_grad=args.update_env)
        else:
            if args.training:
                old_envs_num = int(args.recur_dir[-1].split('@')[-1])
            else:
                old_envs_num = args.num_envs
            self.envs = nn.Parameter(torch.randn((envs.shape[0], old_envs_num)), requires_grad=args.update_env)  # Load outside, here just init a placeholder
        self.args = args

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = invariant_classifier(config, args)

        if args.recur_dir[-1] != 'N.A':
            for p in self.bert.parameters():
                p.requires_grad = False

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        domains=None,
        idx=None,
    ):
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

        pooled_output = self.dropout(pooled_output)

        if labels is not None:
            logits, loss_penalty = self.classifier(pooled_output, train=True, envs=self.envs, idx=idx, hardsplit=self.args.irm_hardsplit)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, loss_penalty, logits) + outputs
        else:
            logits = self.classifier(pooled_output, train=False)
            outputs = (logits, ) + outputs
        return outputs


class invariant_classifier(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        self.num_labels = config.num_labels
        self.num_envs = args.num_envs

        if args.recur_dir[-1] == 'N.A':
            now_envs = 0
        else:
            now_envs = 0
            for i in args.recur_dir:
                num = i.split('@')[-1]
                now_envs += int(num)
        now_envs += args.num_envs

        self.cls = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels, bias=False) for i in range(now_envs)])

        if args.recur_dir[-1] != 'N.A':
            for classifier in self.cls[ : -args.num_envs]:
                classifier.weight.requires_grad = False

    def forward(self, hidden, train=True, envs=None, idx=None, hardsplit=True):
        # Training utilizes different classifier for each env
        if train:
            batch_size = hidden.size()[0]
            env = envs[idx]
            env = F.log_softmax(env, dim=-1)
            if hardsplit:
                env = F.gumbel_softmax(env, tau=1, hard=True)
            else:
                env = F.softmax(env, dim=-1)
            # logits according to the environments.
            logits = []
            for i in range(batch_size):
                e = env[i].unsqueeze(0)
                logit_ = torch.stack([net(hidden[i]) for net in self.cls[-self.args.num_envs: ]])  # num_env * num_label
                logit = torch.mm(e, logit_)
                logits.append(logit)
            logits = torch.cat(logits, dim=0)
            # variance penalty
            W_mean = torch.stack([net.weight for net in self.cls], 0).mean(0)
            var_penalty = [(torch.norm(net.weight - W_mean, p=2) / torch.norm(net.weight, p=1)) ** 2 for net in self.cls]
            loss_penalty = sum(var_penalty) / len(var_penalty)
            return logits, loss_penalty
        else:
            W_mean = torch.stack([net.weight for net in self.cls], 0).mean(0)
            logits = nn.functional.linear(hidden, W_mean)
            return logits



