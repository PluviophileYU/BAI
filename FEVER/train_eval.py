import torch, os, logging, timeit, json
import random
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils import load_and_cache_examples

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)
logging.getLogger("transformers.data.metrics.squad_metrics").setLevel(logging.WARN)  # Reduce model loading logs

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(args, model, train_dataset, tokenizer, tb_writer, exec_time):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # train_dataset = load_and_cache_examples(args, tokenizer, evaluate='train', output_examples=False)
    # train_dataset = load_and_cache_examples(args, tokenizer, evaluate='hans', output_examples=False)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_ratio*t_total, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level, verbosity=0)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_acc = 0

    if args.task_name == 'fever':
        iid_set = 'dev'
        ood_set = 'symv2test'

    if args.recur_dir[-1] == 'N.A':
        xx = 20
    else:
        xx = 100

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
                "idx": batch[4]
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss_task, loss_penalty = outputs[0], outputs[1]
            loss = loss_task + args.inv_penalty_scale * loss_penalty
            # loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics

                tb_writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("train/loss", loss, global_step)
                tb_writer.add_scalar("train/loss_task", loss_task.mean(), global_step)
                tb_writer.add_scalar("train/loss_penalty", loss_penalty.mean(), global_step)
                logging_loss = tr_loss

                if global_step % args.save_steps == 0 and global_step > int(t_total/xx):
                # if global_step % args.save_steps == 0:
                    now_iid_acc = evaluate(args, model, tokenizer, iid_set)
                    now_ood_acc = evaluate(args, model, tokenizer, ood_set)
                    logger.info('iid/ood performance: {} / {} '.format(round(now_iid_acc * 100, 2), round(now_ood_acc * 100, 2)))
                    if args.iid_valid:
                        now_acc = now_iid_acc
                    else:
                        now_acc = now_ood_acc
                    if now_acc > best_acc:
                        best_acc = now_acc
                        logger.info(
                            "Now Best checkpoint and its ood result: {}, {}".format(global_step, round(now_ood_acc * 100, 2)))
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "{}-checkpoint-best".format(exec_time))
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        # logger.info("Saving optimizer and scheduler states to %s", output_dir)

    now_iid_acc = evaluate(args, model, tokenizer, iid_set)
    now_ood_acc = evaluate(args, model, tokenizer, ood_set)
    logger.info('iid/ood performance: {} / {} '.format(round(now_iid_acc * 100, 2), round(now_ood_acc * 100, 2)))
    if args.iid_valid:
        now_acc = now_iid_acc
    else:
        now_acc = now_ood_acc
    if now_acc > best_acc:
        best_acc = now_acc
        logger.info(
            "Now Best checkpoint and its ood result: {}, {}".format(global_step, round(now_ood_acc * 100, 2)))
        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, "{}-checkpoint-best".format(exec_time))
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        # logger.info("Saving optimizer and scheduler states to %s", output_dir)

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, set='dev'):
    dataset, examples, features = load_and_cache_examples(args,
                                                          tokenizer,
                                                          evaluate=set,
                                                          output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    start_time = timeit.default_timer()
    preds = None

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]
            outputs = model(**inputs)
            logits = outputs[0]

            inputs["labels"] = batch[3]

        if preds is None:
            preds = torch.nn.functional.softmax(logits, 1).detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, torch.nn.functional.softmax(logits, 1).detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    # preds = np.concatenate(((preds[:, 0].reshape(-1, 1), preds[:, 1] + preds[:, 2]).reshape(-1, 1)), axis=1)
    # take max of non-entailment rather than taking their sum
    # preds[:, 1] = preds[:, [1, 2]].max(axis=1)
    # preds = preds[:, :2]
    preds = np.argmax(preds, axis=1)
    result = (preds == out_label_ids).mean()
    return result


