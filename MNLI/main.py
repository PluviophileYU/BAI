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
import numpy as np
from args import get_args, num_labels
from train_eval import train, evaluate, set_seed
from ref_main import ref_main
from model import *
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AutoConfig,
    AutoTokenizer,
)
from datetime import datetime, timezone, timedelta
from utils import load_and_cache_examples, env_statistic

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    args = get_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        print(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Set tensorboard
    exec_time = datetime.utcnow().astimezone(timezone(timedelta(hours=8))) \
        .strftime("%Y-%m-%d-%H-%M")
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=os.path.join('../logs', exec_time))
    logger.info('Time stamp for now: {}'.format(exec_time))

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

    # Configurations
    config.num_labels = num_labels[args.task_name]

    if args.model_type == 'bert':
        Model = IRM_Bert

    logger.info("Training/evaluation parameters %s", args)

    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Extract the env
    if args.do_ref:
        env = ref_main(args, tb_writer)
        env_file = os.path.join(args.output_dir, args.task_name, 'env-{}.npy'.format(exec_time))
        np.save(env_file, env)
        logger.info("Store the env matrix at {}".format(env_file))
        env_statistic(env)
    else:
        env_file = os.path.join(args.output_dir, args.task_name, args.env_file)
        env = np.load(env_file).astype(dtype='float32')
        env_statistic(env)

    # Training
    if args.do_train:
        args.recur_dir = args.recur_dir.split(',')
        args.training = True
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate='train', output_examples=False)
        if args.recur_dir[-1] == 'N.A':
            model = Model.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                args=args,
                envs=env,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
        else:
            model = Model.from_pretrained(
                args.recur_dir[-1].split('@')[0],  # Load the last ckpt
                args=args,
                envs=env
            )
            env = np.load(env_file).astype(dtype='float32')
            model.envs = torch.nn.Parameter(torch.from_numpy(env), requires_grad=args.update_env)
        model.to(args.device)

        global_step, tr_loss = train(args, model, train_dataset, tokenizer, tb_writer, exec_time)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            exec_time = '08-31-04-04'
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            checkpoints = [i for i in checkpoints if exec_time in i]

        logger.info("Evaluate the following checkpoints for dev: %s", checkpoints)

        best_acc = 0
        best_ckpt = checkpoints[0]
        for checkpoint in checkpoints:
            # Reload the model
            logger.info("Load the checkpoint: {}".format(checkpoint))
            args.training = False
            model = Model.from_pretrained(checkpoint, args=args, envs=env,)  # , force_download=True)
            model.to(args.device)

            # Evaluate on iid dev set
            iid_result = evaluate(args, model, tokenizer, set='dev')

            # Evaluate on ood set
            if args.task_name == 'mnli':
                ood_set = 'hans'
            ood_result = evaluate(args, model, tokenizer, set=ood_set)
            logger.info("-------------{} Results-------------".format(args.task_name.upper()))
            logger.info("IID performance: {}".format(iid_result))
            logger.info("OOD performance: {}".format(ood_result))


if __name__ == "__main__":
    main()
