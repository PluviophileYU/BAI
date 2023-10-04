import copy
import logging
import os
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
logger = logging.getLogger(__name__)
import numpy as np
import json_lines, json, csv
from collections import Counter

def env_statistic(env):
    hard_env = np.argmax(env, axis=1).tolist()
    env_count = Counter(hard_env)
    env_num = len(env_count)  # number of the enviroments, e.g., 5 envs
    env_count = [env_count[i] for i in range(env_num)]
    logger.info("Pre defined env distributions with argmax: {}".format(str(env_count)))

    env_prob = np.exp(env) / (np.sum(np.exp(env), axis=-1).reshape(-1,1))
    env_prob = np.mean(env_prob, axis=0).tolist()
    env_prob = [round(i, 3) for i in env_prob]
    logger.info("Pre defined env distribution with average probability: {}".format(str(env_prob)))

def load_and_cache_examples(args, tokenizer, evaluate='train', output_examples=False):

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir, 'cache', args.task_name,
        "cached_{}_{}_{}".format(
            args.model_type,
            evaluate,
            str(args.max_seq_length),
        ),
    )
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        # logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        if args.task_name == 'mnli':
            Processor = MnliProcessor()
            if evaluate == 'train':
                examples = Processor.get_train_examples(args.data_dir)
            elif evaluate == 'dev':
                examples = Processor.get_dev_examples(args.data_dir)
            elif evaluate == 'test':
                examples = Processor.get_test_examples(args.data_dir)
            elif evaluate == 'hans':
                examples = Processor.get_hans_examples(args.data_dir)
            if evaluate != 'hans':
                label_list = Processor.get_labels(iid=True)
            else:
                label_list = Processor.get_labels(iid=False)

        label_map = {label: i for i, label in enumerate(label_list)}
        features, dataset = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            label_map=label_map
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    label_map
):
    features = convert_example_to_features(examples=examples,
                                           tokenizer=tokenizer,
                                           max_seq_length=max_seq_length,
                                           label_map=label_map)


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    all_ids = torch.tensor([f.id for f in features], dtype=torch.long)
    all_envs = torch.tensor([f.env for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_masks,
        all_token_type_ids,
        all_labels,
        all_ids,
        all_envs  # Here the envs is just the pre-defined domain split
    )
    return features, dataset


def convert_example_to_features(examples, tokenizer, max_seq_length, label_map):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="feature_converting")):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_seq_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(attention_mask) == max_seq_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                max_seq_length)
        assert len(token_type_ids) == max_seq_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                max_seq_length)
        label = label_map[example.label]

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
                id=int(example.guid),
                env=example.env
            )
        )
    return features


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors.

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))
    @classmethod
    def _read_txt(cls, input_file):
        with open(input_file, 'r') as f:
            data = f.readlines()
        return data

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_iid(self._read_tsv(os.path.join(data_dir, 'mnli', "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_iid(self._read_tsv(os.path.join(data_dir, 'mnli', "dev_matched.tsv")), "dev_matched")

    def get_hans_examples(self, data_dir):
        return self._create_examples_ood(self._read_txt(os.path.join(data_dir, 'mnli', "hans.txt")))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'mnli', "test_matched.tsv")), "test_matched")

    def get_labels(self, iid=True):
        """See base class."""
        if iid:
            return ["contradiction", "entailment", "neutral"]
        else:
            return ["non-entailment", "entailment"]

    def get_env(self):
        return ['fiction', 'government', 'slate', 'telephone', 'travel']

    def _create_examples_iid(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        env_list = self.get_env()
        env_map = {env: i for i, env in enumerate(env_list)}
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            text_a = line[8]
            text_b = line[9]
            label = None if set_type.startswith("test") else line[-1]
            env = env_map[line[3]] if set_type.startswith("train") else 0  # Only training sample has env, for others simply set to 0
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, env=env))
        return examples

    def _create_examples_ood(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            # guid = "%s-%s" % (set_type, line[0])
            text_a = line.split('\t')[5]
            text_b = line.split('\t')[6]
            label = line.split('\t')[0]
            examples.append( # Only training sample has env, for others simply set to 0
                InputExample(guid=i-1, text_a=text_a, text_b=text_b, label=label, env=0))
        return examples


class InputFeatures(object):

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        label,
        id,
        env
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.id = id
        self.env = env


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, env=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.env = env

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
