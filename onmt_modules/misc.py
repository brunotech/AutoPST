# -*- coding: utf-8 -*-

import torch
import random
import inspect
from itertools import islice, repeat
import os


def split_corpus(path, shard_size, default=None):
    """yield a `list` containing `shard_size` line of `path`,
    or repeatly generate `default` if `path` is None.
    """
    return _split_corpus(path, shard_size) if path is not None else repeat(default)


def _split_corpus(path, shard_size):
    """Yield a `list` containing `shard_size` line of `path`.
    """
    with open(path, "rb") as f:
        if shard_size <= 0:
            yield f.readlines()
        else:
            while True:
                if shard := list(islice(f, shard_size)):
                    yield shard
                else:
                    break


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = iter(args)
    first = next(arguments)
    assert all(
        arg == first for arg in arguments
    ), f"Not all arguments have the same value: {args}"


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def set_random_seed(seed, is_cuda):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)


def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    return distance_mat_clipped + max_relative_positions


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    return x_tz_matmul_r.permute(1, 2, 0, 3)


def fn_args(fun):
    """Returns the list of function arguments name."""
    return inspect.getfullargspec(fun).args


def report_matrix(row_label, column_label, matrix):
    header_format = "{:>10.10} " + "{:>10.7} " * len(row_label)
    row_format = "{:>10.10} " + "{:>10.7f} " * len(row_label)
    output = header_format.format("", *row_label) + '\n'
    for word, row in zip(column_label, matrix):
        max_index = row.index(max(row))
        row_format = row_format.replace(
            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
        row_format = row_format.replace(
            "{:*>10.7f} ", "{:>10.7f} ", max_index)
        output += row_format.format(word, *row) + '\n'
        row_format = "{:>10.10} " + "{:>10.7f} " * len(row_label)
    return output


def check_model_config(model_config, root):
    # we need to check the model path + any tokenizer path
    for model in model_config["models"]:
        model_path = os.path.join(root, model)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'{model_path} from model {model_config["id"]} does not exist'
            )
    if (
        "tokenizer" in model_config.keys()
        and "params" in model_config["tokenizer"].keys()
    ):
        for k, v in model_config["tokenizer"]["params"].items():
            if k.endswith("path"):
                tok_path = os.path.join(root, v)
                if not os.path.exists(tok_path):
                    raise FileNotFoundError(
                        f'{tok_path} from model {model_config["id"]} does not exist'
                    )
