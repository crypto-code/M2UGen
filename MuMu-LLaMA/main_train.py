import warnings

warnings.filterwarnings("ignore")

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, 
    FullStateDictConfig)

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from llama.mumu_llama import MuMu_LLaMA

from data.dataset import FinetuneDataset

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from accelerate import Accelerator
from engine_train import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('imagebind-llm pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='7B', type=str,
                        help='Type of LLaMA model')  #
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--mert_path', default="m-a-p/MERT-v1-330M", type=str,
                        help='path to MERT pretrained checkpoint')
    parser.add_argument('--vit_path', default="google/vit-base-patch16-224-in21k", type=str,
                        help='path to ViT pretrained checkpoint')
    parser.add_argument('--vivit_path', default="google/vivit-b-16x2-kinetics400", type=str,
                        help='path to ViViT pretrained checkpoint')
    parser.add_argument('--music_decoder', default="musicgen", type=str,
                        help='Decoder to use musicgen/audioldm2')
    parser.add_argument('--music_decoder_path', default="facebook/musicgen-medium", type=str,
                        help='Path to decoder to use musicgen/audioldm2')
    parser.add_argument('--max_words', default=2048, type=int,
                        help='max number of input words')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-2, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-2, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--stage', type=int, default=1, metavar='N',
                        help='Training stage')
    parser.add_argument('--load_same_stage', action='store_true',
                        help='Load data from same training stage')

    # Dataset parameters
    parser.add_argument('--data_config', default='configs/data/pretrain/EN.yaml', type=str,
                        help='dataset config path')
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # parser.set_defaults(pin_mem=True)

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=420, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--split_epoch', type=int, default=50)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # define the model
    llama_type = args.llama_type
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    llama_tokenzier_path = args.llama_path
    model = MuMu_LLaMA(llama_ckpt_dir, llama_tokenzier_path, args, knn=False, stage=args.stage)
    # model.llama.to(device)
    # model.output_projector.to(device)
    # model.generation_model.to(device)

    model.to(device)
    # model.mert_model.to("cpu")
    # model.vit_model.to("cpu")
    # model.vivit_model.to("cpu")

    # for parameter_name, parameter in model.named_parameters():
    #     print(parameter_name, parameter.device)

    model_without_ddp = model
    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
        print("Model is parallelizable")
        
    print("Model = %s" % str(model_without_ddp))

    print("Trainable Params:")
    print([(key, val.shape) for key, val in model.named_parameters() if val.requires_grad])

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module

    # training detail
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    dataset_train = FinetuneDataset(max_words=args.max_words, tokenizer=model_without_ddp.tokenizer, stage=args.stage)
    print(dataset_train)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = misc.DistributedSubEpochSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, split_epoch=args.split_epoch, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # SummaryWrite
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.output_dir and os.path.exists(f"{args.output_dir}/checkpoint.pth"):
        print(f"Loading model...")
        if args.load_same_stage:
            args.start_epoch = misc.load_model(model_without_ddp, optimizer, loss_scaler,
                                               f"{args.output_dir}/checkpoint.pth")
        else:
            misc.load_model(model_without_ddp, None, None, f"{args.output_dir}/checkpoint.pth")
        print("Model initialized")

    accelerator = Accelerator()
    model = accelerator.prepare_model(model)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, accelerator, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch + 1) % 5 == 0:
            print(f"Saving Model to ", args.output_dir)
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            print(f"Model Saved")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
