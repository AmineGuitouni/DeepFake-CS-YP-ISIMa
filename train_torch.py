import os

import torch.nn as nn

# multiprocessing
import torch.distributed as dist

from train_base import *

# constants
SYNC = False
GET_MODULE = True

def main():
    args = parse_args()

    # Init dist => modified bech ma ya3mlouch error ki ma yal9ahomch
    local_rank = int(os.environ.get("LOCAL_RANK",0))
    global_rank = int(os.environ.get("RANK",0))
    world_size = int(os.environ.get("WORLD_SIZE",1))

    args = init_env(args, local_rank, global_rank)

    # Init the process group
    print('Initializing Process Group...')
    #dist.init_process_group(backend=args.dist_backend, init_method=("env://%s:%s" % (args.master_addr, args.master_port)),
    #   world_size=world_size, rank=global_rank)
    print('Process group ready!')

    model = init_models(args)
    #model = load_dicts(args, model)
    
    # Wrap the model
    #model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = init_optims(args, world_size, model)

    lr_scheduler = init_schedulers(args, optimizer)

    # resume from saved state if one exists
    state, checkpoint_dir = load_state(args, global_rank,
                                       model,
                                       optimizer,
                                       lr_scheduler)
    
    model = state.model
    optimizer = state.optimizer
    lr_scheduler = state.scheduler

    # dataset
    train_sampler, dataloader = init_dataset(args, state, global_rank, world_size)
    val_sampler, val_dataloader = init_dataset(args, state, global_rank, world_size, True)

    state.paths_file = dataloader.dataset.save_path
    state.val_paths_file = val_dataloader.dataset.save_path

    train(args, global_rank, world_size, SYNC, GET_MODULE,
            state, checkpoint_dir,
            model,
            train_sampler, dataloader, val_sampler, val_dataloader,
            optimizer,
            lr_scheduler)

if __name__ == '__main__':
    main()