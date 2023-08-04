# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import wandb

import sys
sys.path.insert(1, 'libs/')
from manifmetric.manifmetric import ManifoldMetric
from manifmetric.nets import EDM
from manifmetric.dataset import block_draw

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    fold_ticks              = None,     # Intervals at which to update dataset fold.
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        ### Init wandb and tensorboard
        # ---------------------------------------- #
        wandb.init(project='creative_metric_gms', name=run_dir, dir=run_dir, 
            config=dict(batch_size=batch_size, fold_ticks=fold_ticks, training_set_kwargs=dataset_kwargs, data_loader_kwargs=data_loader_kwargs, network_kwargs=network_kwargs, loss_kwargs=loss_kwargs, optimizer_kwargs=optimizer_kwargs, augment_kwargs=augment_kwargs), resume='allow', sync_tensorboard=True)
        
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)
        # ---------------------------------------- #

        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    inception_metric, vgg_metric = None, None
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    pre_fold = -1
    while True:

        ### Setup training data for folds
        # ---------------------------------------- #
        if fold_ticks is not None and fold_ticks > 0 and (new_fold := min(cur_tick // fold_ticks, len(dataset_obj.folds)-1)) != pre_fold:
            pre_fold = new_fold
            dataset_obj.update_raw_idx(dataset_obj.folds[new_fold])
            dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed+new_fold)
            dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
            if dist.get_rank() == 0:
                print()
                print(f'Fold {new_fold}')
                print('Fold Num images: ', len(dataset_obj))
                print('Fold Image shape: ', dataset_obj.image_shape)
                print('Fold Label shape: ', dataset_obj.label_shape)
                print()
        # ---------------------------------------- #

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        ### VALIDATION METRICS EVALUATION
        # ---------------------------------------- #
        def val_transform(data):
            img = data[0] if isinstance(data, (tuple, list)) else data
            img = torch.as_tensor(img, dtype=torch.float32) / 255
            if img.shape[1] == 1:
                img = img.repeat([1, 3, 1, 1])
            return img
        
        def model_transform(data):
            img = data[0] if isinstance(data, (tuple, list)) else data
            if img.shape[1] == 1:
                img = img.repeat([1, 3, 1, 1])
            return img
        
        def is_distributed():
            return torch.distributed.is_initialized()
        
        num_gpus = dist.get_world_size()
        rank = dist.get_rank()
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            if rank == 0:
                print(f'Validation metrics...')
            
            if vgg_metric is None or inception_metric is None:
                ### Setup validation set
                head, _ = os.path.split(os.path.normpath(dataset_kwargs['path']))
                validation_path = os.path.join(head, 'test')
                assert os.path.exists(validation_path), f'Path does not exist: {validation_path}'
                validation_set_kwargs = dict(dataset_kwargs, path=validation_path, fold_path=None, fold_id=None, xflip=False, use_labels=False, max_size=None)
                validation_set = dnnlib.util.construct_class_by_name(**validation_set_kwargs) # subclass of training.dataset.Dataset
                validation_split_len = int(np.ceil(len(validation_set) / num_gpus))
                validation_sampler = torch.arange(len(validation_set))[rank*validation_split_len:(rank+1)*validation_split_len]
                validation_loader = torch.utils.data.DataLoader(dataset=validation_set, sampler=validation_sampler, batch_size=batch_gpu, **data_loader_kwargs)
                if rank == 0:
                    print(f'\n>>> Read validation data from {validation_path}')
                    print('Val Num images: ', len(validation_set))
                    print('Val Image shape: ', validation_set.image_shape)
                    print('Val Label shape: ', validation_set.label_shape)
                    print()

                ### Collect validation features
                vgg_metric = ManifoldMetric(model='vgg16')
                feats = vgg_metric.extract_features(validation_loader, device=device, output_shape=len(validation_set), output_ids=validation_sampler, transform=val_transform)
                if is_distributed():
                    torch.distributed.reduce(feats, dst=0, op=torch.distributed.ReduceOp.SUM)
                if rank == 0:
                    vgg_metric.compute_ref_stats(feats, k=5)
                
                inception_metric = ManifoldMetric(model='inceptionv3')
                feats = inception_metric.extract_features(validation_loader, device=device, output_shape=len(validation_set), output_ids=validation_sampler, transform=val_transform)
                if is_distributed():
                    torch.distributed.reduce(feats, dst=0, op=torch.distributed.ReduceOp.SUM)
                if rank == 0:
                    inception_metric.compute_ref_stats(feats)
            
            ### Collect model features
            model_loader = EDM(model=ema).get_iter(size=len(validation_sampler), batch_size=batch_gpu)
            feats = vgg_metric.extract_features(model_loader, device=device, output_shape=len(validation_set), output_ids=validation_sampler, transform=model_transform)
            if is_distributed():
                torch.distributed.reduce(feats, dst=0, op=torch.distributed.ReduceOp.SUM)
            if rank == 0:
                val_metrics = dict()
                vgg_metric.compute_gen_stats(feats, k=5)
                val_metrics['prec'] = vgg_metric.precision()
                val_metrics['comp_prec'] = vgg_metric.comp_precision()
                val_metrics['recall'] = vgg_metric.recall()
                val_metrics['density'] = vgg_metric.density()
                val_metrics['coverage'] = vgg_metric.coverage()
            ### Conserve memory
            del feats
            vgg_metric.gen_stats = None

            feats = inception_metric.extract_features(model_loader, device=device, output_shape=len(validation_set), output_ids=validation_sampler, transform=model_transform)
            if is_distributed():
                torch.distributed.reduce(feats, dst=0, op=torch.distributed.ReduceOp.SUM)
            if rank == 0:
                inception_metric.compute_gen_stats(feats)
                val_metrics['fid'] = inception_metric.fid()
                val_metrics['kid'] = inception_metric.kid()
            ### Conserve memory
            del feats
            inception_metric.gen_stats = None

            ### Save val metrics
            if rank == 0:
                global_step = int(cur_nimg / 1e3)
                jsonl = json.dumps(dict(val_metrics, nimg=cur_nimg, tick=cur_tick, step=global_step))
                with open(os.path.join(run_dir, 'val_metrics.jsonl'), 'at') as fs:
                    fs.write(jsonl + '\n')
                    fs.flush()
                if stats_tfevents is not None:
                    for metric_name, metric_val in val_metrics.items():
                        stats_tfevents.add_scalar(f'val_metrics/{metric_name}', metric_val, global_step=global_step)
        
        ### Draw real samples
        draw_num_samples = 64
        if rank == 1:
            draw_reals = list()
            num_reals = 0
            for data in validation_loader:
                draw_reals.append(val_transform(data))
                num_reals += draw_reals[-1].shape[0]
                if num_reals >= draw_num_samples:
                    break
            block_draw(torch.concat(draw_reals).permute(0, 2, 3, 1),
                path=os.path.join(run_dir, 'val_reals.png'), border=True)
            del draw_reals
        
        ### Draw fake samples
        draw_per_gpu = int(np.ceil(draw_num_samples // num_gpus))
        draw_model_loader = EDM(model=ema).get_iter(size=draw_per_gpu, batch_size=batch_gpu)
        draw_gens = list()
        for data in draw_model_loader:
            draw_gens.append(model_transform(data))
        draw_gens = torch.concat(draw_gens)
        draw_gather_list = [torch.zeros_like(draw_gens)]*num_gpus
        dist.gather(draw_gens, gather_list=draw_gather_list, dst=0)
        if rank == 1:
            block_draw(torch.concat(draw_gather_list).permute(0, 2, 3, 1)[:draw_num_samples],
                path=os.path.join(run_dir, 'val_gens.png'), border=True)
        del draw_gens
        del draw_gather_list

        # ---------------------------------------- #

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
