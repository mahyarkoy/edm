import subprocess
import sys
import argparse
import os

SBATCH_STR = '''#!/bin/bash 
#SBATCH --job-name={job_name}
#SBATCH --ntasks=1 
#SBATCH --account=gard 
#SBATCH --qos=premium 
#SBATCH --partition=ALL 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:{device} 
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output={save_dir}/sbatch_{job_name}.out
#SBATCH --error={save_dir}/sbatch_{job_name}.out
#SBATCH --open-mode=truncate

echo "HOSTNAME: "$(hostname)
echo "tmpdir for the job: "$TMPDIR 
echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES 
echo "CPU allocated: "$(taskset -c -p $$)
echo "GPU allocated: "$CUDA_VISIBLE_DEVICES
nvidia-smi
source /nas/home/mkhayat/.bashrc
conda activate py38edm
export TORCH_EXTENSIONS_DIR={save_dir}
'''

SBATCH_STR_LARGE = '''#!/bin/bash 
#SBATCH --job-name={job_name} 
#SBATCH --ntasks=1 
#SBATCH --account=gard 
#SBATCH --qos=premium_memory
#SBATCH --partition=large_gpu
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:{device}
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output={save_dir}/sbatch_{job_name}.out
#SBATCH --error={save_dir}/sbatch_{job_name}.out
#SBATCH --open-mode=truncate

echo "HOSTNAME: "$(hostname)
echo "tmpdir for the job: "$TMPDIR 
echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES 
echo "CPU allocated: "$(taskset -c -p $$)
echo "GPU allocated: "$CUDA_VISIBLE_DEVICES
nvidia-smi
source /nas/home/mkhayat/.bashrc
conda activate py38edm
'''

COPY_STR = 'cp -r {temp_dir} {save_dir}'
DEL_STR = 'rm -r {temp_dir}'

EXP_CONFIGS = {
    'cifar_separate':
    {
        'cmd': 'torchrun --standalone --nproc_per_node=2 train.py',
        'outdir': 'cifar_{augpipe}_fold{fold_id}_foldticks{fold_ticks}_seeddata{seed}',
        'cond': 0,
        'xflip': 0,
        'data': 'data/cifar/train',
        'fold_path': 'data/cifar/train/folds_10eq_seed{seed}.pk',
        'fold_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'fold_ticks': 0,
        'duration': 25, ### MImgs
        'tick': 4,
        'snap': 25000 // (4*50),
        'dump': 10 * 25000 // (4*50),
        'seed': 1000,
        'cache': 1,
        'workers': 4,
        'batch-gpu': 64,
        'augpipe': 'blit' #bgc
    },
    
    'cifar_test':
    {
        'cmd': 'torchrun --standalone --nproc_per_node=2 train.py',
        'outdir': 'te_cifar_fold{fold_id}_foldticks{fold_ticks}_seeddata{seed}',
        'cond': 0,
        'xflip': 0,
        'data': 'data/cifar/train',
        'fold_path': 'data/cifar/train/folds_10eq_seed{seed}.pk',
        'fold_id': 7,
        'fold_ticks': 0,
        'duration': 0.01, ### MImgs
        'tick': 4,
        'snap': 10 // (4*2),
        'dump': 2 * 10 // (4*2),
        'seed': 1000,
        'cache': 1,
        'workers': 4,
        'batch-gpu': 64,
        'augpipe': 'bgc' #bgc
    },
}

def os_cmd(cmd, env=None, stdout=subprocess.PIPE):
    with subprocess.Popen(cmd.split(), stdout=stdout, stderr=subprocess.STDOUT,
        encoding='latin-1', env=env) as p:
        out, err = p.communicate()
    if err:
        print(f'\n>>> ERROR: The process raised an error: {err.decode()}\n')
    return out

def branch_dict(in_dict):
    '''
    For any list or tuple in in_dict, will branch into all combination.
    Returns a list of dict.
    '''
    collect = list()
    for key, val in in_dict.items():
        if len(collect) == 0:
            if isinstance(val, (list, tuple)):
                collect.extend([{key: sub_val} for sub_val in val])
            else:
                collect.append({key: val})
        else:
            new_collect = list()
            for cdict in collect:
                if isinstance(val, (list, tuple)):
                    for sub_val in val:
                        new_collect.append(dict(**cdict, **{key: sub_val}))
                else:
                    new_collect.append(dict(**cdict, **{key: val}))
            collect = new_collect
    return collect

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Root results directory.')
    parser.add_argument('--config', nargs='*', help=f'Choose from {list(EXP_CONFIGS.keys())}.')
    parser.add_argument('--single_thread', action='store_true', help='Serialize seeds on the same run.')
    parser.add_argument('--save_local', action='store_true', help='Saves on the local machine and copies back.')
    parser.add_argument('--bash', action='store_true', help='Runs with bash instead of sbatch.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use, will be ignored if bash is specified.')
    parser.add_argument('--device', help='The gpu ids to use, e.g. 0,1,2 will be ignored if bash is not specified.')
    parser.add_argument('--large_gpu', action='store_true', help='SBATCH option: use large gpu setting.')
    return parser.parse_args()

if __name__ == '__main__':
    args = setup_args()
    sbatch_str = SBATCH_STR_LARGE if args.large_gpu else SBATCH_STR
    for config_name in args.config:
        print(f'\n>>> Spawning {config_name}')
        config_base = EXP_CONFIGS[config_name]
        
        ### Branch config into a list of configs
        configs = branch_dict(config_base)

        cmd_list = list()
        job_names = list()
        job_save_dirs = list()
        ### Convert into command string
        for config in configs:
            for key, val in config.items():
                if isinstance(val, str):
                    config[key] = val.format(**config)
            config_save_dir = config['outdir']
            temp_save_dir = (os.path.join('$TMPDIR/logs_temp', config_save_dir)
                    if args.save_local else os.path.join(args.save_dir, config_save_dir))
            config['outdir'] = temp_save_dir
            config_str = ' '.join([config['cmd']] + [f'--{key} {val}' for key, val in config.items() if key != 'cmd'])
            if args.save_local:
                config_str = f'{config_str}\n{COPY_STR.format(temp_dir=temp_save_dir, save_dir=args.save_dir)}\n{DEL_STR.format(temp_dir=temp_save_dir)}'
            cmd_list.append(config_str)
            job_names.append(config_save_dir.replace('/', '_'))
            job_save_dirs.append(os.path.join(args.save_dir, config_save_dir))
            os.makedirs(job_save_dirs[-1], exist_ok=True)

        ### Serialize the runs of the seeds if requires
        if args.single_thread:
            cmd_list = ['\n'.join(cmd_list)]
            job_names = f'{config_name}_single_thread'
            
        ### Run
        for cmd_str, job_name, job_save_dir in zip(cmd_list, job_names, job_save_dirs):
            if args.bash:
                ### Save sbatch to file
                bash_file_path = os.path.join(job_save_dir, f'bash_{job_name}.sh')
                set_device_str = f'export CUDA_VISIBLE_DEVICES={args.device}\n' if args.device is not None else ''
                with open(bash_file_path, 'w+') as fs:
                    print('#!/bin/bash\n'+set_device_str+f'export TORCH_EXTENSIONS_DIR={job_save_dir}\n'+cmd_str, file=fs)
                    fs.flush()
                    os.fsync(fs)

                    ### Run the experiment
                    os_cmd(f'bash {bash_file_path}', stdout=sys.stdout)
            else:
                ### Save sbatch to file
                sbatch_cmd = sbatch_str.format(device=args.num_gpus, job_name=job_name, save_dir=job_save_dir)
                sbatch_file = sbatch_cmd+'\n'+cmd_str
                sbatch_file_path = os.path.join(job_save_dir, f'sbatch_{job_name}.sh')
                with open(sbatch_file_path, 'w+') as fs:
                    print(sbatch_file, file=fs)
                    fs.flush()
                    os.fsync(fs)

                    ### Run the experiment
                    os_cmd(f'sbatch {sbatch_file_path}', stdout=sys.stdout)
