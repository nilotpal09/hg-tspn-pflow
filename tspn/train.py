import os
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6


import comet_ml

import dgl
import dgl.function as fn
from dgl import DGLGraph as DGLGraph

from lightning import PflowLightning
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer, profiler

from dataloader import PflowDataset, collate_graphs
import sys
import json
from models.pflow_model import PflowModel
import glob

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'


if __name__ == "__main__":
    config_path = sys.argv[1]

    if len(sys.argv) == 3:
        debug_mode = sys.argv[2]
    else:
        debug_mode = '0'

    with open(config_path, 'r') as fp:
         config = json.load(fp)

    net = PflowLightning(config)

    if debug_mode == '1':

        #profiler = profiler.PyTorchProfiler(filename='/srv01/agrp/nilotpal/projects/SCD/SCD/particle_flow/experiments/profiler_output')
        trainer = Trainer(
            max_epochs=config['num_epochs'],
            gpus=1,
            default_root_dir='/storage/agrp/nilotpal/PFlow/experiments/Set2Set/',
            # resume_from_checkpoint='/storage/agrp/dreyet/PFlow/SCD/particle_flow/experiments/Set2Set/pflow-tspn-all/1f130f02791e42819b16a4c4016ec20a/checkpoints/epoch=289-step=106139.ckpt'
        ) # ,gradient_clip_val=0.1,log_every_n_steps=2,profiler="simple")

    else:
        comet_logger = CometLogger(
            api_key='b7a7KdMattTDdfmhK5egXa1RE',
            save_dir='/storage/agrp/nilotpal/PFlow/experiments/Set2Set/',
            project_name="pflow-tspn-all", 
            workspace="nilotpal09",
            experiment_name=config['name']
        )

        net.set_comet_exp(comet_logger.experiment)
        comet_logger.experiment.log_asset(config_path,file_name='config')

        all_files = glob.glob('./*.py')+glob.glob('models/*.py')
        for fpath in all_files:
            comet_logger.experiment.log_asset(fpath)
        print('creating trainer') #config['num_epochs']
        trainer = Trainer(
            max_epochs = config['num_epochs'],
            gpus = 1,
            default_root_dir = '/storage/agrp/nilotpal/PFlow/experiments/Set2Set/',
            logger = comet_logger,
            # resume_from_checkpoint = '/storage/agrp/dreyet/PFlow/SCD/particle_flow/experiments/Set2Set/pflow-tspn-all/1f130f02791e42819b16a4c4016ec20a/checkpoints/epoch=289-step=106139.ckpt'
        ) # ,gradient_clip_val=0.1,log_every_n_steps=2,profiler="simple")
    
    trainer.fit(net)



