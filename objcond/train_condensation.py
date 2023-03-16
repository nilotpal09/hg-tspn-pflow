import comet_ml

import dgl
import dgl.function as fn
from dgl import DGLGraph as DGLGraph

from lightning import PflowLightning
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer, profiler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataloader import PflowDataset, collate_graphs
import sys
import json
from models.pflow_model import PflowModel
import glob
import argparse
from datetime import date

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def move_to_output_dir(output_dir,config_path):

    print('moving to ',output_dir)
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for subdir in ['scatter','checkpoints','models','configs']:
        subdir = output_dir + subdir
        if not os.path.exists(subdir): os.mkdir(subdir)

    commands = []
    commands.append('cp *.py {}'.format(output_dir))
    commands.append('cp models/*.py {}/models'.format(output_dir))
    commands.append('cp {} {}/configs/condensation.json'.format(config_path,output_dir))
    #commands.append('cd {}'.format(output_dir))

    for com in commands:
        os.system(com)


def train(args, output_dir):

    with open(output_dir+'configs/condensation.json', 'r') as fp:
         config = json.load(fp)

    net = PflowLightning(config)

    if args.debug:

        profiler = profiler.PyTorchProfiler(filename='/srv01/agrp/dreyet/PFlow/SCD/particle_flow/experiments/profiler_output')
        trainer = Trainer(
            max_epochs = config['num_epochs'],
            gpus = 1,
            default_root_dir = '/storage/agrp/dreyet/PFlow/SCD/particle_flow/experiments/' + args.tag + '/',
            # resume_from_checkpoint = '/storage/agrp/nilotpal/PFlow/experiments/Set2Set/pflow-tspn-unsup-track/bfedfa3946da40efad2d2676745653b9/checkpoints/epoch=99-step=9399.ckpt'
        )#, profiler=profiler) # ,gradient_clip_val=0.1,log_every_n_steps=2,profiler="simple")

    else:
        comet_logger = CometLogger(
            api_key='KFlUvC17ueYXdYjnZHBRpa9Bm',
            save_dir=output_dir,
            project_name="condensation_comet", 
            #workspace="",
            experiment_name=config['name'] + args.tag
        )

        net.set_comet_exp(comet_logger.experiment)
        comet_logger.experiment.log_asset(args.config,file_name='config')

        all_files = glob.glob('./*.py')+glob.glob('models/*.py')
        for fpath in all_files:
            comet_logger.experiment.log_asset(fpath)

        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir + '/checkpoints',
            save_top_k=-1, #save all of them!
            filename='condensation-{epoch:02d}',#-{step:02d}',
            every_n_epochs=args.every_n_epochs,
            #every_n_train_steps=1,
        )

        early_stop_callback = EarlyStopping(monitor="node loss", check_finite=True, verbose=True, min_delta=1e-8, patience=100, mode="min")

        print('creating trainer') #config['num_epochs']
        if args.checkpoint=='':
            trainer = Trainer(
                max_epochs = config['num_epochs'],
                gpus = 1,
                default_root_dir = output_dir,
                logger = comet_logger,
                log_every_n_steps=1,
                callbacks=[checkpoint_callback],#,early_stop_callback],
            ) # ,gradient_clip_val=0.1,log_every_n_steps=2,profiler="simple")
        else:
            trainer = Trainer(
                max_epochs = config['num_epochs'],
                gpus = 1,
                default_root_dir = output_dir,
                logger = comet_logger,
                log_every_n_steps=1,
                callbacks=[checkpoint_callback],#,early_stop_callback],
                resume_from_checkpoint = args.checkpoint
            ) # ,gradient_clip_val=0.1,log_every_n_steps=2,profiler="simple")

    trainer.fit(net)


if __name__ == "__main__":

    hayom = date.today().strftime("%d%m%Y")

    parser = argparse.ArgumentParser(description='Run condensation training')
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument('--checkpoint', dest='checkpoint', required=False)
    parser.add_argument('--tag', dest='tag',default=hayom)
    parser.add_argument('--every_n_epochs', dest='every_n_epochs', type=int, default=5)
    parser.add_argument('--debug', dest='debug',type=bool,default=False)

    args = parser.parse_args()

    output_dir = '/storage/agrp/dreyet/PFlow/SCD/particle_flow/experiments/' + args.tag + '/'

    if not os.getcwd()+'/' == output_dir:
        move_to_output_dir(output_dir,args.config)
        jobname   = "run_{}.sh".format(args.tag)
        runscript = open(output_dir + jobname, "w")
        command   = '/usr/local/anaconda/3.8/bin/python'
        for seg in sys.argv: command = command + ' ' + seg
        runscript.write(command)
        runscript.close()
        print('please go to {} and run {}'.format(output_dir,jobname))
    else:
        train(args,output_dir)
