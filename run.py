import os

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from meta_learner import MetaLearner
from args_parser import args_parser
from explight import initialize_exp, set_seed, get_dump_path, describe_model, save_model
import logging
from tqdm import tqdm
import time
logger = logging.getLogger()



class Runner:
    def __init__(self, args, logger_path, writer):
        self.args = args
        self.writer = writer
        self.meta_learner = MetaLearner(args)
        describe_model(self.meta_learner.maml.module, logger_path, name='model')
        describe_model(self.meta_learner.task_selector, logger_path, name='task_selector')
        self.logger_path = logger_path

    def run(self):
        best_score = -1
        pbar = tqdm(range(1, self.args.episode + 1))
        cost_time_ls = []

        for epoch in pbar:
            start = time.time()
            loss_cls = self.meta_learner.train_step(epoch)
            cost_time = time.time() - start
            cost_time_ls.append(cost_time)
            self.writer.add_scalar('loss-cls', loss_cls, epoch)

            pbar.set_description(f"loss={loss_cls:.4f}")
            if epoch % self.args.eval_step == 0:
                score = self.meta_learner.test_step()
                if score > best_score:
                    best_score = score
                logger.info(f"{epoch} | score: {score:.5f}, best_score: {best_score:.5f}")
                self.writer.add_scalars('score', {'score': score, 'best': best_score}, epoch)
        logger.info(f"best score: {best_score:.5f}")
        logger.info(f"time cost: {np.mean(cost_time_ls):.5f}s")



def main():
    args = args_parser()
    torch.cuda.set_device(int(args.gpu))
    set_seed(args.random_seed)
    logger = initialize_exp(args)
    logger_path = get_dump_path(args)
    writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    runner = Runner(args, logger_path, writer)
    runner.run()
    writer.close()


if __name__ == '__main__':
    main()
