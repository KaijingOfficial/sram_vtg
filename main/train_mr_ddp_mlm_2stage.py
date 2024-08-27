import os
import pdb
import sys
sys.path.append('.')
import time
import json
import pprint
import random
import numpy as np
from tqdm import trange
from tqdm.auto import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from main.config import BaseOptions, setup_model, setup_model_mlm, setup_model_grounding
from main.dataset_mlm import \
    DatasetMR, start_end_collate_mr, prepare_batch_inputs_mr, prepare_batch_inputs_mr_mlm
from main.inference_mr_mlm import eval_epoch, start_inference
from utils.basic_utils import set_seed, AverageMeter, dict_to_markdown,save_json
from utils.model_utils import count_parameters
from transformers import CLIPTokenizerFast
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)



def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer, mlm):
    prefix = 'Mlm' if mlm else 'Grounding'
    logger.info(f"[{prefix} Epoch {epoch_i+1}]") if (opt.local_rank in [0,-1]) else None
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                desc="Training Iteration",
                                total=num_training_examples,
                                position=0,
                                disable=not (opt.local_rank in [0,-1]),
                                colour="#2E8B57" if mlm else "BLUE",
                                dynamic_ncols=True):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        model_inputs, targets = prepare_batch_inputs_mr_mlm(batch[1], torch.device("cuda", int(opt.local_rank)), non_blocking=opt.pin_memory, mlm=mlm)
        
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)

        timer_start = time.time()

        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)    
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()

        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        timer_dataloading = time.time()

    # print/add logs
    if int(opt.local_rank) in [0, -1]:
        tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

        to_write = opt.train_log_txt_formatter.format(
            time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
            epoch=epoch_i+1,
            loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)

        logger.info("Epoch time stats:")
        for name, meter in time_meters.items():
            d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
            logger.info(f"{name} ==> {d}")


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt, mlm):
    prefix = 'mlm' if mlm else 'grounding'
    n_epoch = opt.mlm_epoch if mlm else opt.n_epoch
    if int(opt.local_rank) in [0, -1]:   

        opt.tensorboard_log_dir = opt.tensorboard_log_dir + '_' + prefix
        tb_writer = SummaryWriter(opt.tensorboard_log_dir)
        tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
        opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
        opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    else:
        tb_writer = None
    batch_size = opt.bsz if not mlm else opt.mlm_bsz
    # print(batch_size)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate_mr,
        batch_size=batch_size,
        num_workers=opt.num_workers,
        #shuffle=True,
        pin_memory=opt.pin_memory,
        sampler=DistributedSampler(train_dataset)
    )

    prev_best_score = 0.
    es_cnt = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_init else 0
    else:
        start_epoch = opt.start_epoch

    
    for epoch_i in trange(start_epoch, n_epoch, desc="Epoch",disable=not (opt.local_rank in [-1,0]), colour='#7FFFAA' if mlm else 'BLUE',dynamic_ncols=True):
        save_submission_filename = f"epoch{epoch_i}_{prefix}_{opt.dset_name}_{opt.eval_split_name}_preds.jsonl"
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer, mlm)
            lr_scheduler.step()
        eval_epoch_interval = opt.eval_epoch
        if int(opt.local_rank) in [0, -1] and opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer,skip_mlm=True)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))
            if int(opt.local_rank) in [0, -1]:
                with open(opt.eval_log_filepath, "a") as f:
                    f.write(to_write)
                logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
                if metrics_nms is not None:
                    logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

                metrics = metrics_nms if metrics_nms is not None else metrics_no_nms
                for k, v in metrics["brief"].items():
                    tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            # stop_score = metrics["brief"]["MR-full-mAP"]
            stop_score = metrics["brief"][opt.main_metric]
            if stop_score > prev_best_score:

                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    # "optimizer": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    # "opt": opt
                }
                import re
                for filename in os.listdir(opt.results_dir):
                    if re.search('_best.ckpt', filename):
                        os.remove(os.path.join(opt.results_dir,filename))
                        break
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i}_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                # "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        if int(opt.local_rank) in [0, -1] and ((epoch_i + 1) % opt.save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0):  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                # "opt": opt
            }
            # if opt.dense_save != -1:
            #     pass
                # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break
    if int(opt.local_rank) in [0, -1]:
        tb_writer.close()
    return model


def start_training():
    opt = BaseOptions().parse()
    logger.info("Setup config, data and model...") if (opt.local_rank in [0,-1]) else None
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True
    local_rank = int(opt.local_rank)
    dist.init_process_group(backend='nccl')



    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    tokenizer = CLIPTokenizerFast.from_pretrained(opt.tokenizer_path)
    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        v_feat_dim=opt.v_feat_dim,
        q_feat_dim=opt.t_feat_dim,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio,
        use_cache=opt.use_cache,
        add_easy_negative=opt.add_easy_negative,
        easy_negative_only=opt.easy_negative_only,
        tokenizer=tokenizer,
        mlm_ratio=opt.mlm_ratio,
    )

    dataset_config["data_path"] = opt.train_path
    train_mlm_dataset = DatasetMR(**dataset_config)
    dataset_config["mlm_ratio"] = None
    train_grounding_dataset = DatasetMR(**dataset_config)
    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        if len(dataset_config["v_feat_dirs"]) == 1:
            dataset_config["v_feat_dirs"] = [f"data/{opt.dset_name}/vid_clip"]
        elif len(dataset_config["v_feat_dirs"]) == 2:
            dataset_config["v_feat_dirs"] = [f"data/{opt.dset_name}/vid_slowfast",
                                            f"data/{opt.dset_name}/vid_clip"]
        else:
            raise NotImplementedError
        dataset_config["q_feat_dir"] = f"data/{opt.dset_name}/txt_clip"
        dataset_config["data_ratio"] = 1
        dataset_config["mlm_ratio"] = None
        eval_dataset = DatasetMR(**dataset_config)
    else:
        eval_dataset = None
        
    if opt.mlm_warmup_steps > 0:
        # total_steps = opt.n_epoch * len(train_dataset) // opt.bsz
        mlm_total_steps = opt.mlm_epoch
        mlm_warmup_steps = opt.mlm_warmup_steps if opt.mlm_warmup_steps > 1 else int(opt.mlm_warmup_steps * mlm_total_steps)
        opt.mlm_warmup_steps = [mlm_warmup_steps, mlm_total_steps]
        
    if opt.lr_warmup > 0:
        # total_steps = opt.n_epoch * len(train_dataset) // opt.bsz
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]
        
    model, criterion, optimizer, lr_scheduler = setup_model_mlm(opt)
    
    opt.trainable_params, opt.all_params =  count_parameters(model)
    print("Criterion:")
    count_parameters(criterion)
    option_file_path = os.path.join(opt.results_dir,'opt.json')
    save_json(vars(opt),option_file_path,save_pretty=True,sort_keys=True)

    model.to(device)
    criterion.to(device)
    logger.info(f"Using {torch.cuda.device_count()} GPUs.")
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True)

    if int(opt.local_rank) in [0, -1]:
        logger.info(f"Model {model}")
        count_parameters(model)
        logger.info("Start Training...")
        
    model_stage1 = train(model, criterion, optimizer, lr_scheduler, train_mlm_dataset, eval_dataset, opt, mlm=True)
    

        
    model, criterion, optimizer, lr_scheduler = setup_model_grounding(opt,pretrained_model=model_stage1.module)

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True)
    
    train(model, criterion, optimizer, lr_scheduler, train_grounding_dataset, eval_dataset, opt, mlm=False)
    return 

if __name__ == '__main__':

    # import debugpy
    # debugpy.listen(("localhost", 12457))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()

    # best_ckpt_path, eval_split_name, eval_path, debug = start_training()
    start_training()
    # if not debug:
    #     input_args = ["--resume", best_ckpt_path,
    #                   "--eval_split_name", eval_split_name,
    #                   "--eval_path", eval_path]

    #     import sys
    #     sys.argv[1:] = input_args
    #     logger.info("\n\n\nFINISHED TRAINING!!!")
    #     logger.info("Evaluating model at {}".format(best_ckpt_path))
    #     logger.info("Input args {}".format(sys.argv[1:]))
    #     start_inference()
