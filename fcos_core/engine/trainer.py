# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import datetime
import logging
import time
import collections
import torch
import torch.distributed as dist
from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger
from fcos_core.utils.comm import get_rank
from fcos_core.engine.inference import run_test

def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)
def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    arguments,
    args,
):
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    max_AP = 0.5
    min_Loss = 2
    min_detectLoss = 1
    min_maskLoss = 0.5
    start_iter = arguments["iteration"]
    ######Loss hyperparameter######
    a = 1
    b = 1

    save_to_disk = get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        import tensorboardX

        summary_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    for iteration, batch in enumerate(data_loader, start_iter):
        model.train()
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = batch[0].to(device)
        targets = [target.to(device) for target in batch[1]]

        if cfg.MODEL.DDTNET_ON:
            gt = [gt.to(device) for gt in batch[2]]
            contour = [contour.to(device) for contour in batch[3]]
            center = [center.to(device) for center in batch[5]]
            loss_dict = model(images, targets, gt, contour, center)
        else:
            loss_dict = model(images, targets)


        # losses = sum(loss for loss in loss_dict.values())
        dec_loss = loss_dict.get('loss_cls') + loss_dict.get('loss_reg') + loss_dict.get('loss_centerness')
        seg_loss = loss_dict.get('loss_mask')
        losses = a * dec_loss + b * seg_loss

        ## reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        dec_losses_reduced = loss_dict_reduced.get('loss_cls') + loss_dict_reduced.get('loss_reg') + loss_dict_reduced.get('loss_centerness')
        seg_losses_reduced = loss_dict_reduced.get('loss_mask')
        losses_reduced = a * dec_losses_reduced + b * seg_losses_reduced
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % args.log_step == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        # if iteration % args.save_step == 0:
        #     checkpointer.save("model_{:07d}".format(iteration), **arguments)

        if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
            eval_results = run_test(cfg, model, distributed=args.distributed,ovthresh=args.ovthresh)
            if get_rank() == 0 and summary_writer:
                for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                    write_metric(eval_result['metrics1'], 'metrics1/' + dataset, summary_writer, iteration)
                    write_metric(eval_result['metrics2'], 'metrics2/' + dataset, summary_writer, iteration)
                f1 = eval_result['metrics1'].get('f1')
                if f1 > max_AP :
                    checkpointer.save("F1model_{:07d}".format(iteration), **arguments)
                    max_AP = f1
                Loss = eval_result['metrics2'].get('loss')
                if Loss < min_Loss :
                    checkpointer.save("Lossmodel_{:07d}".format(iteration), **arguments)
                    min_Loss = Loss

                if cfg.MODEL.DDTNET_ON:
                    detectLoss = eval_result['metrics2'].get('loss') - eval_result['metrics2'].get('loss_mask')
                    if detectLoss < min_detectLoss :
                        checkpointer.save("DetectLossmodel_{:07d}".format(iteration), **arguments)
                        min_detectLoss = detectLoss
                    maskLoss = eval_result['metrics2'].get('loss_mask')
                    if maskLoss < min_maskLoss :
                        checkpointer.save("MaskLossmodel_{:07d}".format(iteration), **arguments)
                        min_maskLoss = maskLoss


    checkpointer.save("model_final", **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
