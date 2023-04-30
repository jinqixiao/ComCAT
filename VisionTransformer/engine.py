"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

Align_Features_Out_Hook = {}
Module_Features_Out_Hook = {}
Hook_Mid_Output = False
mse = torch.nn.MSELoss()
Loss_Alpha = 0.5
# blocks_no = [0, 1, 2, 9, 10, 11]
blocks_no = [0, 1, -2, -1]
Use_Manifold_Distill = False


def turn_off_hook():
    global Hook_Mid_Output
    Hook_Mid_Output = False


def turn_on_hook():
    global Hook_Mid_Output
    Hook_Mid_Output = True


def align_hook(module, fea_in, fea_out):
    global Hook_Mid_Output
    if not Hook_Mid_Output:
        return
    global Align_Features_Out_Hook
    # print('Align:', module.name + str(fea_in[0].device))
    Align_Features_Out_Hook[module.name + str(fea_in[0].device)] = fea_out
    return None


def module_hook(module, fea_in, fea_out):
    global Hook_Mid_Output
    if not Hook_Mid_Output:
        return
    global Module_Features_Out_Hook
    # print('Module:', module.name + str(fea_in[0].device))
    Module_Features_Out_Hook[module.name + str(fea_in[0].device)] = fea_out
    return None


def add_hook_to_models(model, align_model):
    global Hook_Mid_Output
    for n in blocks_no:
        if align_model is not None:
            block = align_model.blocks[n]
            block.register_forward_hook(hook=align_hook)
            block.name = 'block_' + str(n) + '_'
        if model is not None:
            block = model.blocks[n]
            block.register_forward_hook(hook=module_hook)
            block.name = 'block_' + str(n) + '_'


def tensor_tensor_t(t):
    return t @ t.T


def append_align_loss(align_model, x, loss):
    global Align_Features_Out_Hook
    global Module_Features_Out_Hook
    global Loss_Alpha
    global Use_Manifold_Distill
    align_model(x)
    module_outputs = Module_Features_Out_Hook
    align_outputs = Align_Features_Out_Hook
    # for k in self.align_layer_key:
    #     k = k + str(x.device)
    #     print(nas_outputs[k].shape, align_outputs[k].shape)
    # print('loss:', loss.item())
    for n in blocks_no:
        if hasattr(align_model, 'module'):
            block = align_model.module.blocks[n]
        else:
            block = align_model.blocks[n]
        k = block.name + str(x.device)
        if not Use_Manifold_Distill:
            if k in module_outputs and k in align_outputs:
                loss += mse(module_outputs[k], align_outputs[k]) * Loss_Alpha
        else:
            # manifold loss
            shape = module_outputs[k].shape
            # intra
            intra_loss = 0
            for i in range(shape[0]):
                intra_loss += torch.norm(
                    tensor_tensor_t(module_outputs[k][i, :, :]) - tensor_tensor_t(align_outputs[k][i, :, :]), p=2)
            intra_loss = intra_loss / shape[0]
            # inter
            inter_loss = 0
            for i in range(shape[1]):
                inter_loss += torch.norm(
                    tensor_tensor_t(module_outputs[k][:, i, :]) - tensor_tensor_t(align_outputs[k][:, i, :]), p=2)
            inter_loss = inter_loss / shape[1]
            # random
            i = torch.randint(0, shape[0], [1])
            j = torch.randint(0, shape[1], [1])
            random_loss = torch.norm(
                tensor_tensor_t(module_outputs[k][i, j, :]) - tensor_tensor_t(align_outputs[k][i, j, :]), p=2)
            loss += (intra_loss + inter_loss + random_loss) * Loss_Alpha

    Align_Features_Out_Hook = {}
    Module_Features_Out_Hook = {}
    return loss


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None, align_model=None, alpha=1):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if align_model is not None:
        global Module_Features_Out_Hook
        global Align_Features_Out_Hook
        global Loss_Alpha
        Loss_Alpha = max(Loss_Alpha * alpha, 0.01)
        print('alpha:', Loss_Alpha)
        Module_Features_Out_Hook = {}
        Align_Features_Out_Hook = {}
        turn_on_hook()

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
            if align_model is not None:
                metric_logger.update(loss_ori=loss.item())
                append_align_loss(align_model, samples, loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def search_rank(data_loader, model, device, rank_optimizer, target_params, beta=1.5):
    print("Searching Ranks:")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Search Rank:'
    # beta = 1.5
    model.train()
    # import random
    # random_iter_set = set(random.sample(list(range(0, 6250+1)), 500))
    # i = -1
    for images, target in metric_logger.log_every(data_loader, 50, header):
        # i += 1
        # if i not in random_iter_set:
        #     continue
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if hasattr(model, 'module'):
                output, params = model.module.search_rank_forward(images)
            else:
                output, params = model.search_rank_forward(images)
            loss = criterion(output, target)
            if params > target_params:
                loss *= ((params / target_params) ** beta)

        rank_optimizer.zero_grad()
        loss.backward()
        rank_optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    torch.cuda.synchronize()
    print('flops, target_flops', int(params), int(target_params))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
