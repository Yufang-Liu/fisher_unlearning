import copy, os
import numpy as np
from itertools import chain
from tqdm.autonotebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients
from datasets.TinyImageNet import TinyImageNet
from model.DenseNet import DenseNet
from model.GoogleNet import GoogLeNet

criterions = {}


def _add_criterions(model_fn):
    criterions[model_fn.__name__] = model_fn
    return model_fn


def get_named_layers(net, is_state_dict=True):
    conv2d_idx = 0
    convT2d_idx = 0
    linear_idx = 0
    batchnorm2d_idx = 0
    named_layers = []
    for mod in net.modules():
        if isinstance(mod, torch.nn.Conv2d):
            layer_name = 'Conv2d{}_{}-{}'.format(
                conv2d_idx, mod.in_channels, mod.out_channels
            )
            named_layers.append(layer_name)
            if mod.bias is not None:
                named_layers.append(layer_name + '_bias')
            conv2d_idx += 1
        elif isinstance(mod, torch.nn.ConvTranspose2d):
            layer_name = 'ConvT2d{}_{}-{}'.format(
                conv2d_idx, mod.in_channels, mod.out_channels
            )
            named_layers.append(layer_name)
            if hasattr(mod, "bias"):
                named_layers.append(layer_name + '_bias')
            convT2d_idx += 1
        elif isinstance(mod, torch.nn.BatchNorm2d):
            layer_name = 'BatchNorm2D{}_{}'.format(
                batchnorm2d_idx, mod.num_features)
            named_layers.append(layer_name)
            named_layers.append(layer_name + '_bais')
            if is_state_dict:
                named_layers.append(layer_name + '_running_mean')
                named_layers.append(layer_name + '_running_var')
                named_layers.append(layer_name + '_num_bathes_tracked')
            batchnorm2d_idx += 1
        elif isinstance(mod, torch.nn.Linear):
            layer_name = 'Linear{}_{}-{}'.format(
                linear_idx, mod.in_features, mod.out_features
            )
            named_layers.append(layer_name)
            if hasattr(mod, "bias"):
                named_layers.append(layer_name + '_bias')
            linear_idx += 1
    return named_layers


def hessian(dataset, model):
    model.eval()
    #if isinstance(dataset, TinyImageNet) or isinstance(model, DenseNet) or isinstance(model, GoogLeNet):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    #else:
    #    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    for tp in model.parameters():
        tp.grad_acc = 0
        tp.grad2_acc = 0
    for data, orig_target in tqdm(train_loader):
        data, orig_target = data.cuda(), orig_target.cuda()
        output = model(data)
        prob = F.softmax(output, dim=-1).data
        for y in range(output.shape[1]):
            target = torch.empty_like(orig_target).fill_(y)
            loss = loss_fn(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for tp in model.parameters():
                if tp.requires_grad:
                    tp.grad_acc += torch.sum(orig_target == target).float()/len(orig_target) * tp.grad.data
                    tp.grad2_acc += prob[:, y].float().mean() * tp.grad.data.pow(2)
    model_grad = copy.deepcopy(model)
    for gp, tp in zip(model_grad.parameters(), model.parameters()):
        tp.grad_acc /= len(train_loader)
        tp.grad2_acc /= len(train_loader)
        gp.data = tp.grad2_acc
    return model_grad


def _add_criterions(model_fn):
    criterions[model_fn.__name__] = model_fn
    return model_fn


def get_model_layer(model, name):
    if name.find('.') == -1:
        if name == 'weight':
            return model
        else:
            return getattr(model, name)
    else:
        first_token = name.index('.')
        if name[:first_token].isdigit():
            return get_model_layer(model[int(name[:first_token])],
                                   name[first_token + 1:])
        else:
            return get_model_layer(getattr(model, name[:first_token]),
                                   name[first_token + 1:])


def get_separate_parameters(model, mask_index, total_num):
    named_layer = get_named_layers(model, is_state_dict=False)
    num, idx, flag = 0, 0, False
    mask_params = []
    for n, (k, v) in zip(named_layer, model.named_parameters()):
        if n.startswith('Conv2d') and not n.endswith('bias'):
            flag = False
            while idx < len(mask_index) and mask_index[idx] < num + v.size()[0]:
                if not flag:
                    mask_params += [v]
                    flag = True
                idx += 1
            if num < total_num:
                num += v.size()[0]
        elif n.startswith('Conv2d') and n.endswith('bias'):
            if flag:
                mask_params += [v]
        if n.startswith('BatchNorm') and not n.endswith('bias'):
            if flag:
                mask_params += [v]
        elif n.startswith('BatchNorm') and n.endswith('bias'):
            if flag:
                mask_params += [v]
                flag = False
        if n.startswith('Linear') and not n.endswith('bias'):
            flag = False
            while idx < len(mask_index) and mask_index[idx] < num + v.size()[0]:
                if not flag:
                    mask_params += [v]
                    flag = True
                idx += 1
            if num < total_num:
                num += v.size()[0]
        elif n.startswith('Linear') and n.endswith('bias'):
            if flag:
                mask_params += [v]
                flag = False

    assert num == total_num
    params_id = list(map(id, mask_params))
    all_params = model.parameters()
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    return mask_params, other_params


class Unlearn:
    def __init__(self, remain_loader, forget_loader):
        self.remain_loader = remain_loader
        self.forget_loader = forget_loader

    def mask_activation(self, **kwargs):
        model = kwargs['model']
        remove_ratio = kwargs['remove_ratio']
        largest = kwargs['largest']
        model.eval()
        cfg = kwargs['cfg']
        args = kwargs['args']

        if hasattr(args, "random_exper"):
            ckpt_dir = cfg.ckpt_dir + '/random' + str(args.forget_num)
        elif hasattr(args, "experiment"):
            ckpt_dir = cfg.ckpt_dir + '/' + args.experiment
        else:
            ckpt_dir = cfg.ckpt_dir

        file_name = "{}/mask_{}_{}.npy".format(ckpt_dir,
                                       args.forget_type,
                                        args.forget_num)
        if os.path.isfile(file_name):
            instance_info = np.load(file_name)
        else:
            data_loader = torch.utils.data.DataLoader(self.forget_loader.dataset,
                                                      batch_size=128, shuffle=False)
            count_info = None
            for batch in tqdm(data_loader):
                batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
                data, target = batch
                output_list = model.get_sequential_output(data)
                if count_info is None:
                    count_info = [np.array([]) for _ in range(len(output_list) - 1)]
                for i in range(len(output_list) - 1):
                    if len(output_list[i].size()) == 4:   # conv layer
                        size = output_list[i].size()[-1] * output_list[i].size()[-2]
                        sum_conv = torch.sum(torch.sum(output_list[i], -1), -1) / size
                        sum_conv = torch.sum(sum_conv, dim=0).cpu().detach().numpy()
                    elif len(output_list[i].size()) == 2:  # linear layer
                        sum_conv = torch.sum(output_list[i], dim=0).cpu().detach().numpy()
                    else:
                        print("output size {}".format(output_list[i].size()))
                        exit(1)
                    if len(count_info[i]) == 0:
                        count_info[i] = sum_conv
                    else:
                        count_info[i] += sum_conv
            count_info = list(chain.from_iterable(count_info))
            count_info = [item / len(data_loader) for item in count_info]

            if self.remain_loader is not None:
                train_info = None
                data_loader = torch.utils.data.DataLoader(self.remain_loader.dataset,
                                                          batch_size=128, shuffle=False)
                for batch in tqdm(data_loader):
                    batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
                    data, target = batch
                    output_list = model.get_sequential_output(data)
                    if train_info is None:
                        train_info = [np.array([]) for _ in range(len(output_list) - 1)]
                    for i in range(len(output_list) - 1):
                        if len(output_list[i].size()) == 4:   # conv layer
                            size = output_list[i].size()[-1] * output_list[i].size()[-2]
                            sum_conv = torch.sum(torch.sum(output_list[i], -1), -1) / size
                            sum_conv = torch.sum(sum_conv, dim=0).cpu().detach().numpy()
                        elif len(output_list[i].size()) == 2:  # linear layer
                            sum_conv = torch.sum(output_list[i], dim=0).cpu().detach().numpy()
                        else:
                            print("output size {}".format(output_list[i].size()))
                            exit(1)
                        if len(train_info[i]) == 0:
                            train_info[i] = sum_conv
                        else:
                            train_info[i] += sum_conv

                train_info = list(chain.from_iterable(train_info))
                train_info = [item / len(data_loader) for item in train_info]

            if self.remain_loader is not None:
                instance_info = np.array(count_info)-np.array(train_info)
            else:
                instance_info = np.array(count_info)
            np.save(file_name, instance_info)
        count_info = np.argsort(instance_info)
        #instance_info = list(instance_info)
        #instance_info.sort(reverse=True)
        #print([(idx, item) for idx, item in enumerate(instance_info)])
        print("total neuron number :{}, mask: {}".format(
            len(count_info), int(len(count_info)*remove_ratio)))
        if largest:
            mask_index = count_info[-int(len(count_info) * remove_ratio):]
        else:
            mask_index = count_info[:int(len(count_info) * remove_ratio)]
        mask_index.sort()

        named_layer = get_named_layers(model)
        state_dict = copy.deepcopy(model.state_dict())
        num, idx, temp_idx = 0, 0, []
        for n, (k, v) in zip(named_layer, state_dict.items()):
            if n.startswith('Conv2d') and not n.endswith('bias'):
                temp_idx = []
                while idx < len(mask_index) and mask_index[idx] < num + v.size()[0]:
                    state_dict[k][mask_index[idx] - num] = 0.0
                    temp_idx.append(mask_index[idx] - num)
                    idx += 1
                if num < len(count_info):
                    num += v.size()[0]
            elif n.startswith('Conv2d') and n.endswith('bias'):
                for t in range(len(temp_idx)):
                    state_dict[k][temp_idx[t]] = 0.0
            if n.startswith('BatchNorm') and not n.endswith('num_bathes_tracked'):
                for t in range(len(temp_idx)):
                    state_dict[k][temp_idx[t]] = 0.0
            elif n.startswith('BatchNorm') and n.endswith('num_bathes_tracked'):
                temp_idx = []

            if n.startswith('Linear') and not n.endswith('bias'):
                temp_idx = []
                while idx < len(mask_index) and mask_index[idx] < num + v.size()[0]:
                    state_dict[k][mask_index[idx] - num] = 0.0
                    temp_idx.append(mask_index[idx] - num)
                    idx += 1
                if num < len(count_info):
                    num += v.size()[0]
            elif n.startswith('Linear') and n.endswith('bias'):
                for t in range(len(temp_idx)):
                    state_dict[k][temp_idx[t]] = 0.0
                temp_idx = []
        assert num == len(count_info)
        new_model = copy.deepcopy(model)
        new_model.load_state_dict(state_dict)
        new_model.cuda()
        return new_model, mask_index, len(count_info)

    def mask_fisher(self, **kwargs):
        model = kwargs['model']
        remove_ratio = kwargs['remove_ratio']
        largest = kwargs['largest']
        cfg = kwargs['cfg']
        args = kwargs['args']

        if hasattr(args, "random_exper"):
            ckpt_dir = cfg.ckpt_dir + '/random' + str(args.forget_num)
        elif hasattr(args, "experiment"):
            ckpt_dir = cfg.ckpt_dir + '/' + args.experiment
        else:
            ckpt_dir = cfg.ckpt_dir

        if remove_ratio == 0:
            return model, None, None
        forget_hessian_file = "{}/model_hessian_forget_{}_{}.pt".format(ckpt_dir,
                                                                       args.forget_type,
                                                                       args.forget_num)
        if os.path.isfile(forget_hessian_file):
            forget_grad = copy.deepcopy(model)
            forget_grad.load_state_dict(torch.load(forget_hessian_file))
        else:
            forget_grad = hessian(self.forget_loader.dataset, model)
            torch.save(forget_grad.state_dict(), forget_hessian_file)

        named_layer = get_named_layers(model, is_state_dict=False)

        if self.remain_loader is not None:
            remain_hessian_file = "{}/model_hessian_remain_{}_{}.pt".format(ckpt_dir,
                                                                            args.forget_type,
                                                                            args.forget_num)
            if os.path.isfile(remain_hessian_file):
                remain_grad = copy.deepcopy(model)
                remain_grad.load_state_dict(torch.load(remain_hessian_file))
            else:
                remain_grad = hessian(self.remain_loader.dataset, model)
                torch.save(remain_grad.state_dict(), remain_hessian_file)

            count_info = []
            for n, (k1, v1), (k2, v2) in zip(named_layer, forget_grad.named_parameters(),
                                             remain_grad.named_parameters()):
                if n.startswith('Conv2d') and not n.endswith('bias'):
                    activation = v1.data - v2.data
                    size = activation.size()[-1] * activation.size()[-2]
                    sum_conv = torch.sum(torch.sum(activation, -1), -1) / size
                    sum_conv = sum_conv.view(-1).cpu().detach().numpy()
                    count_info.append(sum_conv)
                if n.startswith('Linear') and not n.endswith('bias'):
                    activation = v1.data - v2.data
                    sum_conv = activation.view(-1).cpu().detach().numpy()
                    count_info.append(sum_conv)
        else:
            count_info = []
            for n, (k1, v1) in zip(named_layer, forget_grad.named_parameters()):
                if n.startswith('Conv2d') and not n.endswith('bias'):
                    activation = v1.data
                    size = activation.size()[-1] * activation.size()[-2]
                    sum_conv = torch.sum(torch.sum(activation, -1), -1) / size
                    sum_conv = sum_conv.view(-1).cpu().detach().numpy()
                    count_info.append(sum_conv)
                if n.startswith('Linear') and not n.endswith('bias'):
                    activation = v1.data
                    sum_conv = activation.view(-1).cpu().detach().numpy()
                    count_info.append(sum_conv)

        count_info = list(chain.from_iterable(count_info))
        mask_index = np.argsort(np.array(count_info))
        if largest:
            mask_index = mask_index[-int(len(count_info) * remove_ratio):]
        else:
            mask_index = mask_index[:int(len(count_info) * remove_ratio)]
        print("total neuron number :{}, mask: {}".format(
            len(count_info), int(len(count_info) * remove_ratio)))
        mask_index.sort()

        named_layer = get_named_layers(model)
        state_dict = copy.deepcopy(model.state_dict())
        num, idx, temp_idx = 0, 0, []
        for n, (k, v) in zip(named_layer, state_dict.items()):
            if n.startswith('Conv2d') and not n.endswith('bias'):
                while idx < len(mask_index) and mask_index[idx] < num + v.size()[0]*v.size()[1]:
                    raw = (mask_index[idx] - num) // v.size()[1]
                    col = (mask_index[idx] - num) % v.size()[1]
                    state_dict[k][raw, col, :, :] = 0.0
                    idx += 1
                if num < len(count_info):
                    num += v.size()[0]*v.size()[1]
            if n.startswith('Linear') and not n.endswith('bias'):
                while idx < len(mask_index) and mask_index[idx] < num + v.size()[0]*v.size()[1]:
                    raw = (mask_index[idx] - num) // v.size()[1]
                    col = (mask_index[idx] - num) % v.size()[1]
                    state_dict[k][raw, col] = 0.0
                    idx += 1
                if num < len(count_info):
                    num += v.size()[0]*v.size()[1]
        assert num == len(count_info)
        new_model = copy.deepcopy(model)
        new_model.load_state_dict(state_dict)
        new_model.cuda()
        return new_model, mask_index, len(count_info)

    def mask_gradients(self, **kwargs):
        model = kwargs['model']
        remove_ratio = kwargs['remove_ratio']
        largest = kwargs['largest']
        forget_data_loader = torch.utils.data.DataLoader(self.forget_loader.dataset,
                                                  batch_size=1, shuffle=False)
        #remain_data_loader = torch.utils.data.DataLoader(self.remain_loader.dataset,
        #                                          batch_size=self.remain_loader.batch_size, shuffle=False)
        layer_out = []
        named_layer = get_named_layers(model, is_state_dict=False)
        for n, (k, v) in zip(named_layer, model.named_parameters()):
            if len(v.size()) == 1 or (n.startswith('Linear')
                                      and v.size()[0] == model.num_classes):
                continue
            layer = get_model_layer(model, k)
            lig = LayerIntegratedGradients(model, layer, multiply_by_inputs=True)
            forget_layer_gradient = None
            for batch in tqdm(forget_data_loader):
                batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
                data, target = batch
                attribution = lig.attribute(data, target=target)
                forget_layer_gradient = torch.cat((forget_layer_gradient, attribution)) \
                    if forget_layer_gradient is not None else attribution
            forget_layer_gradient = torch.sum(forget_layer_gradient, dim=0) / len(self.forget_loader.dataset)

            '''remain_layer_gradient = None
            for batch in tqdm(remain_data_loader):
                batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
                data, target = batch
                attribution = lig.attribute(data, target=target)
                remain_layer_gradient = torch.cat((remain_layer_gradient, attribution)) \
                    if remain_layer_gradient is not None else attribution
            remain_layer_gradient = torch.sum(remain_layer_gradient, dim=0) / len(self.remain_loader.dataset)'''

            layer_gradient = forget_layer_gradient #- remain_layer_gradient

            if n.startswith('Conv'):
                grad_size = layer_gradient.size()
                layer_gradient = torch.sum(layer_gradient, (-1, -2)) \
                                 / grad_size[-1] / grad_size[-2]
                layer_out.append(layer_gradient.cpu().numpy())
            elif n.startswith('Linear'):
                layer_out.append(layer_gradient.cpu().numpy())
        count_info = list(chain.from_iterable(layer_out))
        mask_index = np.argsort(np.array(count_info))
        if largest:
            mask_index = mask_index[-int(len(count_info) * remove_ratio):]
        else:
            mask_index = mask_index[:int(len(count_info) * remove_ratio)]
        print("total neuron number :{}, mask: {}".format(
            len(count_info), int(len(count_info) * remove_ratio)))
        mask_index.sort()

        named_layer = get_named_layers(model)
        state_dict = copy.deepcopy(model.state_dict())
        num, idx, temp_idx = 0, 0, []
        for n, (k, v) in zip(named_layer, state_dict.items()):
            if n.startswith('Conv2d') and not n.endswith('bias'):
                while idx < len(mask_index) and mask_index[idx] < num + v.size()[0]:
                    state_dict[k][mask_index[idx] - num] = 0.0
                    temp_idx.append(mask_index[idx] - num)
                    idx += 1
                if num < len(count_info):
                    num += v.size()[0]
            elif n.startswith('Conv2d') and n.endswith('bias'):
                for t in range(len(temp_idx)):
                    state_dict[k][temp_idx[t]] = 0.0
            if n.startswith('BatchNorm') and not n.endswith('num_bathes_tracked'):
                for t in range(len(temp_idx)):
                    state_dict[k][temp_idx[t]] = 0.0
            elif n.startswith('BatchNorm') and n.endswith('num_bathes_tracked'):
                temp_idx = []

            if n.startswith('Linear') and not n.endswith('bias'):
                while idx < len(mask_index) and mask_index[idx] < num + v.size()[0]:
                    state_dict[k][mask_index[idx] - num] = 0.0
                    temp_idx.append(mask_index[idx] - num)
                    idx += 1
                if num < len(count_info):
                    num += v.size()[0]
            elif n.startswith('Linear') and n.endswith('bias'):
                for t in range(len(temp_idx)):
                    state_dict[k][temp_idx[t]] = 0.0
                temp_idx = []

        assert num == len(count_info)
        new_model = copy.deepcopy(model)
        new_model.load_state_dict(state_dict)
        new_model.cuda()
        return new_model, mask_index, len(count_info)

    def mask_random_filter(self, **kwargs):
        model = kwargs['model']
        remove_ratio = kwargs['remove_ratio']

        named_layer = get_named_layers(model, is_state_dict=False)
        count_info = 0
        for n, (k, v) in zip(named_layer, model.named_parameters()):
            if n.startswith('Conv2d') and not n.endswith('bias'):
                count_info += v.data.size()[0] * v.data.size()[1]
            if n.startswith('Linear') and not n.endswith('bias'):
                count_info += v.data.size()[0] * v.data.size()[1]

        mask_index = np.random.choice(count_info, int(count_info*remove_ratio), replace=False)
        print("total neuron number :{}, mask: {}".format(
            count_info, int(count_info * remove_ratio)))
        mask_index.sort()

        named_layer = get_named_layers(model)
        state_dict = copy.deepcopy(model.state_dict())
        num, idx, temp_idx = 0, 0, []
        for n, (k, v) in zip(named_layer, state_dict.items()):
            if n.startswith('Conv2d') and not n.endswith('bias'):
                while idx < len(mask_index) and mask_index[idx] < num + v.size()[0] * v.size()[1]:
                    raw = (mask_index[idx] - num) // v.size()[1]
                    col = (mask_index[idx] - num) % v.size()[1]
                    state_dict[k][raw, col, :, :] = 0.0
                    idx += 1
                if num < count_info:
                    num += v.size()[0] * v.size()[1]
            if n.startswith('Linear') and not n.endswith('bias'):
                while idx < len(mask_index) and mask_index[idx] < num + v.size()[0] * v.size()[1]:
                    raw = (mask_index[idx] - num) // v.size()[1]
                    col = (mask_index[idx] - num) % v.size()[1]
                    state_dict[k][raw, col] = 0.0
                    idx += 1
                if num < count_info:
                    num += v.size()[0] * v.size()[1]
        assert num == count_info
        new_model = copy.deepcopy(model)
        new_model.load_state_dict(state_dict)
        new_model.cuda()
        return new_model, mask_index, count_info

    def mask_random_channel(self, **kwargs):
        model = kwargs['model']
        remove_ratio = kwargs['remove_ratio']

        named_layer = get_named_layers(model, is_state_dict=False)
        count_info = 0
        for n, (k, v) in zip(named_layer, model.named_parameters()):
            if n.startswith('Conv2d') and not n.endswith('bias'):
                count_info += v.data.size()[0]
            if n.startswith('Linear') and not n.endswith('bias'):
                count_info += v.data.size()[0]

        mask_index = np.random.choice(count_info, int(count_info*remove_ratio), replace=False)
        print("total neuron number :{}, mask: {}".format(
            count_info, int(count_info * remove_ratio)))
        mask_index.sort()

        named_layer = get_named_layers(model)
        state_dict = copy.deepcopy(model.state_dict())
        num, idx, temp_idx = 0, 0, []
        for n, (k, v) in zip(named_layer, state_dict.items()):
            if n.startswith('Conv2d') and not n.endswith('bias'):
                temp_idx = []
                while idx < len(mask_index) and mask_index[idx] < num + v.size()[0]:
                    state_dict[k][mask_index[idx] - num] = 0.0
                    temp_idx.append(mask_index[idx] - num)
                    idx += 1
                if num < count_info:
                    num += v.size()[0]
            elif n.startswith('Conv2d') and n.endswith('bias'):
                for t in range(len(temp_idx)):
                    state_dict[k][temp_idx[t]] = 0.0
            if n.startswith('BatchNorm') and not n.endswith('num_bathes_tracked'):
                for t in range(len(temp_idx)):
                    state_dict[k][temp_idx[t]] = 0.0
            elif n.startswith('BatchNorm') and n.endswith('num_bathes_tracked'):
                temp_idx = []

            if n.startswith('Linear') and not n.endswith('bias'):
                temp_idx = []
                while idx < len(mask_index) and mask_index[idx] < num + v.size()[0]:
                    state_dict[k][mask_index[idx] - num] = 0.0
                    temp_idx.append(mask_index[idx] - num)
                    idx += 1
                if num < count_info:
                    num += v.size()[0]
            elif n.startswith('Linear') and n.endswith('bias'):
                for t in range(len(temp_idx)):
                    state_dict[k][temp_idx[t]] = 0.0
                temp_idx = []
        assert num == count_info
        new_model = copy.deepcopy(model)
        new_model.load_state_dict(state_dict)
        new_model.cuda()
        return new_model, mask_index, count_info

    @_add_criterions
    def activation(self, **kwargs):
        return self.mask_activation(**kwargs)

    @_add_criterions
    def fisher(self, **kwargs):
        return self.mask_fisher(**kwargs)

    @_add_criterions
    def gradients(self, **kwargs):
        return self.mask_gradients(**kwargs)

    @_add_criterions
    def random(self, **kwargs):
        return self.mask_random_channel(**kwargs)

    def __call__(self, criterion_name, **kwargs):
        return criterions[criterion_name](self, **kwargs)
