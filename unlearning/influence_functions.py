from __future__ import print_function
import copy
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader


def cal_grad(model, input):
    model.eval()
    data, target = input
    x = model(data)
    output = F.log_softmax(x, dim=1)
    loss = F.nll_loss(output, target)
    params = [p for p in model.parameters() if p.requires_grad]
    first_derivative = grad(loss, params)
    first_derivative = [_v.detach() for _v in first_derivative]
    return first_derivative


def hvp(y, w, v):
    if len(w) != len(v):
        raise (ValueError("w and v must have the same length."))

        # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, retain_graph=True)

    return_grads = [_v.detach() for _v in return_grads]

    return return_grads


from train import set_seed, test
def cal_if_lissa(cfg, args, model, train_loader, test_loader, forget_loader, loss_fn, damp=0.01, scale=20, epochs=30):
    set_seed(cfg.SEED)
    total_number = len(train_loader.dataset)
    theta = list(model.parameters())

    remove_vec = [torch.zeros_like(params, dtype=torch.float)
                  for params in model.parameters()]
    for idx in range(len(train_loader.dataset)):
        if train_loader.dataset.targets[idx] == args.forget_num:
            remove = train_loader.dataset[idx]
            remove_data = remove[0].unsqueeze(0).cuda()
            remove_target = torch.tensor([remove[1]]).cuda()
            new_vec = cal_grad(model, (remove_data, remove_target))
            remove_vec = [old + v for old, v in zip(remove_vec, new_vec)]
    remove_vec = [_v / total_number for _v in remove_vec]
    state_dict = model.state_dict()
    theta_new = [_v + _r for _v, _r in zip(theta, remove_vec)]
    acc_history,  forget_acc_history = [], []
    for i in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            model.eval()
            data, target = data.cuda(), target.cuda()
            x = model(data)
            output = F.log_softmax(x, dim=1)
            loss = F.nll_loss(output, target)
            theta_delta = [_n - _t for _n, _t in zip(theta_new, theta)]
            theta_new = [_r + (1 - damp) * _d - _h / scale + _t for _r, _d, _h, _t in
                         zip(remove_vec, theta_delta, hvp(loss, theta, theta_delta), theta)]
        res = [(_n - _t) / scale + _t for _n, _t in zip(theta_new, theta)]
        cnt = 0
        for name in state_dict.keys():
            if name.endswith('running_mean') or name.endswith('running_var') or name.endswith('num_batches_tracked'):
                continue
            else:
                state_dict[name] = res[cnt]
                cnt += 1
        assert cnt == len(res)
        new_model = copy.deepcopy(model)
        new_model.load_state_dict(state_dict)
        epoch_test_acc = test(None, new_model, test_loader, loss_fn)
        epoch_forget_acc = test(None, new_model, forget_loader, loss_fn)
        print(epoch_test_acc, epoch_forget_acc)
        acc_history.append(epoch_test_acc)
        forget_acc_history.append(epoch_forget_acc)
    return new_model, acc_history, forget_acc_history

