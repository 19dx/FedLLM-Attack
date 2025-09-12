import random
import torch
#== attack and defense ==
import math
import copy
from functools import reduce
import numpy as np
import torch.nn.functional as F 
#== attack and defense ==

def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round

def global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
    
    elif fed_args.fed_alg == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    # ============== Defense baselines ==================
    elif fed_args.fed_alg == 'median':
        key_list = {}
        for net_id, client in enumerate(clients_this_round):
            net_para = local_dict_list[client]
            if net_id == 0:
                for key in net_para:
                    key_list[key] = [net_para[key].unsqueeze(0)]
            else:
                for key in net_para:
                    key_list[key].append(net_para[key].unsqueeze(0))
        for key in net_para:
            key_value_cat = torch.cat(key_list[key])
            key_value_median, _ = torch.median(key_value_cat, dim=0)
            global_dict[key] = key_value_median
    
    elif fed_args.fed_alg == 'krum' :
        expected_n_attacker = 0
        for malicious_num_client in fed_args.malicious_num_clients:
            expected_n_attacker += malicious_num_client
        
        expected_n_attacker = round(fed_args.sample_clients*expected_n_attacker/fed_args.num_clients)
     
        model_weight_list = []
        for net_id, client in enumerate(clients_this_round):
            net_para = local_dict_list[client]
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat = torch.cat(model_weight_list, dim=0)
        model_weight_krum, aggregate_idx = get_krum(model_weight_cat, expected_n_attacker)
        model_weight_krum = model_weight_krum.reshape(-1)
        # make sure aggregate_idx is on CPU, because clients_this_round is on CPU
        aggregate_idx = aggregate_idx.cpu() if aggregate_idx.is_cuda else aggregate_idx

        aggregate_idx_list = torch.tensor(clients_this_round)[aggregate_idx].tolist()
        aggregate_idx_list.sort()

        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_dict[key] = model_weight_krum[current_idx : current_idx + length].reshape(net_para[key].shape)
            current_idx += length
    
    elif fed_args.fed_alg == 'trimmedmean':
        net_para_list = []
        for net_id, client in enumerate(clients_this_round):
            net_para_list.append(local_dict_list[client])
            
        trimmed_num = 1
        
        # Trimmed mean
        for key in global_dict:
            net_para_stack = torch.stack([net_row[key] for net_row in net_para_list])
            net_shape = net_para_stack.shape[1:]
            net_para_stack = net_para_stack.reshape(len(net_para_list), -1)
            net_para_sorted = net_para_stack.sort(dim=0).values
            result = net_para_sorted[trimmed_num:-trimmed_num, :]
            result_type = result.dtype
            result = result.float().mean(dim=0).type(result_type)
            result = result.reshape(net_shape)
            global_dict[key] = result
    
    elif fed_args.fed_alg == 'foolsgold':
        local_dict_list_this_round = [local_dict_list[i] for i in clients_this_round]
        model_weight_list = []
        for net_id, net_para in enumerate(local_dict_list_this_round):
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat = torch.cat(model_weight_list, dim=0)

        update_mean, update_std, update_cat, global_weight = get_update_static(local_dict_list_this_round, global_dict)
        model_weight_foolsgold, wv = get_foolsgold(update_cat, global_weight)

        current_idx = 0 
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_dict[key] = model_weight_foolsgold[current_idx : current_idx + length].reshape(net_para[key].shape)
            current_idx += length     
    
    elif fed_args.fed_alg == 'residual':
        local_dict_list_this_round = [local_dict_list[i] for i in clients_this_round]
        model_weight_list = []
        global_dict, reweight = IRLS_aggregation_split_restricted(local_dict_list_this_round, 2.0, 0.05)
    
    elif fed_args.fed_alg == 'dnc':
        local_dict_list_this_round = [local_dict_list[i] for i in clients_this_round]
        expected_n_attacker = 0
        for malicious_num_client in fed_args.malicious_num_clients:
            expected_n_attacker += malicious_num_client
            
        expected_n_attacker = round(fed_args.sample_clients*expected_n_attacker/fed_args.num_clients)
            
        current_idx = 0
        update_mean, update_std, update_cat, global_weight = get_update_static(local_dict_list_this_round, global_dict)
        i_final,wv = do_dnc(update_cat,m=expected_n_attacker)
        print("===> DnC Aggregation: ", wv)
        model_weight_foolsgold = foolsgold_wv_update(wv, update_cat, global_weight)
        
        for net_id, net_para in enumerate(local_dict_list_this_round):
            break
        
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_dict[key] = model_weight_foolsgold[current_idx : current_idx + length].reshape(net_para[key].shape)
            current_idx += length

    else:   # Normal dataset-size-based aggregation 
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    
    return global_dict, global_auxiliary

def get_weight(model_weight):
    weight_tensor_result = []
    for k, v in model_weight.items():
        weight_tensor_result.append(v.reshape(-1).float())
    weights = torch.cat(weight_tensor_result)
    return weights

def get_krum(inputs, attacker_num=1):
    inputs = inputs.unsqueeze(0).permute(0, 2, 1)
    n = inputs.shape[-1]
    k = n - attacker_num - 1

    x = inputs.permute(0, 2, 1)

    cdist = torch.cdist(x, x, p=2)
    # find the k+1 nbh of each point
    nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
    # the point closest to its nbh
    i_star = torch.argmin(nbhDist.sum(2))
    mkrum = inputs[:, :, nbh[:, i_star, :].view(-1)].mean(2, keepdims=True)
    return mkrum, nbh[:, i_star, :].view(-1)

def get_update_static(local_dict_list, global_dict):
    model_weight_list = []
    net_id_list = []

    glboal_net_para = global_dict
    global_weight = get_weight(glboal_net_para).unsqueeze(0)

    for net_id, net_para in enumerate(local_dict_list):
        net_id_list.append(net_id)
        model_weight = get_weight(net_para).unsqueeze(0)
        model_update = model_weight - global_weight
        model_weight_list.append(model_update)
    model_weight_cat = torch.cat(model_weight_list, dim=0)
    model_std, model_mean = torch.std_mean(model_weight_cat, unbiased=False, dim=0)

    return model_mean, model_std, model_weight_cat, global_weight

def get_foolsgold(grads, global_weight):
    n_clients = grads.shape[0]
    grads_norm = F.normalize(grads, dim=1)
    device = grads.device # make sure aggregate_idx is on CPU,because clients_this_round is on CPU
    cs = torch.mm(grads_norm, grads_norm.T)
    cs = cs - torch.eye(n_clients, device=device)
    maxcs, _ = torch.max(cs, axis=1)

    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    maxcs_2, _ = torch.max(cs, axis=1)
    wv = 1 - maxcs_2

    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / torch.max(wv)
    wv[(wv == 1)] = 0.99

    # Logit function
    eps = 1e-8  # Add a small constant to prevent division by zero 
    wv = torch.log(wv / (1 - wv + eps)) + 0.5
    wv[(torch.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    model_weight_list = []
    for i in range(0, n_clients):
        if wv[i] != 0:
            current_weight = global_weight + wv[i] * grads[i]
            model_weight_list.append(current_weight.to(device))
    fools_gold_weight = torch.cat(model_weight_list).mean(0, keepdims=True)

    return fools_gold_weight.view(-1), wv

def IRLS_aggregation_split_restricted(local_dict_list, LAMBDA=2, thresh=0.1):
    SHARD_SIZE = 2000

    w = []
    for net_id, net_para in enumerate(local_dict_list):
        w.append(net_para)

    w_med = copy.deepcopy(w[0])

    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)

        if total_num < SHARD_SIZE:
            reweight, restricted_y = reweight_algorithm_restricted(transposed_y_list, LAMBDA, thresh)
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE : (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = reweight_algorithm_restricted(y, LAMBDA, thresh)
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE : (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)

    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    return w_med, reweight

def reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2.0 / num_models)]).to(y.device)

    beta = torch.cat(
        (
            intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
            slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
        ),
        dim=-1,
    )
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh) + line_y * (reweight < thresh)
    return reweight_regulized, restricted_y

def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output

def repeated_median(y):
    eps = np.finfo(float).eps
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float("Inf")] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts

def weighted_average(w_list, weights):
    w_avg = copy.deepcopy(w_list[0])
    weights = weights / weights.sum()
    assert len(weights) == len(w_list)
    for k in w_avg.keys():
        w_avg[k] = 0
        for i in range(0, len(w_list)):
            w_avg[k] += w_list[i][k] * weights[i]
        # w_avg[k] = torch.div(w_avg[k], len(w_list))
    return w_avg, weights

def do_dnc(grad_vec,niters=1,c=1,b=10000,m=2,n=10):
    """
    For all these datasets, we set niters, c, and b in Algorithm 2 to 1, 1, and 10,000, respectively
    """
    n = grad_vec.shape[0]
    d = grad_vec.shape[1]
    I_good = []
    for i in range(niters):
        r = torch.randperm(d)[:b]
        r = r.sort().values
        s_grad_vec = grad_vec[:,r] # n,b
        mu = s_grad_vec.mean(dim=0) #  1,b
        grad_c = s_grad_vec - mu
        svd_res = torch.linalg.svd(grad_c)
        v = svd_res.Vh[0] # top right singular eigenvector, (n)
        outlier_score = torch.Tensor([torch.dot(s_grad_vec[j]-mu, v).pow(2) for j in range(n)])
        i_val, i_good = outlier_score.sort()
        I = i_good[:n-c*m]
        I_good.append(set(I.tolist()))
    I_final = set.intersection(*I_good)
    I_final = torch.LongTensor(list(I_final))
    wvs = torch.zeros(n)
    for i in I_final:
        wvs[i] = 1
    return I_final, wvs

def foolsgold_wv_update(wv, grads, global_weight):
    """
    update global model with aggregated wv values
    
    """
    n_clients = grads.shape[0]
    
    model_weight_list = []
    for i in range(0, n_clients):
        if wv[i] != 0:
            current_weight = global_weight + wv[i] * grads[i]
            model_weight_list.append(current_weight)
    fools_gold_weight = torch.cat(model_weight_list).mean(0, keepdims=True)
    return fools_gold_weight.view(-1)