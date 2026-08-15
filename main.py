import sys, json, argparse, random, re, os, shutil, hashlib
sys.path.append("src/")
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import math
import os.path as osp
import networkx as nx
import pdb

from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
import torch.multiprocessing as mp
# from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch, k_hop_subgraph

from utils import common_tools as ct
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np, masked_mae
from utils.data_convert import generate_samples
from utils.dataloader import DataLoader, AllHistoryDataLoader
from src.model.model import Basic_Model
from src.model import replay
from utils.feature_data import add_time_features
from sklearn.metrics import mean_absolute_error
import torch.nn.functional as F
result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}
continual_matrix = {}
continual_matrices = {"task_context": continual_matrix, "current_state": {}}
forward_transfer = {}
pin_memory = True
n_work = 16

def update(src, tmp):
    for key in tmp:
        if key!= "gpuid":
            src[key] = tmp[key]

def byol_loss(p, z):
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)
    return (2 - 2 * (p * z).sum(dim=1)).sum()


def node_info_nce(online_features, target_features, temperature=0.2, max_nodes=128):
    q = online_features.mean(dim=(0, 3)).transpose(0, 1)
    k = target_features.mean(dim=(0, 3)).transpose(0, 1).detach()
    count = min(q.size(0), k.size(0), int(max_nodes))
    if count < 2:
        return q.sum() * 0.0
    if q.size(0) > count:
        indices = torch.linspace(0, q.size(0) - 1, steps=count, device=q.device).long()
        q, k = q[indices], k[indices]
    else:
        q, k = q[:count], k[:count]
    logits = torch.matmul(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(0, 1)) / float(temperature)
    return F.cross_entropy(logits, torch.arange(count, device=logits.device))

def load_best_model(args):
    if (args.load_first_year and args.year <= args.begin_year+1) or args.train == 0:
        load_path = args.first_year_model_path
        loss = load_path.split("/")[-1].replace(".pkl", "")
    else:
        loss = []
        for filename in os.listdir(osp.join(args.model_path, args.logname+args.time, str(args.year-1))):
            loss.append(filename[0:-4])
        loss = sorted(loss)
        load_path = osp.join(args.model_path, args.logname+args.time, str(args.year-1), loss[0]+".pkl")

    args.logger.info("[*] load from {}".format(load_path))
    checkpoint = torch.load(load_path, map_location=args.device)
    state_dict = checkpoint["model_state_dict"]
    hidden_states_per_year = checkpoint["hidden_states_per_year"]
    if 'tcn2.weight' in state_dict:
        del state_dict['tcn2.weight']
        del state_dict['tcn2.bias']
    model = Basic_Model(args)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        args.logger.info("[*] Missing keys when loading checkpoint: {}".format(missing_keys))
    if unexpected_keys:
        args.logger.info("[*] Unexpected keys when loading checkpoint: {}".format(unexpected_keys))
    if any(key.startswith(("target_backbone.", "target_projection.")) for key in missing_keys):
        model.reset_target_network()
    model.hidden_states_per_year = hidden_states_per_year
    model = model.to(args.device)
    for year, state in model.hidden_states_per_year.items():
        model.hidden_states_per_year[year] = state.to(model.device)
    save_path = "hidden_states_per_year.pth"
    torch.save(model.hidden_states_per_year, save_path)
    print(f"Hidden states per year saved to {save_path}")
    return model, loss[0]

def init(args):
    conf_path = osp.join(args.conf)
    info = ct.load_json_file(conf_path)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    update(vars(args), info)
    vars(args)["path"] = osp.join(args.model_path, args.logname+args.time)
    ct.mkdirs(args.path)
    del info


def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    vars(args)["logger"] = logger
    return logger


def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def train(inputs, args):
    # Model Setting
    global result
    path = osp.join(args.path, str(args.year))
    ct.mkdirs(path)

    # Dataset Definition
    if args.strategy == "all_history_retrained":
        train_loader = AllHistoryDataLoader(
            inputs['_all_history_paths'], inputs['_all_history_node_count'],
            batch_size=args.batch_size, shuffle=True, device=args.device)
        val_loader = DataLoader(inputs["val_x"], inputs["val_y"], batch_size=args.batch_size,
                                shuffle=False, pad_with_last_sample=True)
        vars(args)["sub_adj"] = vars(args)["adj"]
    elif args.strategy == 'incremental' and args.year > args.begin_year:
        train_loader = DataLoader(inputs['train_x'][:, :, args.subgraph.numpy()],inputs['train_y'][:, :, args.subgraph.numpy()], batch_size=args.batch_size, shuffle=True,pad_with_last_sample=True )
        val_loader = DataLoader(inputs['val_x'],inputs['val_y'], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True)
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        train_loader = DataLoader(inputs["train_x"],inputs["train_y"], batch_size=args.batch_size, shuffle=True,pad_with_last_sample=True)
        val_loader = DataLoader(inputs["val_x"],inputs["val_y"], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True)
        vars(args)["sub_adj"] = vars(args)["adj"]
    test_loader = DataLoader(inputs["test_x"],inputs["test_y"], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True)

    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model Definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args)
        model = gnn_model
    else:
        gnn_model = Basic_Model(args).to(args.device)
        model = gnn_model

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    args.logger.info(
        "[*] Model params total:{}, trainable:{}, use_target_branch:{}, sampler_branch:{}, replay_strategy:{}".format(
            total_params,
            trainable_params,
            getattr(args, "use_target_branch", True),
            getattr(args, "sampler_branch", "target"),
            getattr(args, "replay_strategy", "N/A"),
        )
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.device)

    # Model Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    global_train_steps = len(train_loader) // args.batch_size +1

    iters = len(train_loader)
    lowest_validation_loss = 1e7
    counter = 0
    patience = 50
    use_time = []
    min_train_loss = float('inf')
    min_val_loss = float('inf')
    for epoch in range(args.epoch):
        training_loss = 0.0
        start_time = datetime.now()

        # Train Model
        model.train()
        cn = 0
        for batch_idx, data in enumerate(train_loader.get_iterator()):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("x shape {}".format(data['x'].shape))
                args.logger.info("sub_adj shape {}".format(args.sub_adj.shape))
            optimizer.zero_grad()
            use_contrastive = bool(getattr(args, "use_contrastive_loss", False))
            if use_contrastive:
                pred, online_features = model(data, args.year, return_features=True)
                target_features = model.target_representation(data, args.year)
            else:
                pred = model(data,args.year)

            if args.strategy == "incremental" and args.year > args.begin_year:
                pred = pred[:,:, args.mapping, :]
                data['y'] = data['y'][:,:, args.mapping, :]
            prediction_loss = masked_mae(data['y'], pred, 0)
            if use_contrastive:
                contrastive = node_info_nce(online_features, target_features,
                    temperature=getattr(args, "contrastive_temperature", 0.2),
                    max_nodes=getattr(args, "contrastive_max_nodes", 128))
                loss = prediction_loss + float(getattr(args, "contrastive_weight", 0.1)) * contrastive
            else:
                loss = prediction_loss

            # loss = byol_loss(pred,feat)
            training_loss += float(loss)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            model.update_target_network()

            cn += 1

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss/cn
        min_train_loss = min(min_train_loss, training_loss)

        # Validate Model
        model.eval()
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader.get_iterator()):
                pred = model(data,args.year)
                loss = masked_mae_np(data['y'].cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)
        min_val_loss = min(min_val_loss, validation_loss)


        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")

        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': gnn_model.state_dict(),"hidden_states_per_year": gnn_model.hidden_states_per_year}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break
    args.logger.info(f"Min training loss: {min_train_loss:.4f}, Min validation: {min_val_loss:.4f}")

    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
    best_model = Basic_Model(args)
    checkpoint = torch.load(best_model_path, map_location=args.device)
    missing_keys, unexpected_keys = best_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing_keys:
        args.logger.info("[*] Missing keys when loading best checkpoint: {}".format(missing_keys))
    if unexpected_keys:
        args.logger.info("[*] Unexpected keys when loading best checkpoint: {}".format(unexpected_keys))
    if any(key.startswith(("target_backbone.", "target_projection.")) for key in missing_keys):
        best_model.reset_target_network()
    best_model.hidden_states_per_year = checkpoint["hidden_states_per_year"]
    best_model = best_model.to(args.device)

    # Test Model
    test_model(best_model, args, test_loader)
    evaluate_seen_tasks(best_model, args)
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024
    hidden_memory_mb = sum(
        state.numel() * state.element_size()
        for state in best_model.hidden_states_per_year.values()
        if torch.is_tensor(state)
    ) / 1024 / 1024
    history_access_mb = 0.0
    metadata_memory_mb = 0.0
    if args.year > args.begin_year and bool(getattr(args, "replay", False)):
        previous_raw = osp.join(args.raw_data_path, str(args.year - 1) + ".npz")
        if osp.exists(previous_raw):
            history_access_mb = os.path.getsize(previous_raw) / 1024 / 1024
    for metadata_year in {args.year, max(args.begin_year, args.year - 1)}:
        adjacency_path = osp.join(args.graph_path, str(metadata_year) + "_adj.npz")
        if osp.exists(adjacency_path):
            metadata_memory_mb += os.path.getsize(adjacency_path) / 1024 / 1024
    result[args.year] = {
        "total_time": total_time,
        "average_time": sum(use_time)/len(use_time),
        "epoch_num": epoch+1,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "peak_memory_mb": peak_memory_mb,
        "hidden_memory_mb": hidden_memory_mb,
        "history_access_mb": history_access_mb,
        "metadata_memory_mb": metadata_memory_mb,
    }
    args.logger.info(
        "Efficiency stats, params:{}, trainable:{}, peak_memory_mb:{:.2f}, hidden_memory_mb:{:.4f}, history_access_mb:{:.2f}, metadata_memory_mb:{:.2f}".format(
            total_params, trainable_params, peak_memory_mb, hidden_memory_mb,
            history_access_mb, metadata_memory_mb
        )
    )
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for batch_idx, data in enumerate(testset.get_iterator()):
            pred = model(data,args.year)
            loss += masked_mae_np(data['y'].cpu().data.numpy(), pred.cpu().data.numpy(), 0)
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data['y'].cpu().data.numpy())
            cn += 1
        loss = loss/cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        mae = metric(truth_, pred_, args)
        return loss


def evaluate_forward_transfer(args, inputs):
    """Evaluate task t before learning it against a random-init reference."""
    global forward_transfer
    if not bool(getattr(args, "evaluate_fwt", False)) or args.year <= args.begin_year:
        return
    loader_args = dict(batch_size=args.batch_size, shuffle=False, pad_with_last_sample=True)
    def evaluate(model):
        model.eval(); losses=[]
        loader = DataLoader(inputs['test_x'], inputs['test_y'], **loader_args)
        context = max(model.hidden_states_per_year) if model.hidden_states_per_year else args.year
        with torch.no_grad():
            for data in loader.get_iterator():
                pred=model(data,args.year,memory_context_year=context)
                losses.append(masked_mae_np(data['y'].cpu().numpy(),pred.cpu().numpy(),0))
        return float(np.mean(losses))
    # Preserve RNG so constructing the reference does not change subsequent training.
    py_state=random.getstate(); np_state=np.random.get_state(); torch_state=torch.random.get_rng_state()
    cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    previous,_=load_best_model(args)
    previous_mae=evaluate(previous)
    seed_set(int(getattr(args,"fwt_reference_seed",0)))
    reference=Basic_Model(args).to(args.device)
    reference_mae=evaluate(reference)
    random.setstate(py_state); np.random.set_state(np_state); torch.random.set_rng_state(torch_state)
    if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)
    forward_transfer[args.year]={"pre_learning_mae":previous_mae,
        "random_reference_mae":reference_mae,
        "fwt_negative_mae":reference_mae-previous_mae}
    args.logger.info("FWT %s",json.dumps({str(args.year):forward_transfer[args.year]}))


def _evaluate_task(model, args, task_year, memory_context_year):
    task_path = osp.join(args.save_data_path, str(task_year) + "_30day.npz")
    task_data = np.load(task_path, allow_pickle=True)
    loader = DataLoader(task_data['test_x'], task_data['test_y'], batch_size=args.batch_size,
                        shuffle=False, pad_with_last_sample=True)
    losses = []
    with torch.no_grad():
        for data in loader.get_iterator():
            pred = model(data, task_year, memory_context_year=memory_context_year)
            losses.append(masked_mae_np(data['y'].cpu().numpy(), pred.cpu().numpy(), 0))
    del task_data
    return float(np.mean(losses))


def evaluate_seen_tasks(model, args):
    """Build R[t,j] under task-context and current-state TMRB protocols."""
    global continual_matrix, continual_matrices
    if not bool(getattr(args, "evaluate_continual", False)):
        return
    model.eval()
    stage = int(args.year)
    protocols = getattr(args, "continual_memory_protocols", ["task_context", "current_state"])
    if isinstance(protocols, str):
        protocols = [item.strip() for item in protocols.split(",") if item.strip()]
    for protocol in protocols:
        if protocol not in continual_matrices:
            continual_matrices[protocol] = {}
        row = {}
        for task_year in range(args.begin_year, stage + 1):
            if protocol == "task_context":
                context_year = task_year - 1
            elif protocol == "current_state":
                context_year = (max(model.hidden_states_per_year)
                                if model.hidden_states_per_year else stage)
            else:
                raise ValueError("Unknown continual memory protocol: {}".format(protocol))
            row[task_year] = _evaluate_task(model, args, task_year, context_year)
            source = context_year if context_year in model.hidden_states_per_year else "learned_init"
            args.logger.info("CONTINUAL_MEMORY protocol=%s task=%s source=%s",
                             protocol, task_year, source)
        continual_matrices[protocol][stage] = row
        args.logger.info("CONTINUAL_MATRIX protocol=%s %s", protocol,
                         json.dumps({str(stage): row}))
    continual_matrix = continual_matrices.get("task_context", {})


def summarize_matrix(matrix):
    if not matrix:
        return {}
    final_year = max(matrix)
    final_row = matrix[final_year]
    forgetting, relative_forgetting = {}, {}
    bwt_terms = []
    for task, final_error in final_row.items():
        historical = [row[task] for learned, row in matrix.items()
                      if learned >= task and task in row]
        best_error = min(historical)
        forgetting[task] = float(final_error - best_error)
        diagonal = matrix.get(task, {}).get(task)
        if diagonal is not None:
            relative_forgetting[task] = float((final_error - diagonal) / max(abs(diagonal), 1e-12))
            if task != final_year:
                bwt_terms.append(float(diagonal - final_error))
    per_stage = [float(np.mean(list(row.values()))) for row in matrix.values() if row]
    diagonal_values = [matrix[t][t] for t in matrix if t in matrix[t]]
    return {
        "aip_mae": float(np.mean(per_stage)),
        "current_task_mae": float(np.mean(diagonal_values)) if diagonal_values else None,
        "final_seen_task_mae": float(np.mean(list(final_row.values()))),
        "bwt_negative_mae": float(np.mean(bwt_terms)) if bwt_terms else 0.0,
        "average_forgetting_mae": float(np.mean(list(forgetting.values()))) if forgetting else 0.0,
        "average_relative_forgetting": float(np.mean(list(relative_forgetting.values())))
            if relative_forgetting else 0.0,
        "forgetting_by_task": forgetting,
        "relative_forgetting_by_task": relative_forgetting,
    }


def summarize_continual(args):
    summaries = {name: summarize_matrix(matrix)
                 for name, matrix in continual_matrices.items() if matrix}
    args.logger.info("CONTINUAL_SUMMARIES %s", json.dumps(summaries))
    return summaries


def _pad_nodes(array, node_count):
    if array.shape[2] == node_count:
        return array
    if array.shape[2] > node_count:
        return array[:, :, :node_count, :]
    pad_shape = list(array.shape)
    pad_shape[2] = node_count - array.shape[2]
    return np.concatenate([array, np.zeros(pad_shape, dtype=array.dtype)], axis=2)


def build_all_history_inputs(args, current_inputs):
    """Combine all seen training splits; zero labels mask unavailable old nodes."""
    node_count = current_inputs['train_x'].shape[2]
    # Keep current validation/test arrays, but stream historical training files
    # batch-by-batch to avoid multi-GB concatenation.
    merged = {key: current_inputs[key] for key in current_inputs.files}
    merged['_all_history_paths'] = [
        osp.join(args.save_data_path, str(y) + "_30day.npz")
        for y in range(args.begin_year, args.year + 1)]
    merged['_all_history_node_count'] = node_count
    args.logger.info("ALL_HISTORY years=%s-%s files=%s nodes=%s",
                     args.begin_year, args.year, len(merged['_all_history_paths']), node_count)
    return merged

def metric(ground_truth, prediction, args):
    global result
    pred_time = [3,6,12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :i, :,:], prediction[:, :i, :,:], 0)
        rmse = masked_mse_np(ground_truth[:, :i, :,:], prediction[:, :i, :,:], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :i, :,:], prediction[:, :i, :,:], 0)
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
        result[i]["mae"][args.year] = mae
        result[i]["mape"][args.year] = mape
        result[i]["rmse"][args.year] = rmse
    return mae


def main(args):
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)

    for year in range(args.begin_year, args.end_year+1):
        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year) + "_adj.npz"))["x"])
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        if args.auto_lr:
            new_lr = args.lr / 2
            vars(args)["lr"] = max(new_lr, 0.001)  # Ensure lr is at least 0.001
        else:
            vars(args)["lr"] = args.lr
        inputs = generate_samples(30, osp.join(args.save_data_path, str(year)+'_30day'), np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"], graph, val_test_mix=True) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+"_30day.npz"), allow_pickle=True)

        args.logger.info("[*] Year {} load from {}_30day.npz".format(args.year, osp.join(args.save_data_path, str(year))))
        args.logger.info("lr:{}".format(args.lr))
        if args.strategy == "all_history_retrained":
            inputs = build_all_history_inputs(args, inputs)

        if bool(getattr(args, "evaluate_fwt", False)) and year > args.begin_year:
            evaluate_forward_transfer(args, inputs)

        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

        if year == args.begin_year and args.load_first_year:
            model, _ = load_best_model(args)
            test_loader = DataLoader(inputs['test_x'],inputs['test_y'], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True)
            test_model(model, args, test_loader, pin_memory=True)
            continue

        if year > args.begin_year and args.strategy == "static":
            model, loss = load_best_model(args)
            test_loader = DataLoader(inputs['test_x'], inputs['test_y'], batch_size=args.batch_size,
                                     shuffle=False, pad_with_last_sample=True)
            test_model(model, args, test_loader)
            evaluate_seen_tasks(model, args)
            year_path = osp.join(args.path, str(args.year))
            ct.mkdirs(year_path)
            torch.save({"model_state_dict": model.state_dict(),
                        "hidden_states_per_year": model.hidden_states_per_year},
                       osp.join(year_path, str(loss) + ".pkl"))
            args.logger.info("[*] Static baseline: reused the first-period model without updates")
            continue

        if year > args.begin_year and args.strategy == "incremental":
            model, _ = load_best_model(args)

            node_list = list()
            if args.increase:
                cur_node_size = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))

            if args.replay:
                args.logger.info("[*] replay strategy {}".format(args.replay_strategy))
                pre_data = add_time_features(np.load(osp.join(args.raw_data_path, str(year-1)+".npz"))["x"], add_time_in_day=True, add_day_in_week=True)
                cur_data = add_time_features(np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"],add_time_in_day=True, add_day_in_week=True)
                pre_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]).edges)).T
                cur_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T
                vars(args)["topm"] = int(args.replay_ratio*args.graph_size)
                influence_node_list = replay.influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph)
                node_list.extend(list(influence_node_list))
                sampler_dir = osp.join(args.path, "sampler")
                ct.mkdirs(sampler_dir)
                previous_path = osp.join(sampler_dir, str(year - 1) + ".json")
                previous_nodes = set()
                if osp.exists(previous_path):
                    with open(previous_path, "r", encoding="utf-8") as handle:
                        previous_nodes = set(json.load(handle).get("selected_nodes", []))
                selected_nodes = list(map(int, influence_node_list))
                selected_set = set(selected_nodes)
                union = previous_nodes | selected_set
                overlap = len(previous_nodes & selected_set) / len(union) if union else None
                with open(osp.join(sampler_dir, str(year) + ".json"), "w", encoding="utf-8") as handle:
                    json.dump({
                        "year": year,
                        "strategy": args.replay_strategy,
                        "budget": int(args.topm),
                        "selected_nodes": selected_nodes,
                        "selected_scores": [float(args.last_sampler_scores[node]) for node in selected_nodes]
                            if getattr(args, "last_sampler_scores", None) is not None else None,
                        "all_scores": [float(value) for value in args.last_sampler_scores]
                            if getattr(args, "last_sampler_scores", None) is not None else None,
                        "jaccard_with_previous_period": overlap,
                    }, handle, indent=2)

            node_list = list(set(node_list))

            topology_assisted = bool(getattr(args, "topology_assisted_update", True))
            if topology_assisted:
                cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T)
                edge_list = list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)
                graph_node_from_edge = {node for edge in edge_list for node in edge}
                node_list = sorted(set(node_list) & graph_node_from_edge)
                if node_list:
                    subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(
                        node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
            else:
                node_list = sorted(set(node_list))
                subgraph = torch.LongTensor(node_list)
                subgraph_edge_index = torch.empty((2, 0), dtype=torch.long)
                mapping = torch.arange(len(node_list), dtype=torch.long)
            if node_list:
                vars(args)["subgraph"] = subgraph
                vars(args)["subgraph_edge_index"] = subgraph_edge_index
                vars(args)["mapping"] = mapping
            expanded_size = int(args.subgraph.size(0)) if node_list else 0
            logger.info("number of increase/core nodes:{}, nodes after {} hop:{}, total nodes this year {}".format\
                        (len(node_list), args.num_hops, expanded_size, args.graph_size))
            vars(args)["node_list"] = np.asarray(node_list)

        if args.strategy not in ("retrained", "all_history_retrained") and year > args.begin_year and len(args.node_list) == 0:
            model, loss = load_best_model(args)
            ct.mkdirs(osp.join(args.model_path, args.logname+args.time, str(args.year)))
            torch.save({'model_state_dict': model.state_dict(),"hidden_states_per_year": model.hidden_states_per_year}, osp.join(args.model_path, args.logname+args.time, str(args.year), loss+".pkl"))
            test_loader = DataLoader(inputs['test_x'],inputs['test_y'], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True)
            test_model(model, args, test_loader, pin_memory=True)
            logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
            continue


        if args.train:
                train(inputs, args=args)
        else:
            if args.auto_test:
                model, _ = load_best_model(args)
                model.eval()
                test_loader = DataLoader(inputs['test_x'],inputs['test_y'], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True)
                test_model(model, args, test_loader)

    continual_summaries = summarize_continual(args)
    continual_summary = continual_summaries.get("task_context", {})
    output_dir = osp.join(args.path, "metrics")
    ct.mkdirs(output_dir)
    with open(osp.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        efficiency = {str(year): values for year, values in result.items()
                      if isinstance(year, int) and isinstance(values, dict)
                      and "total_time" in values}
        with open(args.conf, "rb") as config_handle:
            config_hash = hashlib.sha256(config_handle.read()).hexdigest()
        json.dump({"dataset": args.dataset, "seed": int(getattr(args, "seed", 0)),
                   "variant": args.logname, "config_hash": config_hash,
                   "continual_matrix": continual_matrix,
                   "continual_summary": continual_summary,
                   "continual_matrices": continual_matrices,
                   "continual_summaries": continual_summaries,
                   "continual_protocols": list(continual_matrices.keys()),
                   "forward_transfer": forward_transfer,
                   "fwt_summary": {
                       "average_fwt_negative_mae": float(np.mean([
                           item["fwt_negative_mae"] for item in forward_transfer.values()]))
                           if forward_transfer else None,
                       "interpretation": "positive means lower pre-learning MAE than random initialization"
                   },
                   "efficiency_by_period": efficiency}, f, indent=2)

    for i in [3, 6, 12]:
        for j in ['mae', 'rmse', 'mape']:
            info = ""
            total = 0.0
            count = 0
            for year in range(args.begin_year, args.end_year + 1):
                if i in result:
                    if j in result[i]:
                        if year in result[i][j]:
                            value = result[i][j][year]
                            info += "{:.2f}\t".format(value)
                            total += value
                            count += 1
            if count > 0:
                avg = total / count
                logger.info("{}\t{}\t{}\t".format(i, j, info) + "average: {:.2f}".format(avg))
            else:
                logger.info("{}\t{}\t{}\t".format(i, j, info) + "average: N/A")

    for year in range(args.begin_year, args.end_year+1):
        if year in result:
            info = "year\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}\tparams\t{}\ttrainable_params\t{}\tpeak_memory_mb\t{}\thidden_memory_mb\t{}\thistory_access_mb\t{}\tmetadata_memory_mb\t{}".format(
                year,
                result[year]["total_time"],
                result[year]["average_time"],
                result[year]['epoch_num'],
                result[year].get("total_params", "N/A"),
                result[year].get("trainable_params", "N/A"),
                result[year].get("peak_memory_mb", "N/A"),
                result[year].get("hidden_memory_mb", "N/A"),
                result[year].get("history_access_mb", "N/A"),
                result[year].get("metadata_memory_mb", "N/A"),
            )
            logger.info(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", type = str, default = "PEMSD4-large")
    parser.add_argument("--conf", type = str, default = 'test.json')
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 1)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--seed", type=int, default=None)
    # parser.add_argument("--first_year_model_path", type = str, default = "res/PEMSD3-stream/model2025-08-18-21:02:04.645828/2011/13.49.pkl", help='specify a pretrained model root')
    parser.add_argument("--first_year_model_path", type = str, default = "res/PEMSD8-mini/model2025-08-19-01:12:54.670652/2012/14.937.pkl", help='specify a pretrained model root')
    # parser.add_argument("--first_year_model_path", type = str, default = "res/PEMSD4-large/model2025-08-18-22:45:39.687639/2009/21.3477.pkl", help='specify a pretrained model root')
    args = parser.parse_args()
    init(args)
    if args.seed is None:
        args.seed = 0
    seed_set(args.seed)

    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["device"] = device
    main(args)
