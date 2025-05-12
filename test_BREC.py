# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation

import os
import numpy as np
import torch
import torch_geometric
import torch_geometric.loader
from loguru import logger
import time
# from BRECDataset_v4 import BRECDataset
from BRECDataset_Wrapper import BRECDataset
from tqdm import tqdm
from torch.nn import CosineEmbeddingLoss
import argparse

from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
import torch_geometric.transforms as T

from Xent_Loss import nt_bxent_loss
from torch_geometric.utils import *

import pickle

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 600
EPOCH = 100
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 120.12  # with 0.995 (0.995^10 > 0.95) original 72.34
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-5
LOSS_THRESHOLD = 0.05
SEED = 2023

global_var = globals().copy()
HYPERPARAM_DICT = dict()
for k, v in global_var.items():
    if isinstance(v, int) or isinstance(v, float):
        HYPERPARAM_DICT[k] = v

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
    "CCoHG": (400, 500),
    "3r2r": (500, 600),
}
parser = argparse.ArgumentParser(description="BREC Test")

parser.add_argument("--P_NORM", type=str, default="2")
parser.add_argument("--EPOCH", type=int, default=EPOCH)
parser.add_argument("--LEARNING_RATE", type=float, default=LEARNING_RATE)
parser.add_argument("--BATCH_SIZE", type=int, default=BATCH_SIZE)
parser.add_argument("--WEIGHT_DECAY", type=float, default=WEIGHT_DECAY)
parser.add_argument("--OUTPUT_DIM", type=int, default=OUTPUT_DIM)
parser.add_argument("--SEED", type=int, default=SEED)
parser.add_argument("--THRESHOLD", type=float, default=THRESHOLD)
parser.add_argument("--MARGIN", type=float, default=MARGIN)
parser.add_argument("--LOSS_THRESHOLD", type=float, default=LOSS_THRESHOLD)
parser.add_argument("--device", type=int, default=0)
parser.add_argument(
    "--loss",
    type=str,
    default="CosineEmbeddingLoss",
    help="Options are ['CosineEmbeddingLoss', 'nt_bxent_loss']",
)
parser.add_argument("--loss_parameter", type=float, default=1)
parser.add_argument(
    '--parts',
    nargs='+',
    default=[],
    help='Options are a subset of '+str(part_dict.keys())
)
parser.add_argument("--name_tag", type=str, default=None)
parser.add_argument("--num_runs", type=int, default=1)
parser.add_argument(
    "--num_layers", type=int, default=10
)  # 9 layers were used for skipcircles dataset
parser.add_argument("--hidden_units", type=int, default=16)
parser.add_argument("--added_dimensions", type=int, default=0)
parser.add_argument("--logging", type=str, default="default.log")
parser.add_argument("--root", type=str, default=".")
# General settings.
args = parser.parse_args()

P_NORM = 2 if args.P_NORM == "2" else torch.inf
EPOCH = args.EPOCH
LEARNING_RATE = args.LEARNING_RATE
BATCH_SIZE = args.BATCH_SIZE
WEIGHT_DECAY = args.WEIGHT_DECAY
OUTPUT_DIM = args.OUTPUT_DIM
SEED = args.SEED
THRESHOLD = args.THRESHOLD
MARGIN = args.MARGIN
LOSS_THRESHOLD = args.LOSS_THRESHOLD
torch_geometric.seed_everything(SEED)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
if args.logging == "default.log" and args.name_tag is not None:
    args.logging = f"{args.name_tag}.log"


# Stage 1: pre calculation
# Here is for some calculation without data. e.g. generating all the k-substructures
def pre_calculation(*args, **kwargs):
    time_start = time.process_time()

    # Do something

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"pre-calculation time cost: {time_cost}")


# Stage 2: dataset construction
# Here is for dataset construction, including data processing
def get_dataset(name, device):
    time_start = time.process_time()

    # Do something
    def makefeatures(data):
        if data.x is None:
            data.x = torch.ones((data.num_nodes, 1))
        data.id = torch.tensor(
            np.random.permutation(np.arange(data.num_nodes))
        ).unsqueeze(1)
        return data

    def addports(data):
        data.ports = torch.zeros(data.num_edges, 1)
        degs = degree(
            data.edge_index[0], data.num_nodes, dtype=torch.long
        )  # out degree of all nodes
        for n in range(data.num_nodes):
            deg = degs[n]
            ports = np.random.permutation(int(deg))
            for i, neighbor in enumerate(data.edge_index[1][data.edge_index[0] == n]):
                nb = int(neighbor)
                data.ports[
                    torch.logical_and(
                        data.edge_index[0] == n, data.edge_index[1] == nb
                    ),
                    0,
                ] = float(ports[i])
        return data

    pre_transform = T.Compose([makefeatures, addports])

    dataset = BRECDataset(name=name, pre_transform=pre_transform)
    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(args, num_nodes, num_features, device):
    time_start = time.process_time()
    # Do something

    n = num_nodes
    gamma = n
    if args.num_runs > 0:
        num_runs = args.num_runs
    else:
        num_runs = gamma

    graph_classification = True
    num_features = num_features
    Conv = GINConv

    class GIN(nn.Module):
        def __init__(self):
            super(GIN, self).__init__()

            dim = args.hidden_units

            self.num_layers = args.num_layers

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(
                Conv(
                    nn.Sequential(
                        nn.Linear(num_features, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Linear(dim, dim),
                    ), train_eps=True
                )
            )
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, OUTPUT_DIM))
            self.fcs.append(nn.Linear(dim, OUTPUT_DIM))

            for i in range(self.num_layers - 1):
                self.convs.append(
                    Conv(
                        nn.Sequential(
                            nn.Linear(dim, dim),
                            nn.BatchNorm1d(dim),
                            nn.ReLU(),
                            nn.Linear(dim, dim),
                        ), train_eps=True
                    )
                )
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, OUTPUT_DIM))

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, Conv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()

            outs = [x]
            x = x.view(-1, x.size(-1))
            run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(
                num_runs, device=edge_index.device
            ).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)

            for i in range(self.num_layers):
                x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(num_runs, -1, x.size(-1)))
            del run_edge_index
            out = None
            for i, x in enumerate(outs):
                x = x.mean(dim=0)
                if graph_classification:
                    x = global_add_pool(x, batch)
                x = self.fcs[i](x)
                if out is None:
                    out = x
                else:
                    out += x
            return out

    model = GIN().to(device)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, device, args):
    """
    When testing on BREC, even on the same graph, the output embedding may be different,
    because numerical precision problem occur on large graphs, and even the same graph is permuted.
    However, if you want to test on some simple graphs without permutation outputting the exact same embedding,
    some modification is needed to avoid computing the inverse matrix of a zero matrix.
    """
    # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use
    # S_epsilon = torch.diag(
    #     torch.full(size=(OUTPUT_DIM, 1), fill_value=EPSILON_MATRIX).reshape(-1)
    # ).to(device)
    def T2_calculation(dataset, log_flag=False):
        with torch.no_grad():
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=BATCH_SIZE)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                pred = model(data.to(device)).detach()
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])
            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            D = X - Y
            if log_flag:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = torch.where(torch.abs(D) < torch.maximum(torch.abs(X), torch.abs(Y))/100, 0, D) # Avoids false positives
            if log_flag:
                logger.info(f"newD = {D}")
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.linalg.pinv(S)
            # If you want to test on some simple graphs without permutation outputting the exact same embedding,
            # please use inv_S with S_epsilon.
            # inv_S = torch.linalg.pinv(S + S_epsilon)
            result = NUM_RELABEL*torch.mm(torch.mm(D_mean.T, inv_S), D_mean)
            if log_flag:
                logger.info(f"result = {result}")
            return result

    time_start = time.process_time()

    # Do something
    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=MARGIN)
    store = []

    file = f"{args.root}/{args.loss}_{str(args.loss_parameter)}.pkl"
    if args.name_tag is not None:
        file = f"{args.root}/{args.name_tag}.pkl"

    for part_name in args.parts:
        part_range = part_dict[part_name]
        logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            logger.info(f"ID: {id}")
            for test_count in range(10):
                dataset_traintest = dataset[
                    id * NUM_RELABEL * 2 : (id + 1) * NUM_RELABEL * 2
                ]
                model = get_model(args, dataset_traintest[0].num_nodes, 1, device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer)  # StepLR(optimizer, gamma=0.5, step_size=250)
                dataset_reliability = dataset[
                    (id + SAMPLE_NUM)
                    * NUM_RELABEL
                    * 2 : (id + SAMPLE_NUM + 1)
                    * NUM_RELABEL
                    * 2
                ]
                model.train()
                for _ in range(EPOCH):
                    traintest_loader = torch_geometric.loader.DataLoader(
                        dataset_traintest, batch_size=BATCH_SIZE
                    )
                    loss_all = 0
                    for data in traintest_loader:
                        optimizer.zero_grad()
                        pred = model(data.to(device))
                        apart = loss_func(
                            pred[0::2],
                            pred[1::2],
                            torch.tensor([-1] * (len(pred) // 2)).to(device),
                        )
                        together = loss_func(
                            torch.cat((pred[0::4], pred[1::4])),
                            torch.cat((pred[2::4], pred[3::4])),
                            torch.tensor([1] * (len(pred) // 2)).to(device),
                        )
                        a = args.loss_parameter
                        if args.loss == "CosineEmbeddingLoss":
                            loss = a*apart+(1-a)*together
                        elif args.loss == "nt_bxent_loss":
                            loss = nt_bxent_loss(pred,
                                                 torch.tensor([(2*a, 2*b) for a in range(len(pred)//2)
                                                               for b in range(len(pred)//2)] +
                                                              [(2*a+1, 2*b+1) for a in range(len(pred)//2)
                                                               for b in range(len(pred)//2)]).to(device),
                                                 0.5, device)
                        loss.backward()
                        optimizer.step()
                        loss_all += len(pred) / 2 * loss.item()
                    loss_all /= NUM_RELABEL
                    logger.info(f"Loss: {loss_all}")
                    if loss_all < LOSS_THRESHOLD:
                        logger.info("Early Stop Here")
                        break
                    scheduler.step(loss_all)

                model.eval()
                T_square_traintest = T2_calculation(dataset_traintest, True)
                T_square_reliability = T2_calculation(dataset_reliability, True)

                if (T_square_traintest > THRESHOLD and T_square_reliability < THRESHOLD
                        and not torch.isclose(T_square_traintest, T_square_reliability, atol=EPSILON_CMP)):
                    break

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
                logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest}")
            logger.info(f"reliability: {reliability_flag} {T_square_reliability}")

            # save to file here
            store.append((part_name, id, isomorphic_flag, T_square_traintest, reliability_flag, T_square_reliability,
                          test_count))

        end = time.process_time()
        time_cost_part = round(end - start, 2)

        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {part_range[1] - part_range[0]}"
        )

        store.append(args)
        with open(file, 'ab') as f:
            pickle.dump(store, f)
        store = []

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")

    Acc = round(cnt / SAMPLE_NUM, 2)
    logger.info(f"Correct in {cnt} / {SAMPLE_NUM}, Acc = {Acc}")

    logger.info(f"Fail in reliability: {fail_in_reliability} / {SAMPLE_NUM}")
    logger.info(correct_list)

    logger.add(f"{args.root}/{args.name_tag}_show.log", format="{message}", encoding="utf-8")
    logger.info(
        "Real_correct\tCorrect\tFail\tnum_layers\thidden_units\tnum_runs\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{args.num_layers}\t{args.hidden_units}\t{args.num_runs}\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{SEED}"
    )


def main():
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    logger.remove(handler_id=None)
    logger.add(f"{args.root}/{args.logging}")
    logger.info(args)

    pre_calculation()
    dataset = get_dataset(name="no_param", device=device)
    evaluation(dataset, device, args)


if __name__ == "__main__":
    main()
