import os
import pickle
from collections import Counter
import itertools
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx


def load_graphml(path, precompute=None):
    data_list = []
    with open(path, "rb") as f:
        while True:
            try:
                if precompute is not None:
                    graph = precompute(nx.convert_node_labels_to_integers(nx.parse_graphml(pickle.load(f))))
                else:
                    graph = from_networkx(nx.convert_node_labels_to_integers(nx.parse_graphml(pickle.load(f))))

                if graph.x is not None:
                    graph.num_nodes = len(graph.x)
                    graph.x = graph.x.unsqueeze(1).to(dtype=torch.float32)
                data_list.append(graph)
            except EOFError:
                break
    return data_list


def diameter_and_cycles(nx_graph):
    result = from_networkx(nx_graph)
    result.diameter = torch.tensor([nx.diameter(nx_graph)])
    k_list = list(range(20, 40))
    cycles = list(nx.simple_cycles(nx_graph, length_bound=max(k_list)))
    result.cycles = torch.zeros((len(nx_graph), len(k_list)))
    cycles.sort(key=lambda c: len(c))
    cycles_len = len(cycles)
    for i, k in enumerate(k_list):
        min_i = next((x for x in range(cycles_len) if len(cycles[x]) == k), 0)
        max_i = next((x for x in reversed(range(cycles_len)) if len(cycles[x]) == k), -1)
        for node, count in Counter(itertools.chain.from_iterable(cycles[min_i:max_i + 1])).items():
            result.cycles[node, i] = count
    return result


class _3r2r(InMemoryDataset):
    def __init__(
            self,
            target="diameter",
            root="Data",
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        if target not in ["diameter", "cycles", "None"]:
            raise ValueError(f"{target} is not a valid target for the 3r2r dataset.")
        self.target = target
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.target, name)

    @property
    def raw_file_names(self):
        return ["3reg2reg1000.graphml"]

    @property
    def processed_file_names(self):
        return ["3r2r.pt"]

    def process(self):
        data_list = load_graphml(self.raw_paths[0], precompute=diameter_and_cycles)

        if self.target == "diameter":
            for graph in data_list:
                graph.y = graph.diameter
        elif self.target == "cycles":
            for graph in data_list:
                graph.y = graph.cycles

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def main():
    dataset = _3r2r()
    print(len(dataset))


if __name__ == "__main__":
    main()