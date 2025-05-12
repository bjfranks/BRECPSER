import os
import pickle
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx


def load_graphml(path):
    data_list = []
    with open(path, "rb") as f:
        while True:
            try:
                graph = from_networkx(nx.convert_node_labels_to_integers(nx.parse_graphml(pickle.load(f))))
                if graph.x is not None:
                    graph.num_nodes = len(graph.x)
                    graph.x = graph.x.unsqueeze(1).to(dtype=torch.float32)
                data_list.append(graph)
            except EOFError:
                break
    return data_list


class CCoHG(InMemoryDataset):
    def __init__(
        self,
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, name)

    @property
    def raw_file_names(self):
        return ["CCoHG.graphml"]

    @property
    def processed_file_names(self):
        return ["CCoHG.pt"]

    def process(self):
        data_list = load_graphml(self.raw_paths[0])
        for i, graph in enumerate(data_list):
            graph.y = torch.tensor([i%2])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def main():
    dataset = CCoHG()
    print(len(dataset))


if __name__ == "__main__":
    main()