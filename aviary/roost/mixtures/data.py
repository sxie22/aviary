import functools

import numpy as np
import torch
from pymatgen.core import Composition
from aviary.roost.data import CompositionData


class MixtureData(CompositionData):
    def __init__(self, *args,  **kwargs):
        """
        delimiter (str): delimiter for mixture component strings.
        comp_nodes (bool): include mixture components as nodes using
            sum of elem_nodes embedding contributions
        elem_nodes (bool): include element nodes, weighted by composition
        """
        self.delimiter = kwargs.pop("delimiter")
        self.comp_nodes = kwargs.pop("comp_nodes")
        self.elem_nodes = kwargs.pop("elem_nodes")
        self.hom_edges = kwargs.pop("hom_edges")
        self.het_edges = kwargs.pop("het_edges")

        if not self.comp_nodes and not self.elem_nodes:
            raise ValueError(
                "Must specify at least one node strategy: "
                "{elem_nodes, comp_nodes}")

        if self.het_edges and not self.hom_edges:
            if not self.comp_nodes or not self.elem_nodes:
                raise ValueError(
                    "hom_edges must be True when using only one type of node.")

        if not self.hom_edges and not self.het_edges:
            raise ValueError(
                "Must specify at least one edge strategy: "
                "{hom_edges, het_edges}"
            )

        super().__init__(*args, **kwargs)

    @functools.lru_cache(maxsize=None)  # Cache data for faster training
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx (int): dataset index

        Raises:
            AssertionError: [description]
            ValueError: [description]

        Returns:
            atom_weights: torch.Tensor shape (M, 1)
                weights of atoms in the material
            atom_fea: torch.Tensor shape (M, n_fea)
                features of atoms in the material
            self_fea_idx: torch.Tensor shape (M*M, 1)
                list of self indices
            nbr_fea_idx: torch.Tensor shape (M*M, 1)
                list of neighbor indices
            target: torch.Tensor shape (1,)
                target value for material
            cry_id: torch.Tensor shape (1,)
                input id for the material

        """
        df_idx = self.df.iloc[idx]
        composition = df_idx[self.inputs][0]
        cry_ids = df_idx[self.identifiers].values

        components = composition.split(self.delimiter)
        components = [c.strip() for c in components]

        elem_weights = {}
        component_formulas = []
        for component in components:
            comp_dict = Composition(component).get_el_amt_dict()
            component_formulas.append(comp_dict)
            elements = list(comp_dict.keys())
            weights = list(comp_dict.values())
            weights = np.divide(weights, np.sum(weights))
            for elem, weight in zip(elements, weights):
                elem_weights[elem] = elem_weights.get(elem, 0) + weight
        elements = list(elem_weights.keys())

        try:
            elem_fea = {elem: self.elem_features[elem] for elem in elements}
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_ids[0]} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_ids[0]} [{composition}] composition cannot be parsed into elements"
            )

        atom_fea = []
        weights = []
        node_types = []
        if self.comp_nodes:
            for comp_dict in component_formulas:
                component_fea = [np.multiply(elem_fea[elem], weight)
                                 for elem, weight in comp_dict.items()]
                atom_fea.append(np.sum(component_fea, axis=0))
                weights.append(1)
                node_types.append(1)
        if self.elem_nodes:
            for elem in elements:
                atom_fea.append(elem_fea[elem])
                weights.append(elem_weights[elem])
                node_types.append(0)
        atom_fea = np.vstack(atom_fea)
        weights = np.atleast_2d(weights).T / np.sum(weights)

        self_fea_idx = []
        nbr_fea_idx = []

        elem_idx = [j for j, i in enumerate(node_types) if i == 0]
        comp_idx = [j for j, i in enumerate(node_types) if i == 1]
        n_elems = len(elem_idx)
        n_comps = len(comp_idx)
        if self.hom_edges:
            for j in comp_idx:
                self_fea_idx += [j] * n_comps
            nbr_fea_idx += list(comp_idx) * n_comps
            for j in elem_idx:
                self_fea_idx += [j] * n_elems
            nbr_fea_idx += list(elem_idx) * n_elems
        if self.het_edges:
            for j in comp_idx:
                self_fea_idx += [j] * n_elems
            nbr_fea_idx += list(elem_idx) * n_comps
            for j in elem_idx:
                self_fea_idx += [j] * n_comps
            nbr_fea_idx += list(comp_idx) * n_elems

        # convert all data to tensors
        atom_weights = torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                targets.append(torch.Tensor([df_idx[target]]))
            elif self.task_dict[target] == "classification":
                targets.append(torch.LongTensor([df_idx[target]]))

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx),
            targets,
            *cry_ids,
        )
