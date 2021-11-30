"""Global feature vector added prior to dense neural network"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from aviary.core import BaseModelClass
from aviary.segments import MeanPooling, SimpleNetwork

from aviary.cgcnn.model import DescriptorNetwork


class GlobalsNet(BaseModelClass):

    def __init__(
        self,
        robust,
        n_targets,
        elem_emb_len,
        nbr_fea_len,
        n_global,
        elem_fea_len=64,
        n_graph=4,
        h_fea_len=128,
        n_trunk=1,
        n_hidden=1,
        **kwargs,
    ):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_elem_fea_len: int
            Number of atom features in the input.
        nbr_fea_len: int
            Number of bond features.
        elem_fea_len: int
            Number of hidden atom features in the convolutional layers
        n_graph: int
            Number of convolutional layers
        h_fea_len: int
            Number of hidden features after pooling
        n_hidden: int
            Number of hidden layers after pooling
        """
        super().__init__(robust=robust, **kwargs)

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "nbr_fea_len": nbr_fea_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
        }

        self.node_nn = DescriptorNetwork(**desc_dict)

        self.model_params.update(
            {
                "robust": robust,
                "n_targets": n_targets,
                "h_fea_len": h_fea_len,
                "n_hidden": n_hidden,
                "n_global": n_global,
            }
        )

        self.model_params.update(desc_dict)

        self.pooling = MeanPooling()

        # define an output neural network
        if self.robust:
            n_targets = [2 * n for n in n_targets]

        out_hidden = [h_fea_len] * n_hidden
        trunk_hidden = [h_fea_len] * n_trunk

        self.trunk_nn = SimpleNetwork(elem_fea_len, h_fea_len, trunk_hidden)

        self.bn = nn.BatchNorm1d(n_global)
        self.output_nns = nn.ModuleList(
            SimpleNetwork(h_fea_len + n_global, n, out_hidden) for n in n_targets
        )

    def forward(self, atom_fea, nbr_fea, self_idx, nbr_idx, crystal_atom_idx, glob_fea):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
            Atom hidden features after convolution

        """
        atom_fea = self.node_nn(
            atom_fea, nbr_fea, self_idx, nbr_idx
        )

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # NOTE required to match the reference implementation
        crys_fea = nn.functional.softplus(crys_fea)

        crys_fea = F.relu(self.trunk_nn(crys_fea))

        glob_fea = self.bn(glob_fea)
        dense_fea = torch.cat([crys_fea, glob_fea], dim=1)
        # apply neural network to map from learned features to target
        return (output_nn(dense_fea) for output_nn in self.output_nns)


class DenseNet(BaseModelClass):
    """Benchmarking Purposes"""
    def __init__(
        self,
        robust,
        n_targets,
        n_global,
        h_fea_len=128,
        n_hidden=1,
        **kwargs,
    ):
        super().__init__(robust=robust, **kwargs)

        self.model_params.update(
            {
                "robust": robust,
                "n_targets": n_targets,
                "h_fea_len": h_fea_len,
                "n_hidden": n_hidden,
                "n_global": n_global,
            }
        )

        # define an output neural network
        if self.robust:
            n_targets = [2 * n for n in n_targets]

        self.bn = nn.BatchNorm1d(n_global)

        out_hidden = [h_fea_len] * n_hidden
        self.output_nns = nn.ModuleList(
            SimpleNetwork(n_global, n, out_hidden) for n in n_targets
        )

    def forward(self, dense_fea, *args):
        dense_fea = self.bn(dense_fea)
        return (output_nn(dense_fea) for output_nn in self.output_nns)
