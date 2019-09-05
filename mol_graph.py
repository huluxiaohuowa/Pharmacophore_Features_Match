import typing as t
from os import path as op
import json
import itertools

from rdkit import Chem
import networkx as nx
from rdkit.Chem import rdmolops

from utils import *

__all__ = [
    'MolGraph',
]

class MolGraph(object):
    def __init__(self,
                 smiles: str,
                 patterns_file: str = op.join(
                     op.dirname(__file__),
                     'datasets',
                     'patterns.json'
                 )):
        """MolGraph to handel pharmacophore features
        
        Args:
            smiles (str): SMILES of a molecule
        """
        super().__init__()
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.num_atoms = self.mol.GetNumAtoms()
        self.num_bonds = self.mol.GetNumBonds()
        self._graph = None
        self._aromatic_ids = None
        self._bond_info = None
        self._h_acceptors = None
        self._h_donors = None
        self._hydrophobic_ids = None
        self._patterns_file = patterns_file
        with open(self._patterns_file) as f:
            self._dic_patterns = json.load(f)
        self._ion_n = None
        self._ion_p = None
        self._sssr_list = None
    
    @property
    def sssr_list(self) -> t.List[t.List[int]]:
        """Get list of list of indices of each ring
        
        Returns:
            t.List[t.List[int]]: list of list of indices of each single ring
        """
        if self._sssr_list is not None:
            return self._sssr_list
        else:
            self._sssr_list = [
                list(ring) for ring in rdmolops.GetSymmSSSR(self.mol)
            ]
            return self._sssr_list
    
    @property
    def sssr(self):
        pass

    @property
    def chains(self):
        sssr = self.sssr_list
        list_atom_idx = list(range(self.mol.GetNumAtoms()))
        sssr_tmp = []
        for i_sssr in sssr:
            sssr_tmp += i_sssr
        sssr_list = list(set(sssr_tmp))
        list_atom_removed_idx = list(set(list_atom_idx).difference(set(sssr_list)))
        return list_atom_removed_idx

    
    @property
    def bond_info(self) -> t.List[t.Tuple]:
        """Bond info for a molecule
        
        Returns:
            t.List[t.Tuple]:
                [
                    (bond start atom_idx, bond end atom_idx), 
                    ...
                ]
        """
        if self._bond_info is not None:
            return self._bond_info
        else:
            self._bond_info = [
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                for bond in self.mol.GetBonds()
            ]
        return self._bond_info

    @property
    def graph(self) -> nx.Graph:
        """MolGraph
        
        Returns:
            nx.classes.graph.Graph: 
        """
        if self._graph is not None:
            return self._graph
        else:
            graph = nx.Graph()
            graph.add_nodes_from(range(self.num_atoms))
            graph.add_edges_from(self.bond_info)
            self._graph = graph
        return self._graph
    
    @property
    def aromatic_ids(self) -> t.List[t.List[int]]:
        """aromatic pharmacophore atom list
        
        Returns:
            t.List[t.List[int]]: [[int, int, ...], ...]
        """
        if self._aromatic_ids is not None:
            return self._aromatic_ids
        else:
            aromatic_atoms = [
                atom.GetIdx() for atom in self.mol.GetAromaticAtoms()
            ]
            aromatic_bond_info = [
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                for bond in self.mol.GetBonds()
                if bond.GetBondType() is Chem.rdchem.BondType.AROMATIC
            ]
            aromatic_graph = nx.Graph()
            aromatic_graph.add_nodes_from(aromatic_atoms)
            aromatic_graph.add_edges_from(aromatic_bond_info)
            aromatic_subgraphs = list(
                nx.connected_component_subgraphs(
                    aromatic_graph
                )
            ) 
            aromatic_ids = [
                list(sub_graph.nodes) for sub_graph in aromatic_subgraphs
            ]
            self._aromatic_ids = aromatic_ids
            return self._aromatic_ids
        
    @property
    def h_acceptors(self) -> t.Tuple[t.Tuple]:
        """Get H bond acceptor atom indices 
        
        Returns:
            t.Tuple[t.Tuple]: ummmm
        """
        if self._h_acceptors is not None:
            return self._h_acceptors
        else:
            self._h_acceptors = self.mol.GetSubstructMatches(
                Chem.MolFromSmarts(self._dic_patterns['HBA'])
            )
            return self._h_acceptors
    
    @property
    def h_donors(self) -> t.Tuple[t.Tuple]:
        """Get H bond donor atom indices
        
        Returns:
            t.Tuple[t.Tuple]: ummmm
        """
        if self._h_donors is not None:
            return self._h_donors
        else:
            self._h_donors = self.mol.GetSubstructMatches(
                Chem.MolFromSmarts(self._dic_patterns['HBD'])
            )
            return self._h_donors
    
    @property
    def ion_p(self) -> t.Tuple[t.Tuple]:
        """negatively ionizable groups
        
        Returns:
            t.Tuple[t.Tuple]: ((id, id), (id, id), ) empty and duplicate items
                removed
        """
        if self._ion_p is not None:
            return self._ion_p
        else:
            ls_ids = [
                self.mol.GetSubstructMatches(
                    Chem.MolFromSmarts(pattern)
                ) for pattern in self._dic_patterns['P']
            ]
            ids = set(itertools.chain.from_iterable(ls_ids))
            self._ion_p = tuple(ids)
            return self._ion_p

    @property
    def ion_n(self) -> t.Tuple[t.Tuple]:
        """Positive ionizable groups
        
        Returns:
            t.Tuple[t.Tuple]: ((id, id), (id, id), ) empty and duplicate items
                removed
        """
        if self._ion_n is not None:
            return self._ion_n
        else:
            ls_ids = [
                self.mol.GetSubstructMatches(
                    Chem.MolFromSmarts(pattern)
                ) for pattern in self._dic_patterns['N']
            ]
            ids = set(itertools.chain.from_iterable(ls_ids))
            self._ion_n = tuple(ids)
            return self._ion_n
    
    @property
    def hydrophobic_ids(self) -> t.List:
        """Get hydrophobic atoms indices
        
        Returns:
            t.List: list of hydrophobic atom indices
        """
        if self._hydrophobic_ids is not None:
            return self._hydrophobic_ids
        else:
            hydrophobic_ids = self.mol.GetSubstructMatches(
                Chem.MolFromSmarts(self._dic_patterns['H'])
            )
            self._hydrophobic_ids = list(itertools.chain.from_iterable(
                hydrophobic_ids
            ))
            return self._hydrophobic_ids
    