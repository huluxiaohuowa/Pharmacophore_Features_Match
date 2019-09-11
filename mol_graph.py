"""Definition of MolGraph to match a molecule with it`s pharmacophore feats"""
import typing as t
from os import path as op
import json
from collections import Counter
import itertools
from copy import deepcopy

from rdkit import Chem
import networkx as nx
from rdkit.Chem import rdmolops

# from utils import *

__all__ = [
    'MolGraph',
]

class MolGraph(object):
    def __init__(
        self,
        smiles: str,
        patterns_file: str = op.join(
            op.dirname(__file__),
            'datasets',
            'patterns.json'
        )
    ):
        """MolGraph to handle pharmacophore features
        
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
        with open(patterns_file) as f:
            self._dic_patterns = json.load(f)
        self._ion_n = None
        self._ion_p = None
        self._sssr_list = None
        self._chains = None
        self._rings = None
        self._nonring_bond_info = None
        self._cracked_graph = None
        self._hydrophobic_rings = None
        self._hydrophobic_chains = None
        self._non_hydrophobic_ids = None
        self._hydrophobic_subgraphs = None
        self._side_hydrophobic_groups = None
        self._side_chains = None
        self._side_rings = None
        self._num_bonds_per_atom = None
        self._side_atoms = None
        self._side_chain_hydrophobic_subgraphs_idx = None
        self._side_ring_hydrophobic_subgraphs_idx = None
        self._murko = None
    
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
    def chains(self) -> t.List[t.List]:
        """Get chains atom indices
        
        Returns:
            t.List[t.List]: [[chain1 atoms], [chain2 atoms], ...]
        """
        if self._chains is not None:
            return self._chains
        else:
            sssr_atoms = itertools.chain.from_iterable(self.sssr_list)
            chains_subgraph = deepcopy(self.graph)
            chains_subgraph.remove_nodes_from(sssr_atoms)
            cracked_chains = nx.connected_component_subgraphs(chains_subgraph)
            self._chains = [
                list(i_graph.nodes) for i_graph in cracked_chains
            ]
            return self._chains
    
    @property
    def nonring_bond_info(self) -> t.Tuple[t.Tuple]:
        """non ring bond info mathed by SMARTS
        
        Returns:
            t.Tuple[t.Tuple]: tuple of tuple of atom pairs
        """
        if self._nonring_bond_info is not None:
            return self._nonring_bond_info
        else:
            self._nonring_bond_info = self.mol.GetSubstructMatches(
                Chem.MolFromSmarts(self._dic_patterns['non_ring_bonds'])
            )
            return self._nonring_bond_info
    
    @property
    def bidirect_non_ring_bond_info(self) -> t.Tuple[t.Tuple]:
        """Zi ji kan
        
        Returns:
            t.Tuple[t.Tuple]: bidirected nonring bond info, eg. 
                (
                    (0, 1),
                    (0, 2),
                    ...
                    (1, 0),
                    (2, 0)    
                ) 
        """
        reversed_bond_info = tuple([
            bond_info[::-1] for bond_info in self.nonring_bond_info
        ])
        new_bond_info = self.nonring_bond_info + reversed_bond_info
        return new_bond_info
        
    # @property
    # def cracked_graph(self):
    #     graph = deepcopy(self.graph)
    #     for i_edge in graph.edges:
    #         on = False
    #         for ring in self.sssr_list:
    #             if i_edge[0] not in ring and i_edge[1] not in ring:
    #                 on = True
    #                 break
    #         if not on:
    #             graph.remove_edge(i_edge[0], i_edge[1])
    
    @property
    def rings_assems(self) -> t.List[t.List]:
        """ring assemblies atoms
        
        Returns:
            t.List[t.List]: list of list of each ring assembly atoms
        """
        if self._rings is not None:
            return self._rings
        else:
            ring_assesms_subgraph = deepcopy(self.graph)
            ring_assesms_subgraph.remove_edges_from(
                self.bidirect_non_ring_bond_info
            )
            cracked_ra = nx.connected_component_subgraphs(ring_assesms_subgraph)
            self._rings = [
                list(i_graph.nodes) for i_graph in cracked_ra
                if len(i_graph.nodes) > 1
            ]
            return self._rings
    
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
    def num_bonds_per_atom(self) -> t.Counter:
        """Num of bonds of each atom
        
        Returns:
            t.Counter: {atom1: num_of_bonds, atom2: num_of_bonds, ...}
        """
        if self._num_bonds_per_atom is not None:
            return self._num_bonds_per_atom
        else:
            bond_info_flat = list(itertools.chain.from_iterable(self.bond_info))
            self._num_bonds_per_atom = Counter(bond_info_flat)
            return self._num_bonds_per_atom
    
    @property
    def side_atoms(self) -> t.List:
        """Side atom indices
        
        Returns:
            t.List: list of indices of side atoms (only one bond connected) 
        """
        if self._side_atoms is None:
            self._side_atoms = [
                i for (i, j) in self.num_bonds_per_atom.items() if j == 1
            ]
        return self._side_atoms
        
    @property
    def graph(self) -> nx.Graph:
        """Transform the molecule to a nx.Graph object
        
        Returns:
            nx.Graph: 
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
    
    @property
    def non_hydrophobic_ids(self) -> t.List:
        """Non-hydrophobic atom idices
        
        Returns:
            t.List: list of non-hydrophobic atom indices
        """
        if self._non_hydrophobic_ids is not None:
            return self._non_hydrophobic_ids
        else:
            self._non_hydrophobic_ids = list(
                set(range(self.num_atoms)) - set(self.hydrophobic_ids)
            )
            return self._non_hydrophobic_ids
    
    @property
    def hydrophobic_subgraphs(self) -> t.List[nx.Graph]:
        """Hydrophobic subgraphs of a molecule
        
        Returns:
            t.List[nx.Graph]: list of subgraphs of the molecule
        """
        if self._hydrophobic_subgraphs is not None:
            return self._hydrophobic_subgraphs
        else:
            graph_tmp = deepcopy(self.graph)
            graph_tmp.remove_nodes_from(self.non_hydrophobic_ids)
            self._hydrophobic_subgraphs = list(
                nx.connected_component_subgraphs(graph_tmp)
            )
            return self._hydrophobic_subgraphs
    
    @property
    def hydrophobic_rings(self) -> t.List[t.List]:
        """Get hydrophobic ring assemblies
        
        Returns:
            t.List[t.List]: list of list of each hydrophobic ring assembliy
                atoms
        """
        if self._hydrophobic_rings is not None:
            return self._hydrophobic_rings
        else:
            self._hydrophobic_rings = [
                ls_atoms for ls_atoms in self.rings_assems
                if set(ls_atoms).issubset(set(self.hydrophobic_ids))
            ]
            return self._hydrophobic_rings
    
    @property
    def hydrophobic_chains(self) -> t.List[t.List]:
        """Get hydrophobic chains
        
        Returns:
            t.List[t.List]: list of list of each hydrophobic chain, including
                linkers and side chains
        """
        if self._hydrophobic_chains is not None:
            return self._hydrophobic_chains
        else:
            self._hydrophobic_chains =[
                ls_atoms for ls_atoms in self.chains
                if set(ls_atoms).issubset(set(self.hydrophobic_ids))
            ]
            return self._hydrophobic_chains
    
    @property
    def hydrophobic_groups(self) -> t.List[t.List]:
        """Compliation of hydrophobic rings and hydrophobic chains
        
        Returns:
            t.List[t.List]: list of list of hydrophobic group atoms
        """
        return self.hydrophobic_chains + self.hydrophobic_rings 
    
    @property
    def side_chain_hydrophobic_subgraphs_idx(self) -> t.Set:
        if self._side_chain_hydrophobic_subgraphs_idx is None:
            graph_idx = set([
                idx
                for idx, i_graph in enumerate(self.hydrophobic_subgraphs)
                if any (set(i_graph.nodes) & set(self.side_atoms))
            ])
            # graphs = [
            #     self.hydrophobic_subgraphs[i] for i in graph_idx
            # ]
            self._side_chain_hydrophobic_subgraphs_idx = graph_idx
        return self._side_chain_hydrophobic_subgraphs_idx
    
    @property
    def side_ring_hydrophobic_subgraphs_idx(self) -> t.Set:
        if self._side_ring_hydrophobic_subgraphs_idx is None:
            graph_idx = set([
                idx
                for idx, i_graph in enumerate(self.hydrophobic_subgraphs)
                if any (
                    set(i_graph.nodes) & 
                    set(itertools.chain.from_iterable(self.side_rings))
                )
            ])
            self._side_ring_hydrophobic_subgraphs_idx = graph_idx
        return self._side_ring_hydrophobic_subgraphs_idx
    
    @property
    def side_hydrophobic_subgraphs_idx(self) -> t.Set:
        return (
            self.side_chain_hydrophobic_subgraphs_idx | 
            self.side_ring_hydrophobic_subgraphs_idx
        )
    
    @property
    def side_hydrophobic_subgraphs(self) -> t.List[nx.Graph]:
        return [
            self.hydrophobic_subgraphs[idx] 
            for idx in self.side_hydrophobic_subgraphs_idx
        ]
    
    @property
    def side_hydrophobic_atoms(self) -> t.List[t.List[int]]:
        return [
            list(graph.nodes) for graph in self.side_hydrophobic_subgraphs
        ]
    # @property
    # def side_chains(self) -> t.List[t.List[int]]:
    #     if self._side_chains is not None:
    #         self._side_chains = [
    #             ls_atoms
    #             for ls_atoms in self.chains
    #             if any (
    #                 set(self.side_atoms) & set(ls_atoms)
    #             )
    #         ]
    #         return self._side_chains
    #     else:
    #         return self._side_chains
    
    @property
    def murko(self) -> nx.Graph:
        if self._murko is None:
            murko = deepcopy(self.graph)
            while True:
                bond_info_flat = list(itertools.chain.from_iterable(
                    murko.edges
                )) 
                num_bonds = Counter(bond_info_flat)
                side_atoms = [
                    atom_idx
                    for atom_idx in num_bonds
                    if num_bonds[atom_idx] == 1
                ]
                if not(side_atoms):
                    break
                else:
                    murko.remove_nodes_from(side_atoms)
            self._murko = murko
        return self._murko
     
    @property
    def side_rings(self) -> t.List[t.List[int]]:
        if self._side_rings is None:
            self._side_rings = []
            for ring in self.sssr_list:
                graph = deepcopy(self.murko)
                graph.remove_nodes_from(ring)
                if nx.is_connected(graph):
                    self._side_rings.append(ring)
            return self._side_rings
        else:
            return self._side_rings
    
    @property
    def side_hydrophobic_groups(self) -> t.List[t.List[int]]:
        if self._side_hydrophobic_groups is not None:
            return self._side_hydrophobic_groups
        else:
            self._side_hydrophobic_groups = []
            return self._side_hydrophobic_groups