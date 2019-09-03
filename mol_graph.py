#%%
import typing as t

from rdkit import Chem
import networkx as nx

#%%
__all__ = [
    'MolGraph',
]

#%%
class MolGraph(object):
    def __init__(self, smiles: str):
        """MolGraph to handel pharmacophore features
        
        Args:
            smiles (str): SMILES of a molecule
        """
        super().__init__()
        self.mol = Chem.MolFromSmiles(smiles)
        self.num_atoms = self.mol.GetNumAtoms()
        self.num_bonds = self.mol.GetNumBonds()
        self._graph = None
        self._aromatic_ids = None
        self._bond_info = None
        self._h_acceptors = None
        self._h_donors = None
        self._hydrophobic_ids = None
    
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
    def aromatic_ids(self) -> t.List[int]:
        """aromatic pharmacophore atom list
        
        Returns:
            t.List[int]: [int, int, ...]
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
    def h_acceptors(self):
        pass
    
    @property
    def h_donors(self):
        pass
    
    @property
    def hydrophobic_ids(self):
        pass
        