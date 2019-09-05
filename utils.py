import typing as t

__all__=[
    "ls_inter",
]

def ls_inter(ls1: t.List, ls2: t.List) -> t.List:
    """return the intersection of two lists
    
    Args:
        ls1 (t.List): list 1
        ls2 (t.List): list 2
    
    Returns:
        t.List: the intersection of two lists
    """
    return list(set(ls1).intersection(set(ls2)))