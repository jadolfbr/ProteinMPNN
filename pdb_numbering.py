from typing import List, Tuple
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.core.pose import get_chain_id_from_chain
from collections import defaultdict


class Resnum(object):
    """
    Simple object to parse resnum strings
    """
    def __init__(self, num = None, chain=None, insert_code = ' '):
        self.num = num
        self.chain = chain
        self.insert_code = insert_code

    def from_string(self, resnum_str, chain):
        if '.' in resnum_str:
            self.insert_code = resnum_str[-1]
            self.num = int(resnum_str.split('.')[0])
        else:
            self.insert_code = ' '
            self.num = int(resnum_str)

        self.chain = chain
        return self

    def from_rosetta(self, res, pose):
        s = pose.pdb_info().pose2pdb(res)
        self.num = int(s.split()[0])
        self.chain = s.split()[1]
        self.icode = pose.pdb_info().icode(res)
        return self

    def get_tuple(self) -> Tuple[int, str, str]:
        """
        Return tuple of resnums (num, chain, insert_code)
        :return:
        """
        return (self.num, self.chain, self.insert_code)

    def get_string(self) -> str:
        """
        Return string as num:insert_code:chain
        :return:
        """
        return f'{self.num}:{self.insert_code}:{self.chain}'

    def get_chain_based_resnum(self, pose) -> int:
        chain_num = get_chain_id_from_chain(self.chain, pose)

        if chain_num == 1:
            return self.get_rosetta_resnum(pose)

        else:
            # Tricky because we have to use the order of the chains to get the actual pose number.
            seq = ""
            for i in range(1, chain_num):
                seq = seq + pose.chain_sequence(i)

            resnum = self.get_rosetta_resnum(pose)
            return resnum - len(seq)

    def get_rosetta_resnum(self, pose) -> int:
        return pose.pdb_info().pdb2pose(self.chain, self.num, self.insert_code)

def parse_pdb_numbering(resnums: str, pose) -> List[Resnum]:
    """
    Positions defined as 12-24A:2-6C:1/2/3.C/4-6/7B.
    Last letter is always chain.
    PDB numbering. Insertion codes after dot.
    Returns a tuple of num:insert_code:chain for further use.
    :param resnums:
    :return: List[Resnum]
    """

    out_resnums = []
    rsets = resnums.split(':')
    for r1 in rsets:
        chain = r1[-1]
        r1 = r1[0:-1]
        r1SP = r1.split('/')
        for r2 in r1SP:
            if '-' in r2:
                r2SP = r2.split('-')
                start = r2SP[0]
                end = r2SP[1]

                start_resnum = Resnum().from_string(start, chain)
                end_resnum = Resnum().from_string(end, chain)

                for i in range(start_resnum.get_rosetta_resnum(pose), end_resnum.get_rosetta_resnum(pose)+1):
                    out_resnums.append(Resnum().from_rosetta(i, pose))

            else:
                out_resnums.append(Resnum().from_string(r2, chain))

    return out_resnums

def get_mpnn_design_dict_from_resnums(resnums: List[Resnum], pose):
    """
    Gets a tuple of design positions for MPNN class.
    List(Tuple(chain, [resnums])

    :param resnums:
    :param pose:
    :return:
    """
    design_dict = defaultdict(list)

    for resnum in resnums:
        design_dict[resnum.chain].append(resnum.get_chain_based_resnum(pose))

    design_positions = []
    for chain in design_dict:
        design_positions.append((chain, design_dict[chain]))
    return design_positions

