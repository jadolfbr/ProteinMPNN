import pyrosetta
from pyrosetta.rosetta.core.pack.task import *
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.core.pack.task.operation import *
from pyrosetta.rosetta.core.select.residue_selector import *
from pyrosetta.rosetta.core.simple_metrics.metrics import *

def thread_at_position(p, start, aa_seq, pack_neighbors=True, threader = None, pack_rounds=1):
    if not threader:
        threader = pyrosetta.rosetta.protocols.simple_moves.SimpleThreadingMover()

    #seq_pos = p.pdb_info().pdb2pose(chain, int(pos_aa[0]))
    threader.set_sequence(aa_seq, start)
    threader.set_pack_neighbors(pack_neighbors)
    threader.set_pack_rounds(pack_rounds)
    threader.apply(p)

    return p

def from_list_to_boolvec(pose, reslist):
    residues_vector = rosetta.utility.vector1_bool(len(pose))
    for i in range(1, pose.total_residue() +1):
        residues_vector[i] = False

    # Set the residues in your list to True
    for res_num in reslist:
        # Make sure the residue number is within the valid range
        if 1 <= res_num <= pose.total_residue():
            residues_vector[res_num] = True

    return residues_vector

def pack_around_region(pose, reslist, pack_neighbors=True, neighbor_dis=6.0, pack_rounds=1):


    # Assuming 'pose' is already defined
    # Assuming 'select_mutated_residues' is already defined
    # Assuming 'neighbor_dis_' and 'pack_rounds_' are already set

    bool_list = from_list_to_boolvec(pose, reslist)

    # Create a TaskFactory
    tf = TaskFactory()
    tf.push_back(InitializeFromCommandline())
    tf.push_back(RestrictToRepacking())

    # Add specific operations based on the condition
    if pack_neighbors:
        print("Packing neighbors")
        select_neighborhood = NeighborhoodResidueSelector(bool_list, neighbor_dis)
        select_not_neighborhood = NotResidueSelector(select_neighborhood)
        prevent_repacking_not_neighborhood = OperateOnResidueSubset(PreventRepackingRLT(), select_not_neighborhood)
        tf.push_back(prevent_repacking_not_neighborhood)
    else:
        select_not_mutated = NotResidueSelector(bool_list)
        prevent_repacking_not_mutated = OperateOnResidueSubset(PreventRepackingRLT(), select_not_mutated)
        tf.push_back(prevent_repacking_not_mutated)

    # Create a PackerTask
    task = tf.create_task_and_apply_taskoperations(pose)
    scorefxn = get_fa_scorefxn()
    # Assuming 'scorefxn_' is already defined and set
    scorefxn.score(pose)  # Segfault Protection

    # Create and apply a PackRotamersMover
    packer = protocols.minimization_packing.PackRotamersMover(scorefxn, task, pack_rounds)
    packer.apply(pose)

    print("Complete")

def pymol_selection_from_reslist(pose, rosetta_reslist):

    ind_selector = ResidueIndexSelector()
    for i in rosetta_reslist:
        ind_selector.append_index(i)

    pmm_metric = SelectedResiduesPyMOLMetric(ind_selector)
    out_str = pmm_metric.calculate(pose)
    return out_str


