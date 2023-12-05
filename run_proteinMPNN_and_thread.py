#!/usr/bin/env python3

from argparse import ArgumentParser
import os,sys
#!/usr/bin/env python3
from argparse import ArgumentParser
import os,sys

import pandas

import cse_code.pdb_numbering  as pdb_numbering
from pyrosetta import *
from pyrosetta.rosetta import *
from collections import defaultdict
from cse_code.protein_mpnn import *
from cse_code.pyrosetta import thread_at_position, pack_around_region, pymol_selection_from_reslist

from pyrosetta.rosetta.core.pose import *
import statistics
from statistics import mode


def most_common(l):
    return mode(l)


def parse_positions(design):
    """
    Returns dictionary of pdb_path: design positions
                          pdb_path: rosetta resnums
                          pose
    :param design:
    :return:
    """
    split_designs = []
    out_dict = defaultdict()
    ros_dict = defaultdict()
    pos_dict = defaultdict()
    chain_dict = defaultdict()

    if design.endswith('.txt'):
        for line in open(design, 'r').readlines():
            line = line.strip()
            if line.startswith('#'): continue
            if not line: continue
            split_designs.append(line)
    else:
        split_designs = design.split(';')

    positions = None
    for d in split_designs:
        pdb_path = d.split(',')[0].strip()

        #Allow setting of design positions through previous definitions.  Useful for different backbones, homologous proteins, etc.
        #Can even probably design antibodies of different lengths( 24-48L )
        if d == '::':
            if positions == None:
                sys.exit("For subsequent designs, one must give design positions to start with. ")
        else:

            positions = d.split(',')[1].strip()

        #Strip out HETATM.  ProteinMPNN can't deal with them, rosetta should skip them.
        lines = open(pdb_path, 'r').readlines()
        lines = [l for l in lines if l.startswith('ATOM')]
        pdb_path = pdb_path.replace(".pdb", '_stripped.pdb')
        with open(pdb_path, 'w') as out:
            [out.write(l) for l in lines]


        pose = pose_from_pdb(pdb_path)
        all_chains = []
        for i in range(1, pose.size()+1):
            c = pose.pdb_info().chain(i)
            if c not in all_chains:
                all_chains.append(str(c))

        resnums = pdb_numbering.parse_pdb_numbering(positions, pose)
        print('resnums', resnums)
        rosetta_resnums = [x.get_rosetta_resnum(pose) for x in resnums]
        design_positions = pdb_numbering.get_mpnn_design_dict_from_resnums(resnums, pose)

        out_dict[pdb_path] = design_positions
        ros_dict[pdb_path] = rosetta_resnums
        pos_dict[pdb_path] = pose

        print("CHAINS", " ".join(all_chains))
        chain_dict[pdb_path] = " ".join(all_chains)

    return out_dict, ros_dict, pos_dict, chain_dict

def write_fasta(open_filehandle, design_basename, design_num, row):
    design_seq = row['seq']
    design_score = row['global_score']
    open_filehandle.write(f'>{design_basename}_{design_num}:{design_score}\n')
    open_filehandle.write(design_seq+'\n')

if __name__ == "__main__":
    parser = ArgumentParser("Runs protein MPNN through code and (eventually) PyRosetta to thread sequences onto PDB structures. ProteinMPNN needs to be on the PYTHONPATH. Use jadolfbr version for a 'bug fix'")
    parser.add_argument("--design", '-d', help = "pdb_path,positions; OR .txt file, one per line, comma separated. No header.  "
                                                 "If want to design subsequent pdbs the same way,then subsequent designs can have :: as positions to indicate it will be the same as previous designs."
                                                 "Positions defined as 12-24A:2-6C:1/2/3.C/4-6/7B. Maybe PyMOL selection string sometime."
                                                 "Last letter is always chain. "
                                                 "PDB numbering. Insertion codes after dot. Uses PyRosetta for parsing"
                                                 "Numbering sets MUST be in order or ProteinMPNN freaks out.  IE, order of the PDB - chain A, B, C etc.", required=True)

    parser.add_argument("--mpnn_dir", '-m', help="Directory for proteinMPNN", default="/home/jadolfbr/ProteinMPNN")

    parser.add_argument("--model_weights" ,'-l', default='v_48_020.pt', help = "020 means that .02A noise was added during training.  This usually doesn't need to be changed.")

    parser.add_argument("--model_type", '-e', default='soluble_model_weights',
                        help = "Type of model weights.  These should be in MPNN directory. vanilla_model_weights is the other one we have right now.")

    parser.add_argument("--outdir", '-o', default='mpnn_designs')

    parser.add_argument("--omit_AAs", '-a', default="",
                        help = "Omit this list of Amino Acids.  WMC for oxidation/reduction. YTKS for phospho and ubiquitination.")

    parser.add_argument("--n_sample_designs", '-n', default=8, type=int,
                        help = "Number of designs per sequence.  Use more for de novo design or protein remodeling.")

    parser.add_argument("--sampling_temp", '-t', default=.1, type=float,
                        help = "Sampling temp for ProteinMPNN.  .0001 used for binder design in RFDiffusion. Lower the temp, generally lowers energies from natives, but may not be able to sample much lower energies ")

    parser.add_argument('--skip_threading', '-k', default=False, action='store_true',
                        help = "Skip threading the sequence onto the PDB, which takes time and can be a bit buggy I bet if NCAAs are involved.")

    #parser = add_qsub_args_to_parser(parser)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pyrosetta.init('-ignore_unrecognized_res -beta -ex1 -ex2 -use_input_sc')

    design_dict, rosetta_dict, pose_dict, chain_dict = parse_positions(args.design)
    print(design_dict)

    #Setup Model in order to compute quickly.
    model = setup_protein_mpnn_model(mpnn_path=args.mpnn_dir, model_type=args.model_type,
                             model_name=args.model_weights)

    data_dir = f'{args.outdir}/sequence_data'
    fasta_dir = f'{args.outdir}/fastas'
    pdb_dir = f'{args.outdir}/threaded_pdbs'
    for d in [data_dir, fasta_dir, pdb_dir]:
        os.makedirs(d, exist_ok=True)

    combined_dfs = []
    for pdb_path in design_dict:


        design_basename = os.path.basename(pdb_path).replace("_stripped.pdb", "")
        design_positions = design_dict[pdb_path]

        setup_mpnn = MPNNSetup(
            pdb_path,
            chain_dict[pdb_path],
            model = model,
            omit_AAs=args.omit_AAs,
            n_seq_per_target=args.n_sample_designs)

        setup_mpnn.set_design_positions(design_positions)

        mpnn_runner = MPNNRunner(setup_mpnn)
        data = mpnn_runner.run()
        design_num = 1

        #Also generate just a list of sequences for alphafold stuff.
        out_fasta = open(f'{fasta_dir}/{design_basename}_all.fasta', 'w')

        mutant_data_str = []
        mutant_data_list_dict = []

        mut_data = defaultdict(list)
        ros_resnums = rosetta_dict[pdb_path]
        positions = []
        chains = []

        pose = pose_dict[pdb_path]

        native_seq = pose.sequence()
        for i in ros_resnums:
            pos_s = pose.pdb_info().pose2pdb(i, ' ')
            pos = int(pos_s.split()[0])
            c = pos_s.split()[1]

            mut_data['position'].append(pos)
            mut_data['chain'].append(c)
            mut_data['native'].append(native_seq[i-1])

        pmm_selection = pymol_selection_from_reslist(pose, ros_resnums)
        pymol_selections = [pmm_selection for x in range(len(data))]

        #Used to get the most common mutation from the set of designs.
        position_to_design = defaultdict(list)

        for i, row in data.iterrows():
            design_name = f'{design_basename}_{design_num}'

            # Generate Fastas for each set of designs, and combined.
            out_fasta_local = open(f'{fasta_dir}/{design_basename}_{design_num}.fasta', 'w')
            write_fasta(out_fasta, design_basename, design_num, row)
            write_fasta(out_fasta_local, design_basename, design_num, row)
            out_fasta_local.close()

            #Generate threaded PDBs using PyRosetta.
            design_chains = get_design_chains(design_positions)
            design_chains = chain_dict[pdb_path].split()

            #Here, it's tricky.  I should have just tried to use PyRosetta to do all this stuff honestly, but too late.
            #We need to get the order of chains, get the chains that are being designed and not being designed.
            #Then we need to thread onto the whole sequence, figure out which positions have changed and then repack
            #those with neighbors.  Yea, much easier in PyRosetta.  Oh well too late.

            native_chain_seqs = defaultdict()
            design_chain_seqs = defaultdict()

            pose = pose_dict[pdb_path].clone()
            c_order = []
            for c_id in range(1, pose.num_chains() + 1):
                c_letter = get_chain_from_chain_id(c_id, pose)
                c_seq = pose.chain_sequence(c_id)
                print(c_letter, c_seq)
                native_chain_seqs[c_letter] = c_seq
                c_order.append(c_letter)

            design_seq = row['seq']
            sequences = design_seq.split('/')
            for c_letter, seq in zip(design_chains, sequences):
                design_chain_seqs[c_letter] = seq

            # Now we order them, concatonate, and get ready to thread them using SimpleThreadingMover.
            new_seq = ""
            native_seq = ""
            for c_letter in c_order:
                if c_letter in design_chain_seqs:
                    c_seq = design_chain_seqs[c_letter]
                else:
                    c_seq = native_chain_seqs[c_letter]

                new_seq = new_seq + c_seq

            for c_letter in c_order:
                native_seq = native_seq + native_chain_seqs[c_letter]

            native_seq = native_seq.replace('X', '-')

            #ProteinMPNN decides to put X in for residue number anomolies.  Great.
            new_seq = new_seq.replace('X', '')

            #print('native/new', f'{len(native_seq)}/{len(new_seq)}')
            #print('native/new', f'/// \n{(native_seq)}\n{(new_seq)}\n')
            assert (len(native_seq) == len(new_seq))

            # Now we have the new sequence. Let's only actually call mutate if it's changed.
            new_seq_trimmed = ""
            for native, design in zip(native_seq, new_seq):
                #print('native/design', f'{native}/{design}')
                if native == design:
                    new_seq_trimmed = new_seq_trimmed + '-'
                else:
                    new_seq_trimmed = new_seq_trimmed + design
            #print("NewSeqTrimmed\n", new_seq_trimmed)

            # Next, we figure out which positions are different and create a nice mask that makes a scan type thing easier in the future.

            designed_positions = []
            i = 1

            ros_resums = rosetta_dict[pdb_path]
            all_des_str = []
            for native, design in zip(native_seq, new_seq):

                if native != design:
                    designed_positions.append(i)

                #All attempted design positions. Could do in another loop, but hey we are here.
                if i in ros_resums:
                    pos_s = pose.pdb_info().pose2pdb(i, ' ')
                    pos = int(pos_s.split()[0])
                    c = pos_s.split()[1]

                    if native == design:
                        mutant = '-'
                    else:
                        mutant = design

                    des_str = f'{native}:{pos}{c}:{mutant}'
                    all_des_str.append(des_str)

                    mut_data[str(design_num)].append(mutant)
                    position_to_design[i].append(mutant)

                i += 1
            mutant_data_str.append(" ".join(all_des_str))

            #Now, we want a mask and a way to have these as a separate file.

            if not args.skip_threading:

                #Now, we add them to the packer task and pack with neighbors.
                #Future: look into ATTN packer as this may have better packed structures, but looks like not able to do neighbors, etc.
                print('native/new\n', f'native\n{native_seq}\nnew\n{new_seq_trimmed}')
                thread_at_position(pose, 1, new_seq_trimmed, pack_neighbors=False, pack_rounds=0)
                pack_around_region(pose, designed_positions)
                pose.dump_pdb(f'{pdb_dir}/{design_basename}_{design_num}.pdb')

            design_num += 1

        data['mutant_str'] = mutant_data_str
        data['pymol_selection'] = pymol_selections
        data.to_csv(f'{data_dir}/{design_basename}.csv', index=False)
        combined_dfs.append(data)
        out_fasta.close()

        mut_data['consensus'] = [most_common(position_to_design[x]) for x in position_to_design]

        mut_df = pandas.DataFrame.from_dict(mut_data)

        #Order columns:
        columns = ['position', 'chain', 'native', 'consensus'] +[str(x+1) for x in range(len(data))]
        mut_df = mut_df[columns]
        print(pdb_path)
        print(mut_df)
        mut_df.to_csv(f'{data_dir}/{design_basename}_mutant_list.csv', index=False)

    combined_df = pandas.concat(combined_dfs)
    combined_df.to_csv(f'{data_dir}/combined_mpnn_data.csv', index=False)

    out_fasta = open(f'{fasta_dir}/combined_mpnn_designs.fasta', 'w')
    for i, row in data.iterrows():
        design_basename = row['name']
        design_num = row['design_num']
        write_fasta(out_fasta, design_basename, design_num, row)
    out_fasta.close()


