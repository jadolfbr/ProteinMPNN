# @title Setup Model
from typing import List, Tuple
import matplotlib.pyplot as plt
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, \
    _scores, _S_to_seq, tied_featurize, parse_PDB
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
import json, time, os, sys, glob, re
from collections import defaultdict
import pandas

def get_torch_device(enable_mps = False):

    # For now, we turn off enable mps as macs need to be updated for it to work in this code base right now.
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available() and enable_mps:
       device = torch.device("mps")
    else:
        device = "cpu"
    return device

def setup_protein_mpnn_model(mpnn_path='/Users/jadolfbr/projects/ProteinMPNN', model_type="soluble_model_weights", model_name="v_48_020.pt"):
    sys.path.append('/Users/jadolfbr/projects/ProteinMPNN')

    device = get_torch_device()

    #device = torch.device("cuda:0" if () else "cpu")
    # v_48_010=version with 48 edges 0.10A noise

    #Noise not used for inference in paper, only training. So this should actually be 0.0
    backbone_noise = 0.00  # Standard deviation of Gaussian noise to add to backbone atoms

    path_to_model_weights = f'{mpnn_path}/{model_type}/'
    hidden_dim = 128
    num_layers = 3
    model_folder_path = path_to_model_weights
    if model_folder_path[-1] != '/':
        model_folder_path = model_folder_path + '/'
    checkpoint_path = model_folder_path + f'{model_name}'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print('Number of edges:', checkpoint['num_edges'])
    noise_level_print = checkpoint['noise_level']
    print(f'Training noise level: {noise_level_print}A')
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim,
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise,
                        k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded")
    return model


def make_tied_positions_for_homomers(pdb_dict_list):
    my_dict = {}
    for result in pdb_dict_list:
        all_chain_list = sorted([item[-1:] for item in list(result) if item[:9] == 'seq_chain'])  # A, B, C, ...
        tied_positions_list = []
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        for i in range(1, chain_length + 1):
            temp_dict = {}
            for j, chain in enumerate(all_chain_list):
                temp_dict[chain] = [i]  # needs to be a list
            tied_positions_list.append(temp_dict)
        my_dict[result['name']] = tied_positions_list
    return my_dict

def get_design_chains(design_positions):
    chains = [x[0] for x in design_positions]
    return chains

class MPNNSetup(object):
    """
    Setup MPNN options.
    Apply method should be used by MPNNRun class.

    If model is not given, will load default model. If using multiple sequences, it is good to use a new model.
    Code is mostly refactored from google colab github example.
    Any chain not given in pose-chains will be ignored completely.
    """

    def __init__(self, pdb_path, pose_chains, model=None, omit_AAs='WMC', homomer=False, n_seq_per_target=1,
                 mpnn_path='/Users/jadolfbr/projects/ProteinMPNN', model_type="soluble_model_weights", model_name="v_48_020.pt", sampling_temp=.1, ):
        """
        design_chains = "A B C"
        omit_AAs='WMC'
        """
        self.design_chains = pose_chains
        self.mpnn_path = mpnn_path
        self.pdb_path = pdb_path

        if not model:
            self.model = setup_protein_mpnn_model(self.mpnn_path, model_type=model_type, model_name = model_name)  # Current default model

        else:
            self.model = model

        self.model_name = model_name.split('.')[0]  # Only gets printed, not used.

        self.sampling_temp = sampling_temp
        self.num_sequences_per_target = n_seq_per_target
        self.save_score = 1  # 0 for False, 1 for True; save score=-log_prob to npy files
        self.save_probs = 1  # 0 for False, 1 for True; save MPNN predicted probabilites per position
        self.score_only = 0  # 0 for False, 1 for True; score input backbone-sequence pairs
        self.conditional_probs_only = 0  # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
        self.conditional_probs_only_backbone = 0  # 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)

        self.max_length = 20000  # Max sequence length

        self.jsonl_path = ''  # Path to a folder with parsed pdb into jsonl
        self.omit_AAs = omit_AAs  # Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.

        self.pssm_multi = 0.0  # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
        self.pssm_threshold = 0.0  # A value between -inf + inf to restric per position AAs
        self.pssm_log_odds_flag = 0  # 0 for False, 1 for True
        self.pssm_bias_flag = 0  # 0 for False, 1 for True

        self.homomer = homomer

        self.design_positions_list = None

        ###### Dictionaries for control/setup ######

        self.chain_id_dict = None
        self.fixed_positions_dict = None
        self.pssm_dict = None
        self.omit_AA_dict = None
        self.bias_AA_dict = None
        self.tied_positions_dict = None
        self.bias_by_res_dict = None

        ###########################################

    def set_design_positions(self, design_positions_list: List[Tuple[str, List[int]]]):
        """
        List of positions is 1-based, but starts at each chain.
        List of Tuple of chain positions
        [(chain, positions), (chain, positions)]
        """
        self.design_positions_list = design_positions_list
        #self._set_design_chains(design_positions_list)

    def set_pdb_path(self, pdb_path):
        """
        Set the PDB Path we will be designing
        :param pdb_path:
        :return:
        """
        self.pdb_path = pdb_path

    def set_omit_AAs(self, omit_aas: str):
        """
        Set omit AAs.  Capital letters as a string.
        :param omit_aas:
        :return:
        """
        self.omit_AAs = omit_aas

    def set_homomer(self, homomer):
        self.homomer = homomer

    def set_aa_bias(self, aa_list: List[str], bias_list: List[float]):
        """
        AA list is list of AAs you want to bias. 
        bias list is the weights on those amino acids. 

        These should be the same length. 
        """
        assert (len(aa_list) == len(bias_list))

        bias_list = [float(item) for item in bias_list]
        AA_list = [str(item) for item in aa_list]

        self.bias_AA_dict = dict(zip(AA_list, bias_list))

    def setup(self):
        """
        Setup for running.  Called by MPPNMover
        :return:
        """

        self.designed_chain_list = []
        self.fixed_chain_list = []

        if not self.design_chains == "":
            self.designed_chain_list = re.sub("[^A-Za-z]+", ",", self.design_chains).split(",")
            # print(self.designed_chain_list)

        self.chain_list = list(set(self.designed_chain_list))

        self.omit_AAs_list = self.omit_AAs
        self.alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

        self.omit_AAs_np = np.array([AA in self.omit_AAs_list for AA in self.alphabet]).astype(np.float32)

        self.bias_AAs_np = np.zeros(len(self.alphabet))

        self.pdb_dict_list = parse_PDB(self.pdb_path, input_chain_list=self.chain_list)
        self.dataset_valid = StructureDatasetPDB(self.pdb_dict_list, truncate=None, max_length=self.max_length)

        self.chain_id_dict = {}
        self.chain_id_dict[self.pdb_dict_list[0]['name']] = (self.designed_chain_list, self.fixed_chain_list)

        ###############################################################
        #### Setup Individual Dictionaries ####
        #### These can be split into class functions ####

        # Setup fixed/design positions.  Need all residues to calculate fixed positions.
        if self.design_positions_list:
            self.fixed_positions_dict = self._create_fixed_positions_dict(self.pdb_dict_list,
                                                                          self.design_positions_list,
                                                                          self.design_chains)

        ###############################################################
        # print(self.chain_id_dict)
        # print(self.chain_list)
        # print(self.pdb_dict_list[0])
        for chain in self.chain_list:
            # print(chain)
            l = len(self.pdb_dict_list[0][f"seq_chain_{chain}"])
            # print(f"Length of chain {chain} is {l}")

        if self.homomer:
            self.tied_positions_dict = make_tied_positions_for_homomers(self.pdb_dict_list)
        else:
            self.tied_positions_dict = None

    def _set_design_chains(self, design_positions):
        chains = [x[0] for x in design_positions]
        self.design_chains = " ".join(chains)

    def _create_fixed_positions_dict(self, pdb_dict_list, design_positions_list: List[Tuple[str, List[int]]],
                                     design_chains: str):
        name = pdb_dict_list[0]['name']
        # print(name)
        fixed_positions_dict = dict()
        fixed_positions_dict[name] = dict()
        # print(fixed_positions_dict)
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == 'seq_chain']
        design_chains = [x[1] for x in design_positions_list]

        fixed_chains = [x for x in all_chain_list if x not in design_chains]
        # print(fixed_chains)
        for chain in fixed_chains:
            l = len(pdb_dict_list[0][f'seq_chain_{chain}'])
            r = [i for i in range(1, l + 1)]
            fixed_positions_dict[name][chain] = r

        for chain_pos in design_positions_list:
            chain = chain_pos[0]
            l = len(pdb_dict_list[0][f'seq_chain_{chain}'])
            r = [i for i in range(1, l + 1)]

            print("Setting design:", chain_pos, "Length:", l)
            designed_pos = chain_pos[1]
            fixed_res = [x for x in r if x not in designed_pos]
            fixed_positions_dict[name][chain] = fixed_res

        # print(fixed_positions_dict)
        return fixed_positions_dict

class MPNNRunner(object):
    """
    Basic class to run ProteinMPNN and collect results.
    Based on the google colab.
    """

    def __init__(self, mpnn_setup: MPNNSetup, print_output=True):
        self.s = mpnn_setup
        self.print_output = print_output

    def set_mppn_setup(self, mpnn_setup: MPNNSetup):
        self.s = mpnn_setup

    def set_print_output(self, print_output):
        self.print_output = print_output

    def get_mpnn_setup(self) -> MPNNSetup:
        """
        Get MPNN setup to make edits before next run.
        :return:
        """
        return self.s

    def run(self) -> pandas.DataFrame:

        #Setup each time in case other things have changed.
        self.s.setup()
        device = get_torch_device()

        # @title RUN
        s = self.s

        native_list = []
        out_list = []

        with torch.no_grad():
            print('Generating sequences...')
            for ix, protein in enumerate(s.dataset_valid):
                score_list = []
                all_probs_list = []
                all_log_probs_list = []
                S_sample_list = []

                batch_clones = [copy.deepcopy(protein) for i in range(1)]
                # print(batch_clones)
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, \
                masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
                pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones,
                                                                                                     device,
                                                                                                     s.chain_id_dict,
                                                                                                     s.fixed_positions_dict,
                                                                                                     s.omit_AA_dict,
                                                                                                     s.tied_positions_dict,
                                                                                                     s.pssm_dict,
                                                                                                     s.bias_by_res_dict)

                pssm_log_odds_mask = (pssm_log_odds_all > s.pssm_threshold).float()  # 1.0 for true, 0.0 for false
                name_ = batch_clones[0]['name']

                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs = s.model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                mask_for_loss = mask * chain_M * chain_M_pos
                native_scores = _scores(S, log_probs, mask_for_loss)
                native_scores = native_scores.cpu().data.numpy()

                native_global_scores = _scores(S, log_probs, mask)
                native_global_scores = native_global_scores.cpu().data.numpy()

                temp = s.sampling_temp

                for j in range(s.num_sequences_per_target):
                    randn_2 = torch.randn(chain_M.shape, device=X.device)
                    if s.tied_positions_dict == None:
                        sample_dict = s.model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask,
                                                     temperature=temp, omit_AAs_np=s.omit_AAs_np,
                                                     bias_AAs_np=s.bias_AAs_np,
                                                     chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
                                                     pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=s.pssm_multi,
                                                     pssm_log_odds_flag=bool(s.pssm_log_odds_flag),
                                                     pssm_log_odds_mask=pssm_log_odds_mask,
                                                     pssm_bias_flag=bool(s.pssm_bias_flag), bias_by_res=bias_by_res_all)

                        S_sample = sample_dict["S"]
                    else:
                        sample_dict = s.model.tied_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx,
                                                          mask=mask, temperature=temp, omit_AAs_np=s.omit_AAs_np,
                                                          bias_AAs_np=s.bias_AAs_np, chain_M_pos=chain_M_pos,
                                                          omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
                                                          pssm_bias=pssm_bias, pssm_multi=s.pssm_multi,
                                                          pssm_log_odds_flag=bool(s.pssm_log_odds_flag),
                                                          pssm_log_odds_mask=pssm_log_odds_mask,
                                                          pssm_bias_flag=bool(s.pssm_bias_flag),
                                                          tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta,
                                                          bias_by_res=bias_by_res_all)
                        # Compute scores
                        S_sample = sample_dict["S"]
                    log_probs = s.model(X, S_sample, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all,
                                        randn_2, use_input_decoding_order=True,
                                        decoding_order=sample_dict["decoding_order"])
                    mask_for_loss = mask * chain_M * chain_M_pos
                    scores = _scores(S_sample, log_probs, mask_for_loss)
                    scores = scores.cpu().data.numpy()
                    all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                    # print("All Probs", sample_dict["probs"].cpu().data.numpy().shape)
                    all_log_probs_list.append(log_probs.cpu().data.numpy())
                    S_sample_list.append(S_sample.cpu().data.numpy())

                    # b_ix is number of repeats essentially.  Not sure why this exists. So we are keeping it at 0
                    b_ix = 0
                    masked_chain_length_list = masked_chain_length_list_list[b_ix]
                    masked_list = masked_list_list[b_ix]
                    seq_recovery_rate = torch.sum(torch.sum(
                        torch.nn.functional.one_hot(S[b_ix], 21) * torch.nn.functional.one_hot(S_sample[b_ix], 21),
                        axis=-1) * mask_for_loss[b_ix]) / torch.sum(mask_for_loss[b_ix])
                    seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                    start = 0
                    end = 0
                    list_of_AAs = []
                    for mask_l in masked_chain_length_list:
                        end += mask_l
                        list_of_AAs.append(seq[start:end])
                        start = end

                    seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                    l0 = 0
                    for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                        l0 += mc_length
                        seq = seq[:l0] + '/' + seq[l0:]
                        l0 += 1
                    scores = _scores(S_sample, log_probs, mask_for_loss)
                    scores = scores.cpu().data.numpy()

                    global_scores = _scores(S_sample, log_probs, mask)  # score the whole structure-sequence
                    global_scores = global_scores.cpu().data.numpy()

                    score = scores[b_ix]
                    global_score = global_scores[b_ix]

                    native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                    start = 0
                    end = 0
                    list_of_AAs = []
                    for mask_l in masked_chain_length_list:
                        end += mask_l
                        list_of_AAs.append(native_seq[start:end])
                        start = end
                    native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                    l0 = 0
                    for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                        l0 += mc_length
                        native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                        l0 += 1

                    out_info = defaultdict()

                    score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                    global_score_print = np.format_float_positional(np.float32(global_score), unique=False, precision=4)
                    seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()),
                                                               unique=False, precision=4)
                    line = '>T={}, sample={}, score={}, seq_recovery={}\n{}\n'.format(temp, b_ix, score_print,
                                                                                      seq_rec_print, seq)

                    if self.print_output:
                        print(line.rstrip())

                    # Create output dictionary
                    out_info['name'] = name_
                    out_info['native_path'] = self.s.pdb_path
                    out_info['native_design_residues_score'] = native_scores.mean()
                    out_info['native_global_score'] = native_global_scores.mean()
                    out_info['designed_residues_score'] = float(score_print)
                    out_info['global_score'] = float(global_score_print)
                    out_info['global_score_diff'] = float(global_score_print) - native_global_scores.mean()
                    out_info['local_score_diff'] = float(score_print) - native_scores.mean()

                    out_info['seq_recovery'] = float(seq_rec_print)
                    out_info['T'] = temp
                    # out_info['all_probs'] = sample_dict["probs"].cpu().data.numpy()
                    out_info['sample_num'] = j
                    # out_info['S_sample_list'] = S_sample.cpu().data.numpy()
                    out_info['log_probs'] = log_probs.cpu().data.numpy()
                    out_info['seq'] = seq
                    out_info['native_seq'] = native_seq
                    out_info['line'] = line

                    out_list.append(out_info)

        all_probs_concat = np.concatenate(all_probs_list)
        all_log_probs_concat = np.concatenate(all_log_probs_list)
        S_sample_concat = np.concatenate(S_sample_list)

        #Create a dataframe to easily do downstream tasks.
        df = pandas.DataFrame.from_dict(out_list)
        print(df)
        df = df.sort_values(['global_score_diff'], ascending=True)
        df['design_num'] = range(1, len(df)+1)
        df['design_name'] = df['name']+'_'+df['design_num'].astype(str)
        return df


if __name__ == "__main__":
    #Example of running this in code:
    setup_mpnn = MPNNSetup(
        "path_to_pdb")

    design_positions = [('A', [1, 2, 3, 4, 5]), ('B', [10, 11, 12, 13])]
    setup_mpnn.set_design_positions(design_positions)

    data = MPNNRunner(setup_mpnn).run()
