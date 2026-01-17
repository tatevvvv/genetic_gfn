import os
import glob
import numpy as np
import hashlib
from tdc import Oracle, Evaluator

from rdkit.Chem import MolFromSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
try:
    from openbabel import pybel
except ImportError:
    try:
        import pybel
    except ImportError:
        raise ImportError("Please install openbabel: conda install -c conda-forge openbabel or pip install openbabel-wheel")

import subprocess
import multiprocessing


def int_div(smiles):
    evaluator = Evaluator(name = 'Diversity')
    return evaluator(smiles)


def get_scores(smiles, mode="QED", n_process=16):
    smiles_groups = []
    group_size = len(smiles) / n_process
    for i in range(n_process):
        smiles_groups += [smiles[int(i * group_size):int((i + 1) * group_size)]]

    temp_data = []
    pool = multiprocessing.Pool(processes = n_process)
    for index in range(n_process):
        temp_data.append(pool.apply_async(get_scores_subproc, args=(smiles_groups[index], mode, )))
    pool.close()
    pool.join()
    scores = []
    for index in range(n_process):
        scores += temp_data[index].get()

    return scores

def get_scores_subproc(smiles, mode):
    scores = []
    mols = [MolFromSmiles(s) for s in smiles]
    oracle_QED = Oracle(name='QED')
    oracle_SA = Oracle(name='SA')

    if mode == "QED":
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle_QED([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "SA":
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle_SA([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "DRD2":
        oracle = Oracle(name='DRD2')
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "GSK3B":
        oracle = Oracle(name='GSK3B')
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "JNK3":
        oracle = Oracle(name='JNK3')
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "docking_PLPro_7JIR":
        for i in range(len(smiles)):
            if mols[i] != None:
                # docking_score = docking(smiles[i], receptor_file="data/targets/7jir+w2.pdbqt", box_center=[51.51, 32.16, -0.55])
                docking_score, unnormalized = docking(smiles[i], receptor_file="data/targets/7jir+w2.pdbqt", box_center=[51.51, 32.16, -0.55], return_raw=True)
                scores += [docking_score, unnormalized]
            else:
                scores += [-1.0, -1.0]

    elif mode == "docking_PLPro_7JIR_mpo":
        for i in range(len(smiles)):
            if mols[i] != None:
                # docking_score = docking(smiles[i], receptor_file="data/targets/7jir+w2.pdbqt", box_center=[51.51, 32.16, -0.55])
                docking_score, unnormalized = docking(smiles[i], receptor_file="data/targets/7jir+w2.pdbqt", box_center=[51.51, 32.16, -0.55], return_raw=True)
                qed = oracle_QED(smiles[i])
                sa = oracle_SA(smiles[i])  #(10 - oracle_SA(smiles[i])) / 9
                scores += [0.8 * docking_score + 0.1 * qed + 0.1 * (10 - sa) / 9, unnormalized, qed, sa]
                try:
                    ligand_mol_file = f"./docking/tmp/mol_{smiles[i]}.mol"
                    ligand_pdbqt_file = f"./docking/tmp/mol_{smiles[i]}.pdbqt"
                    docking_pdbqt_file = f"./docking/tmp/dock_{smiles[i]}.pdbqt"
                    for filename in [ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file]:
                        if os.path.exists(filename):
                            os.remove(filename)
                except:
                    pass
                
            else:
                scores += [-1.0, -1.0, 0.0, 0.0]

    elif mode == "docking_RdRp":
        for i in range(len(smiles)):
            if mols[i] != None:
                docking_score = docking(smiles[i], receptor_file="data/targets/RDRP.pdbqt", box_center=[93.88, 83.08, 97.29])
                scores += [docking_score]
            else:
                scores += [-1.0]

    elif mode == "docking_RdRp_mpo":
        for i in range(len(smiles)):
            if mols[i] != None:
                docking_score, unnormalized = docking(smiles[i], receptor_file="data/targets/RDRP.pdbqt", box_center=[93.88, 83.08, 97.29], return_raw=True)
                qed = oracle_QED(smiles[i])
                sa = oracle_SA(smiles[i])  #(10 - oracle_SA(smiles[i])) / 9
                scores += [0.8 * docking_score + 0.1 * qed + 0.1 * (10 - sa) / 9, unnormalized, qed, sa]
                try:
                    ligand_mol_file = f"./docking/tmp/mol_{smiles[i]}.mol"
                    ligand_pdbqt_file = f"./docking/tmp/mol_{smiles[i]}.pdbqt"
                    docking_pdbqt_file = f"./docking/tmp/dock_{smiles[i]}.pdbqt"
                    for filename in [ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file]:
                        if os.path.exists(filename):
                            os.remove(filename)
                except:
                    pass
                
            else:
                scores += [-1.0, -1.0, 0.0, 0.0]

    else:
        raise Exception("Scoring function undefined!")


    return scores


def sanitize_smiles_for_filename(smiles):
    """
    Sanitize SMILES string for use in file paths.
    Uses MD5 hash to create a unique, safe filename that avoids issues with
    special characters like /, \, :, *, ?, ", <, >, |, etc.
    """
    # Use MD5 hash to create a unique, safe filename
    safe_hash = hashlib.md5(smiles.encode('utf-8')).hexdigest()
    return safe_hash


def docking(smiles, receptor_file, box_center, box_size=[20, 20, 20], return_raw=False):
    if smiles == "":
        if return_raw:
            return -1., -1.
        return -1.0

    # Sanitize SMILES for use in file paths (avoid special chars like /, \, etc.)
    safe_name = sanitize_smiles_for_filename(smiles)
    ligand_mol_file = f"./docking/tmp/mol_{safe_name}.mol"
    ligand_pdbqt_file = f"./docking/tmp/mol_{safe_name}.pdbqt"
    docking_pdbqt_file = f"./docking/tmp/dock_{safe_name}.pdbqt"

    # 3D conformation of SMILES
    try:
        run_line = 'obabel -:%s --gen3D -O %s' % (smiles, ligand_mol_file)
        result = subprocess.check_output(run_line.split(), stderr=subprocess.STDOUT,
                    timeout=30, universal_newlines=True)
    except subprocess.TimeoutExpired as e:
        # Store error for debugging (will be caught by caller)
        import sys
        if hasattr(sys, '_docking_errors'):
            sys._docking_errors.append(f"OpenBabel timeout for {smiles[:50]}: {str(e)}")
        if return_raw:
            return -1., -1.
        return -1.0
    except subprocess.CalledProcessError as e:
        # Store error for debugging
        import sys
        if hasattr(sys, '_docking_errors'):
            sys._docking_errors.append(f"OpenBabel failed for {smiles[:50]}: returncode={e.returncode}, stderr={e.stderr[:200] if e.stderr else 'None'}")
        if return_raw:
            return -1., -1.
        return -1.0
    except Exception as e:
        # Store error for debugging
        import sys
        if hasattr(sys, '_docking_errors'):
            sys._docking_errors.append(f"OpenBabel exception for {smiles[:50]}: {type(e).__name__}: {str(e)}")
        if return_raw:
            return -1., -1.
        return -1.0

    # docking by quick vina
    try:
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = './docking/qvina/qvina02 --receptor %s --ligand %s --out %s' % (receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' % (box_center[0], box_center[1], box_center[2])
        run_line += ' --size_x %s --size_y %s --size_z %s' % (box_size[0], box_size[1], box_size[2])
        run_line += ' --cpu %d' % (4)
        run_line += ' --num_modes %d' % (10)
        run_line += ' --exhaustiveness %d ' % (4)
        result = subprocess.check_output(run_line.split(),
                                            stderr=subprocess.STDOUT,
                                            timeout=100, universal_newlines=True)
        result_lines = result.split('\n')
        affinity_list = list()
        check_result = False
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
            affinity_score = affinity_list[0]

        if return_raw:
            return reverse_sigmoid_transformation(affinity_score), affinity_score
        else:
            return reverse_sigmoid_transformation(affinity_score)

    except subprocess.TimeoutExpired as e:
        # Store error for debugging
        import sys
        if hasattr(sys, '_docking_errors'):
            sys._docking_errors.append(f"qvina02 timeout for {smiles[:50]}: {str(e)}")
        if return_raw:
            return -1., -1.
        return -1.0
    except subprocess.CalledProcessError as e:
        # Store error for debugging - capture full output
        import sys
        if hasattr(sys, '_docking_errors'):
            # Get full error output (stderr or stdout)
            full_output = ""
            if e.stderr:
                full_output = e.stderr
            elif e.stdout:
                full_output = e.stdout
            else:
                full_output = 'No error message'
            
            # Try to extract the actual error (skip citation messages)
            error_lines = full_output.split('\n')
            actual_errors = []
            for line in error_lines:
                if any(keyword in line.lower() for keyword in ['error', 'failed', 'cannot', 'unable', 'invalid', 'not found']):
                    actual_errors.append(line)
            
            if actual_errors:
                error_msg = '\n'.join(actual_errors[:5])  # First 5 error lines
            else:
                error_msg = full_output[:500]  # First 500 chars if no clear error lines
            
            sys._docking_errors.append(f"qvina02 failed for {smiles[:50]}: returncode={e.returncode}\nError: {error_msg}")
        if return_raw:
            return -1., -1.
        return -1.0
    except Exception as e:
        # Store error for debugging
        import sys
        if hasattr(sys, '_docking_errors'):
            sys._docking_errors.append(f"Docking exception for {smiles[:50]}: {type(e).__name__}: {str(e)}")
        if return_raw:
            return -1., -1.
        return -1.0


def reverse_sigmoid_transformation(original_score): 
    if original_score > 99.9:
        return -1.0 
    else: # return (0, 1)
        _low = -12
        _high = -8
        _k = 0.25
        def _reverse_sigmoid_formula(value, low, high, k) -> float:
            try:
                return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
            except:
                return 0

        transformed = _reverse_sigmoid_formula(original_score, _low, _high, _k) 
        return transformed