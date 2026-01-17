#!/usr/bin/env python
"""
Scorer module for multi-objective optimization.
Supports QED, SA, and docking scores for various protein targets.
"""
import os
import sys
import numpy as np
from tdc import Oracle as TDCOracle
from rdkit import Chem
from rdkit.Chem import MolFromSmiles

# Use local docking module (self-contained, no directory changes needed)
# Import from parent directory (oracle/docking.py)
import sys
import os
_oracle_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _oracle_dir not in sys.path:
    sys.path.insert(0, _oracle_dir)
from docking import docking

# Target configurations for docking
TARGET_CONFIGS = {
    'parp1': {
        'box_center': [26.413, 11.282, 27.238],
        'box_size': [18.521, 17.479, 19.995],
        'receptor_file': 'parp1.pdbqt'
    },
    'fa7': {
        'box_center': [10.131, 41.879, 32.097],
        'box_size': [20.673, 20.198, 21.362],
        'receptor_file': 'fa7.pdbqt'
    },
    '5ht1b': {
        'box_center': [-26.602, 5.277, 17.898],
        'box_size': [22.5, 22.5, 22.5],
        'receptor_file': '5ht1b.pdbqt'
    },
    'braf': {
        'box_center': [84.194, 6.949, -7.081],
        'box_size': [22.032, 19.211, 14.106],
        'receptor_file': 'braf.pdbqt'
    },
    'jak2': {
        'box_center': [114.758, 65.496, 11.345],
        'box_size': [19.033, 17.929, 20.283],
        'receptor_file': 'jak2.pdbqt'
    }
}

# Get base path for targets
# __file__ is at: multi_objective/oracle/scorer/scorer.py
# Go up 3 levels to get to multi_objective/, then targets/ is in multi_objective/
# For sars_cov2, we need project root, so go up 4 levels
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
MULTI_OBJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
TARGETS_DIR = os.path.join(MULTI_OBJ_DIR, 'targets')

# Initialize TDC oracles (lazy loading)
_qed_oracle = None
_sa_oracle = None

def _get_qed_oracle():
    global _qed_oracle
    if _qed_oracle is None:
        _qed_oracle = TDCOracle(name='QED')
    return _qed_oracle

def _get_sa_oracle():
    global _sa_oracle
    if _sa_oracle is None:
        _sa_oracle = TDCOracle(name='SA')
    return _sa_oracle


def normalize_docking_score(raw_score, dock_min=-20.0, dock_max=-10.0):
    """
    Normalize docking score to [0, 1].
    
    Args:
        raw_score: Raw Vina docking score (negative, typically -20 to -10)
        dock_min: Minimum docking score (most negative, worst binding)
        dock_max: Maximum docking score (least negative, best binding)
    
    Returns:
        Normalized score in [0, 1], where 1 is best binding
    """
    if raw_score < -100 or raw_score > 0:  # Invalid score
        return 0.0
    # Normalize: better binding (less negative) -> higher score
    # Map [-20, -10] to [0, 1]
    normalized = (raw_score - dock_min) / (dock_max - dock_min)
    return max(0.0, min(1.0, normalized))  # Clamp to [0, 1]


def normalize_sa_score(sa_score, sa_min=1.0, sa_max=5.0):
    """
    Normalize SA score to [0, 1] for the hit task range.
    
    Args:
        sa_score: SA score (1.0 to 10.0, lower is better)
        sa_min: Minimum SA in desired range (1.0)
        sa_max: Maximum SA in desired range (5.0)
    
    Returns:
        Normalized score in [0, 1], where 1 is best (SA=1.0) and 0 is worst (SA=5.0)
    """
    if sa_score < 1.0 or sa_score > 10.0:  # Invalid score
        return 0.0
    # For SA in [1.0, 5.0], normalize so SA=1.0 -> 1.0, SA=5.0 -> 0.0
    # We want lower SA to be better, so invert the scale
    if sa_score <= sa_max:
        normalized = 1.0 - (sa_score - sa_min) / (sa_max - sa_min)
    else:
        # SA > 5.0, penalize
        normalized = 0.0
    return max(0.0, min(1.0, normalized))  # Clamp to [0, 1]


def get_scores(objective_name, mols, return_normalized=False, return_raw_scores=False):
    """
    Get scores for a list of molecules for a given objective.
    
    Args:
        objective_name: String name of the objective (e.g., 'qed', 'sa', 'parp1', 'fa7', etc.)
        mols: List of RDKit molecule objects or SMILES strings
        return_normalized: If True, return normalized scores for hit task (DSd, SAc in [0,1])
        
    Returns:
        List of scores (one per molecule). 
        If return_normalized=True, returns normalized scores.
        If return_raw_scores=True, returns tuple (normalized_scores, raw_scores).
    """
    if not mols:
        return []
    
    # Convert molecules to SMILES if needed
    smiles_list = []
    for mol in mols:
        if isinstance(mol, str):
            smiles_list.append(mol)
        else:
            try:
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)
            except:
                smiles_list.append(None)
    
    scores = []
    
    # Handle different objective types
    objective_lower = objective_name.lower()
    
    if objective_lower == 'qed':
        oracle = _get_qed_oracle()
        for smiles in smiles_list:
            if smiles:
                try:
                    score = oracle([smiles])[0]
                    scores.append(float(score))  # Already in [0, 1]
                except:
                    scores.append(0.0)
            else:
                scores.append(0.0)
    
    elif objective_lower == 'sa':
        oracle = _get_sa_oracle()
        for smiles in smiles_list:
            if smiles:
                try:
                    score = oracle([smiles])[0]
                    if return_normalized:
                        # Normalize SA to [0, 1] for hit task
                        score = normalize_sa_score(score)
                    scores.append(float(score))
                except:
                    scores.append(0.0)
            else:
                scores.append(0.0)
    
    elif objective_lower in TARGET_CONFIGS:
        # Docking score for a specific target
        config = TARGET_CONFIGS[objective_lower]
        receptor_file = os.path.join(TARGETS_DIR, config['receptor_file'])
        
        # Convert to absolute path (we'll change directories for docking)
        receptor_file = os.path.abspath(receptor_file)
        
        # Check if receptor file exists
        if not os.path.exists(receptor_file):
            print(f"Warning: Receptor file not found: {receptor_file}")
            if return_raw_scores:
                return [0.0] * len(smiles_list), [-1.0] * len(smiles_list)
            return [0.0] * len(smiles_list)
        
        raw_scores_list = []
        for smiles in smiles_list:
            if smiles:
                try:
                    # Verify receptor file exists (use absolute path)
                    if not os.path.exists(receptor_file):
                        raise FileNotFoundError(f"Receptor file not found: {receptor_file}")
                    
                    # Find qvina02 - try multiple locations
                    qvina_path = None
                    possible_paths = [
                        os.path.join(MULTI_OBJ_DIR, 'docking', 'qvina', 'qvina02'),
                        # Bundled binary in this repo (some setups place qvina02 under targets/)
                        os.path.join(MULTI_OBJ_DIR, 'targets', 'qvina02'),
                        os.path.join(PROJECT_ROOT, 'sars_cov2', 'genetic_gfn', 'docking', 'qvina', 'qvina02'),
                        os.path.join(PROJECT_ROOT, 'docking', 'qvina', 'qvina02'),
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            qvina_path = path
                            break
                    
                    if qvina_path is None:
                        # Try to find in PATH
                        import shutil
                        qvina_in_path = shutil.which('qvina02')
                        if qvina_in_path:
                            qvina_path = qvina_in_path
                    
                    if qvina_path is None:
                        raise FileNotFoundError(f"qvina02 not found. Checked: {possible_paths}")

                    # Best-effort: ensure executable bit is set (common when binaries are copied into repos)
                    try:
                        if os.path.exists(qvina_path) and not os.access(qvina_path, os.X_OK):
                            os.chmod(qvina_path, 0o755)
                    except Exception:
                        pass
                    
                    # Set up docking tmp directory.
                    # When running many processes in parallel on one node, a shared tmp directory
                    # can cause file collisions. Allow overriding via DOCKING_TMP_DIR.
                    docking_tmp_dir = os.environ.get("DOCKING_TMP_DIR") or os.path.join(MULTI_OBJ_DIR, 'docking', 'tmp')
                    os.makedirs(docking_tmp_dir, exist_ok=True)
                    
                    # Call docking with error capture (no directory change needed)
                    # NOTE: docking(..., return_raw=True) returns (transformed_score_in_[0,1], raw_affinity_negative)
                    docking_score_raw = -1.0
                    try:
                        docking_score_transformed, docking_score_raw = docking(
                            smiles,
                            receptor_file=receptor_file,
                            box_center=config['box_center'],
                            box_size=config['box_size'],
                            return_raw=True,
                            qvina_path=qvina_path,
                            docking_tmp_dir=docking_tmp_dir
                        )
                    except Exception:
                        docking_score_raw = -1.0
                    
                    raw_scores_list.append(float(docking_score_raw))
                    
                    if return_normalized:
                        # Hit-task docking normalization:
                        # Use the reverse-sigmoid transformed docking score in [0, 1] returned by docking().
                        # NOTE: is_hit is still computed elsewhere using the raw (negative) affinity.
                        scores.append(float(docking_score_transformed))
                    else:
                        scores.append(float(docking_score_raw))
                        
                except Exception as e:
                    # Handle docking failures silently
                    raw_scores_list.append(-1.0)  # Use -1.0 to indicate failure
                    scores.append(0.0)
            else:
                raw_scores_list.append(0.0)
                scores.append(0.0)
        
        if return_raw_scores:
            return scores, raw_scores_list
    
    elif objective_lower in ['gsk3b', 'jnk3', 'drd2']:
        # Other TDC oracles
        try:
            oracle = TDCOracle(name=objective_name.upper())
            for smiles in smiles_list:
                if smiles:
                    try:
                        score = oracle([smiles])[0]
                        scores.append(float(score))
                    except:
                        scores.append(0.0)
                else:
                    scores.append(0.0)
        except:
            scores = [0.0] * len(smiles_list)
    
    else:
        raise ValueError(f"Unknown objective: {objective_name}. Available objectives: "
                         f"qed, sa, gsk3b, jnk3, drd2, parp1, fa7, 5ht1b, braf, jak2")
    
    if return_raw_scores:
        # For non-docking objectives, raw and normalized are the same
        return scores, scores.copy()
    return scores

