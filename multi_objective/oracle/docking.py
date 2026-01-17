#!/usr/bin/env python
"""
Self-contained docking module for multi-objective optimization.
Duplicated from sars_cov2 to avoid directory dependencies.
"""
import os
import hashlib
import subprocess
import contextlib

try:
    from openbabel import pybel
except ImportError:
    try:
        import pybel
    except ImportError:
        raise ImportError("Please install openbabel: conda install -c conda-forge openbabel or pip install openbabel-wheel")


def sanitize_smiles_for_filename(smiles):
    """
    Sanitize SMILES string for use in file paths.
    Uses MD5 hash to create a unique, safe filename that avoids issues with
    special characters like /, \, :, *, ?, ", <, >, |, etc.
    """
    # Use MD5 hash to create a unique, safe filename
    safe_hash = hashlib.md5(smiles.encode('utf-8')).hexdigest()
    return safe_hash


def reverse_sigmoid_transformation(original_score):
    """
    Transform docking affinity score using reverse sigmoid.
    
    Args:
        original_score: Raw docking affinity in kcal/mol (negative value)
        
    Returns:
        Transformed score in [0, 1] range
    """
    if original_score > 99.9:
        return -1.0 
    else:  # return (0, 1)
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


def docking(smiles, receptor_file, box_center, box_size=[20, 20, 20], return_raw=False, 
            qvina_path=None, docking_tmp_dir=None):
    """
    Perform molecular docking using qvina02.
    
    Args:
        smiles: SMILES string of the molecule
        receptor_file: Path to receptor PDBQT file (absolute path)
        box_center: [x, y, z] coordinates of docking box center
        box_size: [width, height, depth] of docking box
        return_raw: If True, return both transformed and raw scores
        qvina_path: Path to qvina02 executable (if None, will try to find it)
        docking_tmp_dir: Directory for temporary docking files (if None, uses ./docking/tmp)
        
    Returns:
        If return_raw=True: (transformed_score, raw_affinity)
        If return_raw=False: transformed_score
        Returns -1.0 on failure
    """
    if smiles == "":
        if return_raw:
            return -1., -1.
        return -1.0

    # Get paths - use absolute paths to avoid directory changes
    if docking_tmp_dir is None:
        # Default: use docking/tmp in the same directory as this file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        docking_tmp_dir = os.path.join(base_dir, 'docking', 'tmp')
    
    os.makedirs(docking_tmp_dir, exist_ok=True)
    
    # Sanitize SMILES for use in file paths (avoid special chars like /, \, etc.)
    safe_name = sanitize_smiles_for_filename(smiles)
    ligand_mol_file = os.path.join(docking_tmp_dir, f"mol_{safe_name}.mol")
    ligand_pdbqt_file = os.path.join(docking_tmp_dir, f"mol_{safe_name}.pdbqt")
    docking_pdbqt_file = os.path.join(docking_tmp_dir, f"dock_{safe_name}.pdbqt")
    
    # Find qvina02 if not provided
    if qvina_path is None:
        # Try to find qvina02 in common locations
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        possible_paths = [
            os.path.join(base_dir, 'docking', 'qvina', 'qvina02'),
            # Bundled binary in this repo (some setups place qvina02 under targets/)
            os.path.join(base_dir, 'targets', 'qvina02'),
            os.path.join(base_dir, 'sars_cov2', 'genetic_gfn', 'docking', 'qvina', 'qvina02'),
            './docking/qvina/qvina02',
            'qvina02',  # In PATH
        ]
        for path in possible_paths:
            if os.path.exists(path) or (path == 'qvina02' and os.system(f'which {path}') == 0):
                qvina_path = path
                break
        
        if qvina_path is None:
            raise FileNotFoundError("qvina02 executable not found. Please specify qvina_path.")
    
    # Make qvina_path absolute if it's a file path
    if not os.path.isabs(qvina_path) and os.path.exists(qvina_path):
        qvina_path = os.path.abspath(qvina_path)

    # Best-effort: ensure executable bit is set (common when binaries are copied into repos)
    try:
        if os.path.exists(qvina_path) and not os.access(qvina_path, os.X_OK):
            os.chmod(qvina_path, 0o755)
    except Exception:
        pass
    
    # Make receptor_file absolute
    if not os.path.isabs(receptor_file):
        receptor_file = os.path.abspath(receptor_file)

    # 3D conformation of SMILES using OpenBabel
    try:
        run_line = f'obabel -:"{smiles}" --gen3D -O {ligand_mol_file}'
        result = subprocess.check_output(run_line, shell=True, stderr=subprocess.STDOUT,
                    timeout=30, universal_newlines=True)
    except subprocess.TimeoutExpired as e:
        # Clean up any partial files
        with contextlib.suppress(FileNotFoundError, OSError):
            if os.path.exists(ligand_mol_file):
                os.remove(ligand_mol_file)
        if return_raw:
            return -1., -1.
        return -1.0
    except subprocess.CalledProcessError as e:
        # Clean up any partial files
        with contextlib.suppress(FileNotFoundError, OSError):
            if os.path.exists(ligand_mol_file):
                os.remove(ligand_mol_file)
        if return_raw:
            return -1., -1.
        return -1.0
    except Exception as e:
        # Clean up any partial files
        with contextlib.suppress(FileNotFoundError, OSError):
            if 'ligand_mol_file' in locals() and os.path.exists(ligand_mol_file):
                os.remove(ligand_mol_file)
        if return_raw:
            return -1., -1.
        return -1.0

    # Docking by quick vina
    try:
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        
        # Build qvina02 command with absolute paths
        run_line = f'{qvina_path} --receptor {receptor_file} --ligand {ligand_pdbqt_file} --out {docking_pdbqt_file}'
        run_line += f' --center_x {box_center[0]} --center_y {box_center[1]} --center_z {box_center[2]}'
        run_line += f' --size_x {box_size[0]} --size_y {box_size[1]} --size_z {box_size[2]}'
        run_line += f' --cpu 4 --num_modes 10 --exhaustiveness 4'
        
        result = subprocess.check_output(run_line, shell=True, stderr=subprocess.STDOUT,
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
            if not lis or not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
        
        if not affinity_list:
            raise ValueError("No affinity scores found in qvina02 output")
        
        affinity_score = affinity_list[0]

        # Clean up temporary files
        with contextlib.suppress(FileNotFoundError, OSError):
            os.remove(ligand_mol_file)
            os.remove(ligand_pdbqt_file)
            os.remove(docking_pdbqt_file)

        if return_raw:
            return reverse_sigmoid_transformation(affinity_score), affinity_score
        else:
            return reverse_sigmoid_transformation(affinity_score)

    except subprocess.TimeoutExpired as e:
        # Clean up temporary files on error
        with contextlib.suppress(FileNotFoundError, OSError):
            if 'ligand_mol_file' in locals():
                os.remove(ligand_mol_file)
            if 'ligand_pdbqt_file' in locals():
                os.remove(ligand_pdbqt_file)
            if 'docking_pdbqt_file' in locals() and os.path.exists(docking_pdbqt_file):
                os.remove(docking_pdbqt_file)
        if return_raw:
            return -1., -1.
        return -1.0
    except subprocess.CalledProcessError as e:
        # Clean up temporary files on error
        with contextlib.suppress(FileNotFoundError, OSError):
            if 'ligand_mol_file' in locals():
                os.remove(ligand_mol_file)
            if 'ligand_pdbqt_file' in locals():
                os.remove(ligand_pdbqt_file)
            if 'docking_pdbqt_file' in locals() and os.path.exists(docking_pdbqt_file):
                os.remove(docking_pdbqt_file)
        if return_raw:
            return -1., -1.
        return -1.0
    except Exception as e:
        # Clean up temporary files on error
        with contextlib.suppress(FileNotFoundError, OSError):
            if 'ligand_mol_file' in locals():
                os.remove(ligand_mol_file)
            if 'ligand_pdbqt_file' in locals():
                os.remove(ligand_pdbqt_file)
            if 'docking_pdbqt_file' in locals() and os.path.exists(docking_pdbqt_file):
                os.remove(docking_pdbqt_file)
        if return_raw:
            return -1., -1.
        return -1.0

