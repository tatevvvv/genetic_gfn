import os
import yaml
import random
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
# import tdc
# from tdc.generation import MolGen
import wandb
# from main.utils.chem import *
from botorch.utils.multi_objective.hypervolume import Hypervolume

from oracle.scorer.scorer import get_scores
from utils.metrics import compute_success, compute_diversity


class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
            self.weights = np.array(args.alpha_vector) 
        self.mol_buffer = mol_buffer
        # self.sa_scorer = tdc.Oracle(name = 'SA')
        # self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0
        self.current_div = 1.
        self.hypervolume = Hypervolume(ref_point=torch.zeros(len(args.objectives)))
        
        # Hit task tracking
        self.hit_count = 0
        self.is_hit_task = False  # Will be set based on objectives
        # Constraint ranges for hit task
        self.qed_min, self.qed_max = 0.5, 1.0
        self.sa_min, self.sa_max = 1.0, 5.0
        self.dock_min, self.dock_max = -20, -10
        
        # Hit task: CSV file for recording all molecules
        self.hit_task_csv_file = None
        if args and hasattr(args, 'output_dir'):
            csv_dir = args.output_dir
            os.makedirs(csv_dir, exist_ok=True)
            # CSV file will be set when task_label is set
        
    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))
    
    def record(self, smiles, qed_score, sa_score, dock_score_raw):
        """
        Record a molecule to CSV file for hit task.
        Format: smiles,DS,QED,SA,\n
        
        Args:
            smiles: SMILES string
            qed_score: QED score
            sa_score: SA score
            dock_score_raw: Raw docking score (DS)
        """
        if self.hit_task_csv_file is None:
            # Initialize CSV file when first molecule is recorded
            # Use a default name if task_label is not set yet
            if self.task_label:
                csv_filename = f"hit_task_molecules_{self.task_label}.csv"
            else:
                # Generate a unique filename based on timestamp or use default
                import time
                csv_filename = f"hit_task_molecules_{int(time.time())}.csv"
            self.hit_task_csv_file = os.path.join(self.args.output_dir, csv_filename)
            # Write header
            os.makedirs(os.path.dirname(self.hit_task_csv_file), exist_ok=True)
            with open(self.hit_task_csv_file, 'w') as f:
                f.write("smiles,DS,QED,SA,\n")
        
        # Append molecule data
        with open(self.hit_task_csv_file, 'a') as f:
            f.write(f"{smiles},{dock_score_raw:.6f},{qed_score:.6f},{sa_score:.6f},\n")

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            scores_each = [item[1][2] for item in temp_top100]  # normalized scores
            raw_scores_each = [item[1][3] if len(item[1]) > 3 else item[1][2] for item in temp_top100]  # raw scores
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    scores_each = [item[1][2] for item in temp_top100]  # normalized scores
                    raw_scores_each = [item[1][3] if len(item[1]) > 3 else item[1][2] for item in temp_top100]  # raw scores
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    scores_each = [item[1][2] for item in temp_top100]  # normalized scores
                    raw_scores_each = [item[1][3] if len(item[1]) > 3 else item[1][2] for item in temp_top100]  # raw scores
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)
                # For this case, we don't have scores_each, so use empty lists
                scores_each = []
                raw_scores_each = []
        
        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        # avg_sa = np.mean(self.sa_scorer(smis))
        # diversity_top100 = self.diversity_evaluator(smis)
        mols = []
        for s in smis:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    mols.append(mol)
            except:
                pass
        
        diversity_top100 = compute_diversity(mols)
        hv = self.hypervolume.compute(torch.tensor(scores_each))
            
        self.current_div = diversity_top100
        
        # Calculate hit ratio for hit task
        hit_ratio = self.hit_count / n_calls if n_calls > 0 else 0.0
        
        # Extract and calculate average property values
        log_str = f'{n_calls}/{self.max_oracle_calls} | '
        
        if self.is_hit_task and len(raw_scores_each) > 0:
            # Extract QED, SA, and docking scores from raw_scores_each
            qed_idx = self.objectives_lower.index('qed') if 'qed' in self.objectives_lower else None
            sa_idx = self.objectives_lower.index('sa') if 'sa' in self.objectives_lower else None
            dock_idx = None
            docking_targets = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
            for i, obj_lower in enumerate(self.objectives_lower):
                if obj_lower in docking_targets:
                    dock_idx = i
                    break
            
            # Calculate averages
            if qed_idx is not None:
                avg_qed = np.mean([raw_scores[qed_idx] if len(raw_scores) > qed_idx else 0.0 
                                  for raw_scores in raw_scores_each])
                log_str += f'avg_qed: {avg_qed:.3f} | '
            if sa_idx is not None:
                avg_sa = np.mean([raw_scores[sa_idx] if len(raw_scores) > sa_idx else 0.0 
                                 for raw_scores in raw_scores_each])
                log_str += f'avg_sa: {avg_sa:.3f} | '
            if dock_idx is not None:
                avg_dock = np.mean([raw_scores[dock_idx] if len(raw_scores) > dock_idx else 0.0 
                                    for raw_scores in raw_scores_each])
                log_str += f'avg_dock: {avg_dock:.3f} | '
            
            log_str += f'hits: {self.hit_count} | hit_ratio: {hit_ratio:.4f}'
        else:
            # For non-hit tasks, show standard metrics
            log_str += f'avg_top1: {avg_top1:.3f} | '
            log_str += f'avg_top10: {avg_top10:.3f} | '
            log_str += f'avg_top100: {avg_top100:.3f} | '
            log_str += f'hv: {hv:.3f} | '
            log_str += f'div: {diversity_top100:.3f}'
        
        print(log_str)

        try:
            if self.is_hit_task and len(raw_scores_each) > 0:
                # Log property values for hit task
                log_dict = {
                    "n_oracle": n_calls,
                    "diversity_top100": diversity_top100,
                }
                
                # Add property averages
                qed_idx = self.objectives_lower.index('qed') if 'qed' in self.objectives_lower else None
                sa_idx = self.objectives_lower.index('sa') if 'sa' in self.objectives_lower else None
                dock_idx = None
                docking_targets = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
                for i, obj_lower in enumerate(self.objectives_lower):
                    if obj_lower in docking_targets:
                        dock_idx = i
                        break
                
                if qed_idx is not None:
                    avg_qed = np.mean([raw_scores[qed_idx] if len(raw_scores) > qed_idx else 0.0 
                                      for raw_scores in raw_scores_each])
                    log_dict["avg_qed"] = avg_qed
                if sa_idx is not None:
                    avg_sa = np.mean([raw_scores[sa_idx] if len(raw_scores) > sa_idx else 0.0 
                                     for raw_scores in raw_scores_each])
                    log_dict["avg_sa"] = avg_sa
                if dock_idx is not None:
                    avg_dock = np.mean([raw_scores[dock_idx] if len(raw_scores) > dock_idx else 0.0 
                                       for raw_scores in raw_scores_each])
                    log_dict["avg_dock"] = avg_dock
                
                log_dict["hit_count"] = self.hit_count
                log_dict["hit_ratio"] = hit_ratio
            else:
                # Standard logging for non-hit tasks
                log_dict = {
                    "avg_top1": avg_top1, 
                    "avg_top10": avg_top10, 
                    "avg_top100": avg_top100, 
                    "hv": hv,
                    "diversity_top100": diversity_top100,
                    "n_oracle": n_calls,
                }
            
            wandb.log(log_dict)
        except:
            pass


    def __len__(self):
        return len(self.mol_buffer)
    
    
    def set_objectives(self, objectives, alpha_vector):
        self.objectives = objectives
        self.weights = np.array(alpha_vector)
        # Store lowercase version for consistent comparison
        self.objectives_lower = [obj.lower() for obj in objectives]
        
        # Check if this is a hit task (qed, sa, and a docking target)
        if len(objectives) == 3:
            if 'qed' in self.objectives_lower and 'sa' in self.objectives_lower:
                # Check if third objective is a docking target
                docking_targets = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
                if any(target in self.objectives_lower for target in docking_targets):
                    self.is_hit_task = True
    
    def _check_hit(self, qed_score, sa_score, dock_score_raw):
        """Check if molecule satisfies all hit task constraints."""
        qed_ok = self.qed_min <= qed_score <= self.qed_max
        sa_ok = self.sa_min <= sa_score <= self.sa_max
        dock_ok = self.dock_min <= dock_score_raw <= self.dock_max
        return qed_ok and sa_ok and dock_ok
    
    def moo_evaluator(self, mol):
        """
        Evaluate molecule for multi-objective optimization.
        For hit task: R(x) = DSd(x) × QED(x) × SAc(x) where all are normalized to [0, 1]
        Otherwise: weighted sum
        """
        scores = np.zeros(len(self.objectives))
        raw_scores = np.zeros(len(self.objectives))  # Store raw scores for hit checking
        
        # Get scores (normalized for hit task, raw otherwise)
        # For hit task, we need both normalized and raw scores
        if self.is_hit_task:
            # Find indices for each objective type
            qed_idx = self.objectives_lower.index('qed') if 'qed' in self.objectives_lower else None
            sa_idx = self.objectives_lower.index('sa') if 'sa' in self.objectives_lower else None
            dock_idx = None
            docking_targets = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
            for i, obj_lower in enumerate(self.objectives_lower):
                if obj_lower in docking_targets:
                    dock_idx = i
                    break
            
            # Get scores for each objective
            for i, obj in enumerate(self.objectives):
                if i == dock_idx:
                    # For docking, get both normalized and raw in one call
                    norm_scores, raw_scores_list = get_scores(
                        obj, [mol], return_normalized=True, return_raw_scores=True
                    )
                    scores[i] = norm_scores[0]
                    raw_scores[i] = raw_scores_list[0]
                elif i == qed_idx:
                    # QED: get raw score (already in [0, 1])
                    raw_score = get_scores(obj, [mol], return_normalized=False)[0]
                    raw_scores[i] = raw_score
                    scores[i] = raw_score
                elif i == sa_idx:
                    # SA: get raw score and normalize
                    raw_score = get_scores(obj, [mol], return_normalized=False)[0]
                    raw_scores[i] = raw_score
                    from oracle.scorer.scorer import normalize_sa_score
                    scores[i] = normalize_sa_score(raw_score)
                else:
                    # Fallback (shouldn't happen in hit task)
                    raw_score = get_scores(obj, [mol], return_normalized=False)[0]
                    raw_scores[i] = raw_score
                    scores[i] = raw_score
        else:
            # Standard MOO: just get raw scores
            for i, obj in enumerate(self.objectives):
                score = get_scores(obj, [mol], return_normalized=False)[0]
                scores[i] = score
                raw_scores[i] = score
        
        # Compute reward
        if self.is_hit_task:
            # Hit task: R(x) = DSd(x) × QED(x) × SAc(x)
            # All scores should already be normalized to [0, 1]
            reward = np.prod(scores)
            
            # Check if this is a hit (satisfies all constraints)
            # Use the indices we already found above
            qed_idx = self.objectives_lower.index('qed') if 'qed' in self.objectives_lower else None
            sa_idx = self.objectives_lower.index('sa') if 'sa' in self.objectives_lower else None
            dock_idx = None
            docking_targets = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
            for i, obj_lower in enumerate(self.objectives_lower):
                if obj_lower in docking_targets:
                    dock_idx = i
                    break
            
            if qed_idx is not None and sa_idx is not None and dock_idx is not None:
                is_hit = self._check_hit(
                    raw_scores[qed_idx], 
                    raw_scores[sa_idx], 
                    raw_scores[dock_idx]
                )
                if is_hit:
                    self.hit_count += 1
                
                # Record ALL molecules to CSV for hit task (not just hits)
                # Get SMILES from mol object
                from rdkit import Chem
                try:
                    mol_smiles = Chem.MolToSmiles(mol)
                    self.record(
                        smiles=mol_smiles,
                        qed_score=raw_scores[qed_idx],
                        sa_score=raw_scores[sa_idx],
                        dock_score_raw=raw_scores[dock_idx]
                    )
                except:
                    pass  # Skip if SMILES conversion fails
        else:
            # Standard multi-objective: weighted sum
            reward = np.matmul(scores, self.weights.reshape(-1, 1))
        
        # Return reward, normalized scores, and raw scores (for hit task)
        if self.is_hit_task:
            return reward, scores, raw_scores
        else:
            return reward, scores, scores  # For non-hit tasks, raw and normalized are the same

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass
            else:
                reward, scores, raw_scores = self.moo_evaluator(mol)
                # Store: [reward, oracle_call_number, normalized_scores, raw_scores]
                self.mol_buffer[smi] = [float(reward), len(self.mol_buffer)+1, scores, raw_scores]
            return self.mol_buffer[smi][0]
    
    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)
        else:  ### a string of SMILES 
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls
    
class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = "Default"
        self.args = args
        self.n_jobs = args.n_jobs
        # self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = args.smi_file
        self.oracle = Oracle(args=self.args)
        # if self.smi_file is not None:
        #     self.all_smiles = self.load_smiles_from_file(self.smi_file)
        # else:
        #     data = MolGen(name = 'ZINC')
        #     self.all_smiles = data.get_data()['smiles'].tolist()
            
        # self.sa_scorer = tdc.Oracle(name = 'SA')
        # self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        # self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters = ['PAINS', 'SureChEMBL', 'Glaxo'], property_filters_flag = False)

    # def load_smiles_from_file(self, file_name):
    #     with open(file_name) as f:
    #         return self.pool(delayed(canonicalize)(s.strip()) for s in f)
            
    def sanitize(self, mol_list):
        new_mol_list = []
        smiles_set = set()
        for mol in mol_list:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    print('bad smiles')
        return new_mol_list
        
    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish)
    
    def log_result(self):

        print(f"Logging final results...")

        # import ipdb; ipdb.set_trace()

        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        assert len(self.mol_buffer) > 0 

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > 10000:
            results = results[:10000]
        
        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))
        
        # Currently logging the top-100 moelcules, will update to PDD selection later
        # data = [[i+1, results_all_level[-1][i][1][0], results_all_level[-1][i][1][1], \
        #         wandb.Image(Draw.MolToImage(Chem.MolFromSmiles(results_all_level[-1][i][0]))), results_all_level[-1][i][0]] for i in range(100)]
        # columns = ["Rank", "Score", "#Oracle", "Image", "SMILES"]
        # wandb.log({"Top 100 Molecules": wandb.Table(data=data, columns=columns)})
        
        # # Log batch metrics at various oracle calls
        # data = [[log_num_oracles[i]] + self._analyze_results(r) for i, r in enumerate(results_all_level)]
        # columns = ["#Oracle", "avg_top100", "avg_top10", "avg_top1", "Diversity", "avg_SA", "%Pass", "Top-1 Pass"]
        # wandb.log({"Batch metrics at various level": wandb.Table(data=data, columns=columns)})
        
    def save_result(self, suffix=None):

        print(f"Saving molecules...")
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)
    
    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        if len(smis_pass) == 0:
            top1_pass = -1
        else:
            top1_pass = np.max([scores_dict[s] for s in smis_pass])
        return [np.mean(scores), 
                np.mean(scores[:10]), 
                np.max(scores), 
                self.diversity_evaluator(smis), 
                np.mean(self.sa_scorer(smis)), 
                float(len(smis_pass) / 100), 
                top1_pass]

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args, mol_buffer={})

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish
        
    def _optimize(self, oracle, config):
        raise NotImplementedError
            
    def optimize(self, oracle, config, seed=0, project="test"):
        # run = wandb.init(project=project, config=config, reinit=True, entity="mol_opt")
        if self.args.wandb != 'disabled':
            project = 'pmo' if self.args.method.startswith('genetic_gfn') else 'pmo_baselines'
            run = wandb.init(project=project, group=oracle.name, config=config, reinit=True)
            wandb.config.oracle = oracle.name
            wandb.config.method = self.args.method
            wandb.run.name = oracle.name + "_" + self.args.method + "_" + self.args.run_name + "_" + str(seed) + "_" + wandb.run.id
        # wandb.run.name = self.model_name + "_" + oracle.name + "_" + wandb.run.id
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        self.seed = seed
        # Set task_label for CSV file naming (use run_name if available, otherwise generate from seed)
        if hasattr(self.args, 'run_name') and self.args.run_name != "default":
            self.oracle.task_label = self.args.run_name + "_seed" + str(seed)
        else:
            self.oracle.task_label = self.model_name + "_seed" + str(seed)
        self._optimize(oracle, config)
        if self.args.log_results:
            self.log_result()
        self.save_result(self.model_name + "_" + str(len(oracle[0])) + "_" + str(seed))
        # self.reset()
        if self.args.wandb != 'disabled':
            run.finish()
        self.reset()

    def production(self, oracle, config, num_runs=5, project="production"):
        # Production seed pool (hard-coded).
        # NOTE: Intentionally fixed to seeds 0-9 so `--n_runs 10` runs exactly these.
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        if num_runs is None:
            num_runs = len(seeds)

        if num_runs > len(seeds):
            raise ValueError(f"Requested num_runs={num_runs} but only {len(seeds)} seeds are available/provided.")

        seeds = seeds[:num_runs]
        for seed in seeds:
            self.optimize(oracle, config, seed, project)
            self.reset()

