import os
import pySuStaIn
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

# This class provides an end-to-end pipeline to run the SuStaIn algorithm on neuroimaging data.
# It manages the entire workflow: from preprocessing the raw biomarker data to training the MCMC model for discovering distinct phenotypes. 
# Finally, it helps decode the expected temporal progression of these abnormalities.
# Author: Antonio Scardace

class SustainEngine:

    # Initializes the engine by setting up the dataset, defining the target features, creating the
    # necessary output directories, and configuring the Z-score mappings for the SuStaIn model.

    def __init__(self, df: pd.DataFrame, tau_cols: list[str], vol_cols: list[str], output_dir: str, z_map: dict[str, tuple[list[int], int]]) -> None:
        
        self.df = df.copy()
        self.tau_cols = tau_cols
        self.vol_cols = vol_cols
        self.final_biomarkers = tau_cols + vol_cols
        self.output_dir = output_dir
        self.z_map = z_map
        self.data_matrix: np.ndarray | None = None
        self.Z_vals: np.ndarray | None = None
        self.Z_max: np.ndarray | None = None

        os.makedirs(os.path.join(self.output_dir, 'pickle_files'), exist_ok=True)

    # This method applies normative modeling using healthy controls (CN) as a reference. 
    # It orchestrates internal helpers to remove confounding effects and prepare Z-scores.

    def apply_normative_modeling(self) -> None:
        residuals = self.__compute_residuals()
        self.__compute_z_scores(residuals)
        self.__initialize_z_thresholds()

    def __compute_residuals(self) -> pd.DataFrame:
        X_cov = self.df[['age', 'sex', 'total_intracranial_volume']].rename(columns={'total_intracranial_volume': 'ICV'})
        X_cov['sex'] = X_cov['sex'].map({'M': 0, 'F': 1})
        cn_mask = self.df['diagnosis'] == 'CN'
        reg = LinearRegression().fit(X_cov[cn_mask], self.df.loc[cn_mask, self.final_biomarkers])
        return self.df[self.final_biomarkers] - reg.predict(X_cov)

    def __compute_z_scores(self, res: pd.DataFrame) -> None:
        cn_mask = self.df['diagnosis'] == 'CN'
        z_scores = (res - res[cn_mask].mean()) / res[cn_mask].std()
        z_scores[self.vol_cols] *= -1
        self.data_matrix = z_scores.values

    def __initialize_z_thresholds(self) -> None:
        n_tau, n_vol = len(self.tau_cols), len(self.vol_cols)
        self.Z_vals = np.array([self.z_map['tau']['stages']] * n_tau + [self.z_map['vol']['stages']] * n_vol)
        self.Z_max = np.array([self.z_map['tau']['max_score']] * n_tau + [self.z_map['vol']['max_score']] * n_vol)
    
    # This method trains the final SuStaIn model using the optimal number of subtypes. 
    # It then assigns a specific disease subtype and a progression stage to each subject in the dataset.

    def fit_and_assign(self, n_subtypes: int, iterations: int) -> tuple[pd.DataFrame, np.ndarray]:
        
        print(f"Running final MCMC training on {n_subtypes} latent trajectories...")
        sustain_model = pySuStaIn.ZscoreSustain(
            self.data_matrix, self.Z_vals, self.Z_max,
            biomarker_labels=self.final_biomarkers,
            N_startpoints=10, N_S_max=n_subtypes, N_iterations_MCMC=iterations,
            output_folder=self.output_dir, dataset_name='AD_Final_Model', use_parallel_startpoints=False
        )
        
        samples_sequence, _, ml_subtype, _, ml_stage, _, _ = sustain_model.run_sustain_algorithm()
        self.df['sustain_subtype'], self.df['sustain_stage'] = ml_subtype, ml_stage
        return self.df, samples_sequence

    # This method analyzes the MCMC sampling sequence.
    # It determines the most probable ordering of biomarker abnormalities for a specific disease subtype.

    def get_biomarker_order(self, samples_sequence: np.ndarray, subtype: int) -> tuple[np.ndarray, list[str], np.ndarray]:

        num_biomarkers = len(self.final_biomarkers)
        num_events = num_biomarkers * 3
        subtype_sequences = samples_sequence[:, :, subtype]
        expected_stages = np.zeros(num_biomarkers)

        for bio_idx in range(num_biomarkers):
            probs = np.mean(subtype_sequences == bio_idx, axis=0)
            if np.sum(probs) > 0:
                expected_stages[bio_idx] = np.average(np.arange(num_events), weights=probs)

        ordered_idx = np.argsort(expected_stages)
        ordered_labels = [self.final_biomarkers[i] for i in ordered_idx]
        return ordered_idx, ordered_labels, expected_stages