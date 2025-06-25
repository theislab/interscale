import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from anndata import AnnData


def split_adata_patient_stratified(adata: AnnData, patient_obs: str, stratify_obs: str, 
                                  val_size: float = 0.1, test_size: float = 0, 
                                  seed: int = 40, split_key: str = 'split', 
                                  return_summary: bool = True) -> AnnData:
    """
    Split the AnnData object into train, val, and optionally test sets with patient-specific assignment
    and stratification to ensure balanced distribution across stratification groups.
    
    Parameters:
    - adata: The AnnData object to split.
    - patient_obs: The column in .obs that indicates patient IDs.
    - stratify_obs: The column in .obs to stratify by (e.g., condition, cell type).
    - val_size: The proportion of patients to include in the validation split.
    - test_size: The proportion of patients to include in the test split.
    - seed: Random seed for reproducibility.
    - split_key: Name of the column to store split assignments.
    - return_summary: Whether to print summary statistics.

    Returns:
    -------
        adata: AnnData
            adata with new .obs column '{split_key}' with values 'train', 'val', 'test'.
    """
    np.random.seed(seed)
    
    # Get unique patients and their stratification labels
    patient_stratify_df = adata.obs[[patient_obs, stratify_obs]].drop_duplicates()
    
    if len(patient_stratify_df) != len(patient_stratify_df[patient_obs].unique()):
        raise ValueError(f"Patients have multiple {stratify_obs} labels. Each patient should have a single stratification label.")
    
    # Create a mapping from patient to stratification label
    patient_to_stratify = dict(zip(patient_stratify_df[patient_obs], patient_stratify_df[stratify_obs]))
    
    # Get unique stratification labels
    stratify_labels = patient_stratify_df[stratify_obs].unique()
    print(f"Stratification labels: {stratify_labels}")
    
    # Initialize split assignments
    adata.obs[split_key] = None
    
    # Split patients within each stratification group
    train_patients = []
    val_patients = []
    test_patients = []
    
    for label in stratify_labels:
        # Get patients for this stratification label
        label_patients = patient_stratify_df[patient_stratify_df[stratify_obs] == label][patient_obs].tolist()
        
        if len(label_patients) < 2:
            print(f"Warning: Only {len(label_patients)} patient(s) for stratification label '{label}'. All assigned to train.")
            train_patients.extend(label_patients)
            continue
        
        # Split patients for this label
        if test_size > 0:
            # Split into train, val, test
            temp_size = val_size + test_size
            if len(label_patients) * temp_size < 1:
                print(f"Warning: Not enough patients for label '{label}' to create val/test sets. All assigned to train.")
                train_patients.extend(label_patients)
                continue
                
            train_temp, temp_patients = train_test_split(
                label_patients, 
                test_size=temp_size, 
                random_state=seed
            )
            
            # Split temp into val and test
            relative_val_size = val_size / temp_size
            val_temp, test_temp = train_test_split(
                temp_patients,
                test_size=1 - relative_val_size,
                random_state=seed
            )
            
            train_patients.extend(train_temp)
            val_patients.extend(val_temp)
            test_patients.extend(test_temp)
        else:
            # Split into train and val only
            if len(label_patients) * val_size < 1:
                print(f"Warning: Not enough patients for label '{label}' to create val set. All assigned to train.")
                train_patients.extend(label_patients)
                continue
                
            train_temp, val_temp = train_test_split(
                label_patients,
                test_size=val_size,
                random_state=seed
            )
            
            train_patients.extend(train_temp)
            val_patients.extend(val_temp)
    
    # Assign splits to all samples based on patient assignment
    adata.obs.loc[adata.obs[patient_obs].isin(train_patients), split_key] = 'train'
    adata.obs.loc[adata.obs[patient_obs].isin(val_patients), split_key] = 'val'
    if test_size > 0:
        adata.obs.loc[adata.obs[patient_obs].isin(test_patients), split_key] = 'test'
    
    # Generate summary statistics
    if return_summary:
        print(f"\nSplit Summary:")
        print(f"Total patients: {len(patient_stratify_df[patient_obs].unique())}")
        print(f"Train patients: {len(train_patients)}")
        print(f"Val patients: {len(val_patients)}")
        if test_size > 0:
            print(f"Test patients: {len(test_patients)}")
        
        print(f"\nSample counts:")
        split_counts = adata.obs[split_key].value_counts()
        for split, count in split_counts.items():
            print(f"{split}: {count} samples")
        
        print(f"\nStratification balance:")
        for split in ['train', 'val', 'test'] if test_size > 0 else ['train', 'val']:
            if split in adata.obs[split_key].values:
                split_data = adata[adata.obs[split_key] == split]
                print(f"\n{split.upper()} split stratification:")
                stratify_dist = split_data.obs[stratify_obs].value_counts()
                for label, count in stratify_dist.items():
                    print(f"  {label}: {count} samples")
    
    return adata 