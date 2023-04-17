import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_regression

def load_dataset(data_path):

    # Read Data 
    
    df_clinical = pd.read_csv(data_path["clinical"])
    df_proteins = pd.read_csv(data_path["proteins"])
    df_peptides = pd.read_csv(data_path["peptides"])

    # Target Preparation 

    patients = {}
    df_clinical[f'updrs_3_next_year'] = 0

    for patient_id in df_clinical.patient_id.unique():
        patient = df_clinical[df_clinical.patient_id == patient_id]
        for month in patient.visit_month.values:
            future_score = patient[patient.visit_month == month + 12][f'updrs_3'].to_list() 
            if (future_score == []): future_score = np.NaN
            patient.loc[patient.visit_month == (month), ['updrs_3_next_year']] = future_score
        patients[patient_id] = patient

    clinical_features = pd.concat(patients.values(), ignore_index=True).set_index('visit_id').iloc[:,7:]
    clinical_features.dropna(inplace=True)

    # Feature Preparation
    
    protein_features = df_proteins.pivot(index='visit_id', columns='UniProt', values='NPX')
    peptide_features = df_peptides.pivot(index='visit_id', columns='Peptide', values='PeptideAbundance')

    # Merge Features and Target 

    df = clinical_features \
        .merge(protein_features, left_index=True, right_index=True, how='left') \
        .merge(peptide_features, left_index=True, right_index=True, how='left')    

    df['visit_month'] = df.reset_index().visit_id.str.split('_').apply(lambda x: int(x[1])).values
    protein_list = protein_features.columns.to_list()
    peptide_list = peptide_features.columns.to_list()
    
    # Transform Features and Target 

    x = df[protein_list + ["visit_month"]]
    y = df[clinical_features.columns]

    x.visit_month = x.visit_month.astype('float')
    y = y.astype('float')

    feature_transformer = ColumnTransformer([
        (
            'numerical',
            make_pipeline(IterativeImputer(), StandardScaler(), SelectKBest(score_func=f_regression, k=50)),
            make_column_selector(dtype_include='number')
        ),
    ])

    x = feature_transformer.fit_transform(x, y.values.ravel())
    y = y.to_numpy()

    # Return Features and Target 

    return x, y
