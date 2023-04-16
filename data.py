import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_dataset(data_path):

    # Read Data 
    
    df_clinical = pd.read_csv(data_path["clinical"])
    df_proteins = pd.read_csv(data_path["proteins"])
    df_peptides = pd.read_csv(data_path["peptides"])

    # Target Preparation 

    patients = {}
    month_windows = [0, 6, 12, 24]

    for e in range(1,5):
        for m in month_windows:
            df_clinical[f'updrs_{e}_plus_{m}_months'] = 0

    for patient in df_clinical.patient_id.unique():
        temp = df_clinical[df_clinical.patient_id == patient]
        month_list = []

        for month in temp.visit_month.values:
            month_list.append([month, month + 6, month + 12, month + 24])

        for month in range(len(month_list)):
            for x in range(1,5):
                arr = temp[temp.visit_month.isin(month_list[month])][f'updrs_{x}'].fillna(0).to_list()
                if len(arr) == 4:
                    for e, i in enumerate(arr):
                        m = month_list[month][0]
                        temp.loc[temp.visit_month == m, [f'updrs_{x}_plus_{month_windows[e]}_months']] = i
                else:
                    temp = temp[~temp.visit_month.isin(month_list[month])]

        patients[patient] = temp

    clinical_features = pd.concat(patients.values(), ignore_index=True).set_index('visit_id').iloc[:,7:]

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

    x = df[protein_list + peptide_list + ["visit_month"]]
    y = df[clinical_features.columns]

    x.visit_month = x.visit_month.astype('float')
    y = y.astype('float')

    feature_transformer = ColumnTransformer([
        (
            'numerical',
            make_pipeline(IterativeImputer(), StandardScaler()),
            make_column_selector(dtype_include='number')
        ),
    ])

    x = feature_transformer.fit_transform(x)
    y = y.to_numpy()

    # Return Features and Target 

    return x, y
