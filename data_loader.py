import pandas as pd
import numpy as np


def load_data(
    species_file="species.xlsx",
    selected_species_file="c1.txt",
    group_a_file="c_sample_tax.csv",
    group_b_file="o_sample_tax.csv"
):
    species_df = pd.read_excel(species_file, index_col=0)
    selected_species_df = pd.read_csv(selected_species_file, sep="\t")
    selected_species = selected_species_df["species"].tolist()

    group_a_df = pd.read_csv(group_a_file, header=None)
    group_b_df = pd.read_csv(group_b_file, header=None)

    group_a_samples = group_a_df.iloc[1].dropna().tolist()
    group_b_samples = group_b_df.iloc[1].dropna().tolist()

    filtered_species = species_df.loc[
        species_df.index.isin(selected_species)
    ]
    X_df = filtered_species.T

    all_samples = group_a_samples + group_b_samples
    labels = [0] * len(group_a_samples) + [1] * len(group_b_samples)

    existing_samples = [s for s in all_samples if s in X_df.index]
    existing_labels = [labels[all_samples.index(s)] for s in existing_samples]

    X = X_df.loc[existing_samples].values
    y = np.array(existing_labels)
    feature_names = X_df.columns.tolist()

    return X, y, feature_names
