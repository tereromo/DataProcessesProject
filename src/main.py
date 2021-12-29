from modeling import modeling, Model

import pandas as pd

from preprocessing import preprocess_data

from exploration import explo_plot

# Data exploration

# Data preprocessing

# Data modeling

# Model evaluation

def main():
    df = pd.read_csv('C:/Users/gorka/OneDrive/Escritorio/covid19_data.csv', index_col = 'ID')
    
    explo_plot(df)

    df_preprocessed = preprocess_data(df, 0, 0)
    print("Preprocessed")
    print(df_preprocessed)

    modeling(df_preprocessed, Model.logistic_reg_model())

    return 0

main()
