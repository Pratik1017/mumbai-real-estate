import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import os


def drop_col(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df.drop(columns=['price_sqft'] , inplace=True)
        return df
    except Exception as e:
          
          print(f"Error loading data: {e}")
          return None


def create_trf():

    try:
        ohe_cols = ['status', 'neworold', 'type_of_building', 'category']
        trf1 = ColumnTransformer(
             transformers=[
                  ("ohe" , OneHotEncoder(
                       drop="first",
                       handle_unknown='ignore',
                       sparse_output=False
                  ),
                  ohe_cols
                  )
             ],
             remainder="passthrough",
             verbose_feature_names_out=False
        )
        return trf1

    except Exception as e:
          
          print(f"Error loading data: {e}")
          return None
     


def export_X_train(X_train: pd.DataFrame) -> pd.DataFrame | None:
    try:

        path_link = os.path.join("data","interim")
        os.makedirs(path_link , exist_ok=True)

        X_train.to_csv(
            os.path.join(path_link, "X_train_cleaned_v2.csv"),
            index=False
        )
        return X_train

    except Exception as e:
      
      print(f"Error loading data: {e}")
      return None



def export_X_test(X_test: pd.DataFrame) -> pd.DataFrame | None:
    try:
        
        path_link = os.path.join("data","interim")
        os.makedirs(path_link , exist_ok=True)

        X_test.to_csv(
                    os.path.join(path_link, "X_test_cleaned_v2.csv"),
                    index=False
                )
        return X_test

    except Exception as e:
      
      print(f"Error loading data: {e}")
      return None




if __name__ == "__main__":

    # Data Loading
    X_train = pd.read_csv("data/interim/X_train.csv")
    X_test = pd.read_csv("data/interim/X_test.csv")

    if X_train is not None and X_test is not None:
        X_train = drop_col(X_train)
        X_test = drop_col(X_test)

        if X_train is not None and X_test is not None:
            X_train = export_X_train(X_train)
            X_test = export_X_test(X_test)