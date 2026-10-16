import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import os

# Step 1
def remove_cols(df: pd.DataFrame) -> pd.DataFrame | None:
  try:
    df.drop(columns = ['address' , "balcony"], inplace=True)
    return df
  except Exception as e:
        
        print(f"Error loading data: {e}")
        return None


# Step 2
def change_dtype(df: pd.DataFrame) -> pd.DataFrame | None:
  try:

    df['bedrooms'] = df['bedrooms'].astype(np.int32)
    df['bathrooms'] = df['bathrooms'].astype(np.int32)
    return df

  except Exception as e:
        
        print(f"Error loading data: {e}")
        return None


# Add in main file : df.drop(index=[2856],inplace=True)


# Step 4
def under_cons(df: pd.DataFrame) -> pd.DataFrame | None:
  try:

    nan_status = df[df['status'].isnull()]
    nan_status['status'] = "Under Construction"
    df.update(nan_status)
    return df

  except Exception as e:
        
        print(f"Error loading data: {e}")
        return None

  # Step 5 - Missing Value Imputation of column = furnished_status


def missing_impute(df: pd.DataFrame) -> pd.DataFrame | None:

    try:

        # Features used for prediction
        features = [
            "price",
            "area",
            "bedrooms",
            "bathrooms"
        ]
    
        # Separate known and missing values
        train_df = df[df["furnished_status"].notna()].copy()
        missing_df = df[df["furnished_status"].isna()].copy()
    
        X_train = train_df[features]
        y_train = train_df["furnished_status"].astype(int)
    
        X_missing = missing_df[features]
    
        # Model pipeline
        knn_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ])
    
        # Train model
        knn_model.fit(X_train, y_train)
    
        # Predict missing categories
        predictions = knn_model.predict(X_missing)
    
        # Fill missing values
        df["furnished_status"] = df["furnished_status"].astype(float)
    
        df.loc[missing_df.index, "furnished_status"] = predictions
    
        return df

    except Exception as e:
        
        print(f"Error loading data: {e}")
        return None


def export_X_train(X_train: pd.DataFrame) -> pd.DataFrame | None:
    try:

        path_link = os.path.join("data","interim")
        os.makedirs(path_link , exist_ok=True)

        X_train.to_csv(
            os.path.join(path_link, "X_train_cleaned.csv"),
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
                    os.path.join(path_link, "X_test_cleaned.csv"),
                    index=False
                )
        return X_test

    except Exception as e:
      
      print(f"Error loading data: {e}")
      return None





if __name__ == "__main__":

    # Data Loading
    X_train = pd.read_csv("data/raw/X_train.csv")
    X_test = pd.read_csv("data/raw/X_test.csv")

    X_train = remove_cols(X_train)
    X_test = remove_cols(X_test)

    if X_train is not None and X_test is not None:
       X_train = change_dtype(X_train)
       X_test = change_dtype(X_test)

       if X_train is not None and X_test is not None:
        
        X_train = under_cons(X_train)
        X_test = under_cons(X_test)

        if X_train is not None and X_test is not None:

           X_train = missing_impute(X_train)
           X_test = missing_impute(X_test)

           if X_train is not None and X_test is not None:

              X_train = export_X_train(X_train)
              X_test = export_X_test(X_test)


       

