
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
import os
import yaml

try:
    with open("params.yaml","r") as f:
        params = yaml.safe_load(f)

        test_size = params["data_ingestion"]["test_size"]
        random_state = params["data_ingestion"]["random_state"]

except Exception as e:
    print(f"The Error is {e}")

def load_data(url: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Removing Unwanted Columns
def drop_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df.drop(
            columns=[
                "latitude", "longitude", "parking",
                "Lift", "Landmarks", "desc"
            ],
            inplace=True,
            errors="ignore"
        )
        return df
    except Exception as e:
        print(f"Error dropping columns: {e}")
        return None


# Renaming Columns
def rename_cols(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df.rename(columns={
            "Address": "address",
            "Bedrooms": "bedrooms",
            "Bathrooms": "bathrooms",
            "Balcony": "balcony",
            "Status": "status",
            "Furnished_status": "furnished_status",
            "Price_sqft": "price_sqft",
            "Category": "category"
        }, inplace=True)

        return df

    except Exception as e:
        print(f"Error renaming columns: {e}")
        return None


# Extract Locality From Address
def add_split(address: str | float) -> str | float:
    if pd.isna(address):
        return np.nan

    address = str(address).lower().strip()
    address = address.split("-")[1] if "-" in address else address
    address = address.split(",")[0]

    return address.strip()


# Removing Null Addresses
def add_null(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["address"].notna()].copy()
    return df


WEST_SPECIFIC = [
    "andheri west", "bandra west", "khar west", "santacruz west", 
    "vile parle west", "jogeshwari west", "goregaon west", "malad west"
]

EAST_SPECIFIC = [
    "andheri east", "goregaon east", "malad east", "bandra east", 
    "vile parle east", "central navi mumbai"
]

# General
NORTH_LOCALITIES = [
    "borivali", "dahisar", "kandivali", "malad", "mira road", 
    "bhayandar", "vasai", "virar", "uttan", "thakur village", 
    "akurli nagar", "kurar village", "dindoshi", "kasturpark", 
    "gorai", "kashimira", "kalyan", "kalyan road", "charkop"
]

WEST_LOCALITIES = [
    "juhu", "versova", "lokhandwala", "oshiwara", "pali hill"
]

EAST_LOCALITIES = [
    "chembur", "ghatkopar", "vikhroli", "powai", "chandivali", 
    "kalina", "sakinaka", "kurla", "kanjurmarg", "bhandup", 
    "mulund", "govandi", "deonar", "nahur", "trombay", 
    "chunabhatti", "tilak nagar", "central line", "harbour line", 
    "around mumbai", "khargar", "beyond thane", "majiwada"
]

SOUTH_LOCALITIES = [
    "colaba", "fort", "churchgate", "cuffe parade", "marine lines", 
    "nariman point", "csmt", "cst", "kalbadevi", "girgaon", 
    "malabar hill", "walkeshwar", "breach candy", "tardeo", 
    "grant road", "mumbai central", "byculla", "mazgaon", 
    "lower parel", "parel", "worli", "prabhadevi", 
    "n m joshi marg", "central mumbai"
]

# Validated PIN Code Matching
NORTH_PINS = {
    "400066", "400091", "400092", "400095", "400097", "400101", 
    "400103", "400104", "401101", "401105", "401107", "421003",
    "400068", "401201", "401202"
}

WEST_PINS = {
    "400049", "400050", "400051", "400052", "400053", "400054", 
    "400056", "400061", "400064", "400055", "400058", "400062"
}

EAST_PINS = {
    "400037", "400057", "400063", "400069", "400070", "400071", 
    "400072", "400076", "400077", "400080", "400086", "400093",
    "400098", "400601", "421202" 
}

SOUTH_PINS = {
    "400001", "400002", "400003", "400005", "400006", "400007", 
    "400008", "400009", "400010", "400011", "400012", "400013", 
    "400018", "400020", "400021", "400026"
}


def categorize_mumbai(address: str | float) -> str | float:
    try:
        if pd.isna(address):
            return np.nan

        address = str(address).lower().strip()
        address = re.sub(r"\s+", " ", address)

        pin_match = re.search(r"(?<!\d)\d{6}(?!\d)", address)
        pin = pin_match.group() if pin_match else None

        # Step 1: Specific localities
        if any(x in address for x in WEST_SPECIFIC):
            return "West Mumbai"

        if any(x in address for x in EAST_SPECIFIC):
            return "East Mumbai"

        # Step 2: General localities
        if any(x in address for x in NORTH_LOCALITIES):
            return "North Mumbai"

        if any(x in address for x in WEST_LOCALITIES):
            return "West Mumbai"

        if any(x in address for x in EAST_LOCALITIES):
            return "East Mumbai"

        if any(x in address for x in SOUTH_LOCALITIES):
            return "South Mumbai"

        # Step 3: PIN codes
        if pin in NORTH_PINS:
            return "North Mumbai"

        if pin in WEST_PINS:
            return "West Mumbai"

        if pin in EAST_PINS:
            return "East Mumbai"

        if pin in SOUTH_PINS:
            return "South Mumbai"

        # Step 4: Already categorized addresses
        if address in ["north", "north mumbai"]:
            return "North Mumbai"

        if address in ["west", "west mumbai", "western suburbs"]:
            return "West Mumbai"

        if address in ["east", "east mumbai"]:
            return "East Mumbai"

        if address in ["south", "south mumbai"]:
            return "South Mumbai"

        return np.nan

    except Exception as e:
        print(f"Error categorizing address: {e}")
        return np.nan


# Removing Unclassified Addresses
def drop_null(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df = df[
            df["category"].isin([
                "North Mumbai",
                "South Mumbai",
                "West Mumbai",
                "East Mumbai"
            ])
        ].copy()

        return df

    except Exception as e:
        print(f"Error filtering categories: {e}")
        return None


def split( df: pd.DataFrame ) -> tuple[ pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None:
    try:
        X = df.drop(columns=["price"])
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        path_link = os.path.join("data", "raw")
        os.makedirs(path_link, exist_ok=True)

        X_train.to_csv(os.path.join(path_link, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(path_link, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(path_link, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(path_link, "y_test.csv"), index=False)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Error splitting data: {e}")
        return None


if __name__ == "__main__":

    df = load_data("data/base-file/mumbai.csv")

    if df is not None:
        df = drop_columns(df)

        if df is not None:
            df = rename_cols(df)

            if df is not None:
                df = add_null(df)

                df["address"] = df["address"].apply(add_split)

                df["category"] = df["address"].apply(
                    categorize_mumbai
                )

                df = drop_null(df)

                if df is not None:
                    split(df)
