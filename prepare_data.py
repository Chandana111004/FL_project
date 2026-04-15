"""
Data preparation script for Fertility Risk Prediction
Using NFHS-5 (India DHS) Individual Recode dataset
4 Risk Classes: No Risk / Low Risk / High Risk / Critical Risk
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import pyreadstat

# ── Feature columns from NFHS-5 ───────────────────────────────
SELECTED_COLUMNS = [
    # Demographics
    'v012',    # current age
    'v025',    # urban/rural
    'v106',    # education level
    'v190',    # wealth index
    'v502',    # currently in union/married

    # Fertility & Children
    'v201',    # total children ever born
    'v208',    # births in last 5 years
    'v213',    # currently pregnant
    'v228',    # ever had terminated pregnancy
    'v221',    # marriage to first birth interval
    'v511',    # age at first cohabitation
    'v525',    # age at first sex

    # Contraception
    'v312',    # current contraceptive method
    'v626a',   # unmet need for contraception
    'v602',    # fertility preference

    # Blood Pressure
    'sb18s',   # systolic BP
    'sb18d',   # diastolic BP

    # Blood / Anemia
    'v453',    # hemoglobin level
    'v457',    # anemia level

    # Body
    'v445',    # BMI
    'v437',    # weight
    'v438',    # height

    # Medical conditions
    's728a',   # currently has diabetes
    's728b',   # currently has hypertension

    # Lifestyle
    'v463a',   # smokes cigarettes
    'v463c',   # chews tobacco
    's720',    # drinks alcohol

    # Socioeconomic
    'v481',    # health insurance
    'v116',    # toilet facility
    'v161',    # cooking fuel
]

FEATURE_NAMES = [
    'age', 'residence_type', 'education_level', 'wealth_index', 'marital_status',
    'total_children_born', 'births_last_5yrs', 'currently_pregnant',
    'terminated_pregnancy', 'marriage_to_first_birth', 'age_first_cohabitation',
    'age_first_sex', 'contraceptive_method', 'unmet_contraception_need',
    'fertility_preference', 'systolic_bp', 'diastolic_bp',
    'hemoglobin_level', 'anemia_level', 'bmi', 'weight', 'height',
    'has_diabetes', 'has_hypertension', 'smokes', 'chews_tobacco',
    'drinks_alcohol', 'has_insurance', 'toilet_facility', 'cooking_fuel'
]


def load_nfhs5_data(filepath='data/nfhs5/raw/IAIR7EFL.DTA'):
    """Load NFHS-5 DTA file with selected columns only"""
    print("Loading NFHS-5 dataset...")
    print("This may take 5-10 minutes for the first time...")

    df, meta = pyreadstat.read_dta(filepath, usecols=SELECTED_COLUMNS)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def clean_data(df):
    """Clean and preprocess the data"""
    print("\nCleaning data...")

    # Rename columns to readable names
    # Force correct column order before renaming
    CORRECT_ORDER = [
    	'v012','v025','v106','v190','v502',
    	'v201','v208','v213','v228','v221','v511','v525',
    	'v312','v626a','v602',
    	'sb18s','sb18d','v453','v457','v445','v437','v438',
    	's728a','s728b','v463a','v463c','s720','v481','v116','v161'
    ]
    df = df[CORRECT_ORDER]
    df.columns = FEATURE_NAMES

    # ── Handle missing values ──────────────────────────────────

    # Columns with high missing (ANC related) — fill with 0
    high_missing = ['marriage_to_first_birth', 'age_first_cohabitation']
    for col in high_missing:
        df[col] = df[col].fillna(0)

    # Medical columns — fill with median
    medical_cols = ['systolic_bp', 'diastolic_bp', 'hemoglobin_level',
                    'anemia_level', 'bmi', 'weight', 'height']
    for col in medical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # All other columns — fill with mode
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── Convert all to numeric ─────────────────────────────────
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print(f"✓ Cleaned data: {len(df):,} rows")
    print(f"  Missing values remaining: {df.isnull().sum().sum()}")
    return df


def create_4class_risk_label(df):
    """
    Create 4-class fertility risk label based on medical indicators

    Class 0 → No Risk
    Class 1 → Low Risk
    Class 2 → High Risk
    Class 3 → Critical Risk
    """
    print("\nCreating 4-class risk labels...")

    risk_score = pd.Series(0, index=df.index)

    # ── Age Risk ───────────────────────────────────────────────
    risk_score += ((df['age'] < 18) | (df['age'] > 40)).astype(int) * 2
    risk_score += ((df['age'] >= 18) & (df['age'] < 20)).astype(int) * 1
    risk_score += ((df['age'] > 35) & (df['age'] <= 40)).astype(int) * 1

    # ── Blood Pressure Risk ────────────────────────────────────
    # Stored as actual mmHg
    risk_score += ((df['systolic_bp'] > 140) | (df['diastolic_bp'] > 90)).astype(int) * 3
    risk_score += (((df['systolic_bp'] >= 130) & (df['systolic_bp'] <= 140)) |
                   ((df['diastolic_bp'] >= 85) & (df['diastolic_bp'] <= 90))).astype(int) * 1

    # ── Anemia Risk ────────────────────────────────────────────
    # CORRECT: 1=severe, 2=moderate, 3=mild, 4=not anemic
    risk_score += (df['anemia_level'] == 1).astype(int) * 3  # severe
    risk_score += (df['anemia_level'] == 2).astype(int) * 2  # moderate
    risk_score += (df['anemia_level'] == 3).astype(int) * 1  # mild

    # ── Hemoglobin Risk ───────────────────────────────────────
    
    # Stored as actual x10 (112 = 11.2 g/dL)
    # Only score if value > 0 (0 means not measured)
    risk_score += ((df['hemoglobin_level'] > 0) & 
                   (df['hemoglobin_level'] < 70)).astype(int) * 3
    risk_score += ((df['hemoglobin_level'] >= 70) & 
                   (df['hemoglobin_level'] < 90)).astype(int) * 2
    risk_score += ((df['hemoglobin_level'] >= 90) & 
                   (df['hemoglobin_level'] < 100)).astype(int) * 1

    # ── BMI Risk ──────────────────────────────────────────────
    # Stored as 100x actual (2224 = 22.24)
    risk_score += (df['bmi'] < 1850).astype(int) * 2
    risk_score += (df['bmi'] > 3000).astype(int) * 2
    risk_score += ((df['bmi'] >= 2500) & (df['bmi'] <= 3000)).astype(int) * 1

    # ── Medical Conditions ────────────────────────────────────
    # 0=no, 1=yes, 8=don't know (treat 8 as no)
    risk_score += (df['has_diabetes'] == 1).astype(int) * 3
    risk_score += (df['has_hypertension'] == 1).astype(int) * 3

    # ── High Parity ───────────────────────────────────────────
    risk_score += (df['total_children_born'] > 4).astype(int) * 2
    risk_score += (df['total_children_born'] > 6).astype(int) * 1

    # ── Early Pregnancy ───────────────────────────────────────
    # Only count if value > 0 (0 means never/not applicable)
    risk_score += ((df['age_first_cohabitation'] > 0) & 
                   (df['age_first_cohabitation'] < 18)).astype(int) * 2
    risk_score += ((df['age_first_sex'] > 0) & 
                   (df['age_first_sex'] < 16)).astype(int) * 2

    # ── Terminated Pregnancy ──────────────────────────────────
    # 0=no, 1=yes
    risk_score += (df['terminated_pregnancy'] == 1).astype(int) * 1

    # ── Unmet Contraception Need ──────────────────────────────
    # 1=unmet need for spacing, 2=unmet need for limiting
    risk_score += ((df['unmet_contraception_need'] == 1) |
                   (df['unmet_contraception_need'] == 2)).astype(int) * 1

    # ── Lifestyle Risks ───────────────────────────────────────
    # 0=no, 1=yes
    risk_score += (df['smokes'] == 1).astype(int) * 1
    risk_score += (df['chews_tobacco'] == 1).astype(int) * 1
    risk_score += (df['drinks_alcohol'] == 1).astype(int) * 1
    
    # ── Socioeconomic Risks ───────────────────────────────────
    # education: 0=no education (higher risk)
    # wealth: 1=poorest, 2=poorer (higher risk)
    risk_score += (df['education_level'] == 0).astype(int) * 1
    risk_score += (df['wealth_index'] == 1).astype(int) * 1  # only poorest
    # Remove insurance — too many missing as 0
    # ── Assign 4 classes ──────────────────────────────────────

    labels = pd.cut(
        risk_score,
        bins=[-1, 2, 5, 9, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # Print distribution
    counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)
    print("\nRisk label distribution:")
    for cls, name in zip([0,1,2,3],
                         ['No Risk', 'Low Risk', 'High Risk', 'Critical']):
        count = counts.get(cls, 0)
        print(f"  Class {cls} ({name}): {count:,} ({count/total*100:.1f}%)")

    return labels.values


def create_federated_partitions(X, y, num_clients=5):
    """Partition data evenly across clients"""
    print(f"\nCreating {num_clients} federated partitions...")
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    partitions = np.array_split(indices, num_clients)
    return partitions


def save_federated_data(X, y, partitions, output_dir='data/processed_dp'):
    """Save partitioned data for federated learning"""
    os.makedirs(output_dir, exist_ok=True)

    # Global test set (20%)
    print("\nCreating train/test split...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    print(f"  ✓ Global test set: {len(X_test):,} samples")

    # Recalculate partitions on training data
    new_partitions = create_federated_partitions(
        X_temp, y_temp, num_clients=len(partitions)
    )

    # Save client partitions
    print("\nSaving client partitions...")
    for i, partition_indices in enumerate(new_partitions):
        X_client = X_temp[partition_indices]
        y_client = y_temp[partition_indices]

        X_train, X_val, y_train, y_val = train_test_split(
            X_client, y_client,
            test_size=0.2, random_state=42, stratify=y_client
        )

        client_dir = os.path.join(output_dir, f'client_{i}')
        os.makedirs(client_dir, exist_ok=True)

        np.save(os.path.join(client_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(client_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(client_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(client_dir, 'y_val.npy'), y_val)

        dist = np.bincount(y_train.astype(int), minlength=4)
        print(f"  Client {i}: {len(X_train):,} train | "
              f"No:{dist[0]} Low:{dist[1]} High:{dist[2]} Critical:{dist[3]}")

    # Save metadata
    metadata = {
        'feature_names': FEATURE_NAMES,
        'num_features': len(FEATURE_NAMES),
        'num_classes': 4,
        'num_clients': len(partitions),
        'class_names': ['No Risk', 'Low Risk', 'High Risk', 'Critical Risk']
    }

    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n✓ All data saved to {output_dir}")
    return metadata


def main():
    print("="*60)
    print("NFHS-5 Fertility Risk — Data Preparation (4 Classes)")
    print("="*60)

    # ── Step 1: Load ───────────────────────────────────────────
    dta_path = 'data/nfhs5/raw/IAIR7EFL.DTA'
    if not os.path.exists(dta_path):
        print(f"ERROR: File not found at {dta_path}")
        return

    df = load_nfhs5_data(dta_path)

    # ── Step 2: Clean ──────────────────────────────────────────
    df = clean_data(df)

    # ── Step 3: Create labels ──────────────────────────────────
    y = create_4class_risk_label(df)

    # ── Step 4: Scale features ─────────────────────────────────
    print("\nScaling features...")
    X = df.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save scaler for app to use
    os.makedirs('data/processed_dp', exist_ok=True)
    with open('data/processed_dp/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved")

    print(f"\nFinal dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Classes: {np.bincount(y.astype(int))}")

    # ── Step 5: Partition ──────────────────────────────────────
    partitions = create_federated_partitions(X, y, num_clients=5)

    # ── Step 6: Save ───────────────────────────────────────────
    metadata = save_federated_data(X, y, partitions)

    print("\n" + "="*60)
    print("✓ Data preparation complete!")
    print("="*60)
    print(f"Features: {metadata['num_features']}")
    print(f"Classes: {metadata['num_classes']} → {metadata['class_names']}")
    print(f"Clients: {metadata['num_clients']}")
    print("\nNext: run  flwr run")
    print("="*60)


if __name__ == "__main__":
    main()
