##### LIBRARIES #####

import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xanfis.models.bio_anfis import BioAnfisRegressor
import matplotlib.pyplot as plt

##### SETTINGS #####

DATA_PATH = "/Users/davidpaquette/Documents/Thesis/Project/Data/ANFIS"
CITIES = ["Montreal", "Rome"]
EPOCHS = 250
POP_SIZE = 30

##### TRAINING LOOP #####

for city in CITIES:
    print(f"\nTraining ANFIS pairwise model for {city}...")

    file_path = os.path.join(DATA_PATH, f"{city}_pairwise_training_data.csv")
    df = pd.read_csv(file_path)

    X = df.drop(columns=["target_time_min"]).values
    y = df["target_time_min"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Normalize target
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Dimensionality reduction (retain 95% variance)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"PCA reduced to {pca.n_components_} components.")

    # Create and train model
    model = BioAnfisRegressor(
        num_rules=15,
        mf_class="Sigmoid",
        optim="BaseGA",
        optim_params={"epoch": EPOCHS, "pop_size": POP_SIZE},
        obj_name="RMSE",
        seed=42,
        verbose=True
    )

    model.fit(X_train_pca, y_train_scaled, lb=-1, ub=1)

    if hasattr(model, 'loss_train'):
        plt.figure(figsize=(8, 5))
        plt.plot(model.loss_train, label='Training RMSE', color='darkorange')
        plt.title(f'Pairwise ANFIS Training Curve for {city}')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"{DATA_PATH}/{city}_pairwise_anfis_training_curve.png", dpi=200)
        plt.show()
    else:
        print("Training loss history not found in model object.")

    # Predictions
    y_pred_scaled = model.predict(X_test_pca)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Evaluation
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{city} Evaluation Metrics:")
    print(f"RMSE: {rmse:.3f} min")
    print(f"MAE: {mae:.3f} min")
    print(f"R² Score: {r2:.4f}")

    # Save outputs
    model.save_model(save_path=DATA_PATH, filename=f"{city}_pairwise_model.pkl")
    joblib.dump(scaler_X, os.path.join(DATA_PATH, f"{city}_pairwise_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(DATA_PATH, f"{city}_pairwise_scaler_y.pkl"))
    joblib.dump(pca, os.path.join(DATA_PATH, f"{city}_pairwise_pca.pkl"))

    print(f"Model and scalers saved for {city}.\n")
