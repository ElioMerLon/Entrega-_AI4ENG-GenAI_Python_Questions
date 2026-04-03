import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def generar_caso_de_uso_evaluar_umbral_optimo():
    n = random.randint(40, 100)
    m = random.randint(3, 6)

    X = np.random.randn(n, m)
    cols = [f"x{i}" for i in range(m)]

    logits = X[:, 0] + np.random.randn(n)
    y = (logits > 0).astype(int)

    df = pd.DataFrame(X, columns=cols)
    target_col = "target"
    df[target_col] = y

    input_data = {
        "df": df.copy(),
        "target_col": target_col
    }

    X_df = df.drop(columns=[target_col])
    y_series = df[target_col]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_df, y_series)

    probs = model.predict_proba(X_df)[:, 1]

    best_f1 = -1
    best_t = 0

    for t in np.arange(0, 1.01, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_series, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    output_data = (float(best_t), float(best_f1))

    return input_data, output_data