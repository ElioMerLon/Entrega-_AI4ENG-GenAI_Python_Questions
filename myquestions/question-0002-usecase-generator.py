import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_agrupar_ventas_mensuales():
    n = random.randint(12, 30)

    fechas = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    fechas_sample = np.random.choice(fechas, size=n)

    df = pd.DataFrame({
        "fecha": [pd.Timestamp(f).strftime("%Y-%m-%d") for f in fechas_sample],
        "ventas": np.round(np.random.uniform(100, 2000, size=n), 2)
    })

    input_data = {"df": df.copy()}

    df_aux = df.copy()
    df_aux["fecha"] = pd.to_datetime(df_aux["fecha"])
    df_aux["mes"] = df_aux["fecha"].dt.to_period("M").astype(str)

    output_data = (
        df_aux.groupby("mes", as_index=False)
              .agg(ventas_totales=("ventas", "sum"))
              .sort_values("mes")
              .reset_index(drop=True)
    )

    return input_data, output_data