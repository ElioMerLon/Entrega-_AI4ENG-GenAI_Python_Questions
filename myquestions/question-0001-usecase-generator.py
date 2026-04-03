import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_resumen_clientes():
    clientes = ["Ana", "Luis", "Marta", "Pedro", "Sofia", "Carlos"]
    n = random.randint(8, 20)

    df = pd.DataFrame({
        "cliente": [random.choice(clientes) for _ in range(n)],
        "monto": np.round(np.random.uniform(5, 500, size=n), 2),
        "cantidad": np.random.randint(1, 10, size=n)
    })

    input_data = {"df": df.copy()}

    output_data = (
        df.groupby("cliente", as_index=False)
          .agg(
              total_gastado=("monto", "sum"),
              total_unidades=("cantidad", "sum"),
              ticket_promedio=("monto", "mean")
          )
          .sort_values("total_gastado", ascending=False)
          .reset_index(drop=True)
    )

    return input_data, output_data