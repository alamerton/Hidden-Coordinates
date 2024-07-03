# %%
import plotly_express as px
import pandas as pd
import numpy as np

# %%
df = pd.DataFrame({
    "index": np.random.randint(1, 100, size=10),
    "abs_pos": np.random.rand(10),
    "pred": np.random.choice(['X', 'Y', 'Z'], size=10),
})

print(df)


# %%
px.histogram(df, x="abs_pos", color="pred", facet_row="index", barnorm="fraction").show()

# %%
