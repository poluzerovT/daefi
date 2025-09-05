import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = np.array(range(10))
y = x**2

df = pd.DataFrame({
    'x': x,
    'y': y
})

plt.plot(df['x'], df['y'])
plt.show()

print('OK')
