import numpy as np
import matplotlib.pyplot as plt
import math

B = 4.
C = 1.
D = 3.5

alpha = np.linspace(-0.4,0.4,1000)
F = D * np.sin(C * np.arctan(B * alpha))
plt.plot(alpha,F) 
plt.show()
