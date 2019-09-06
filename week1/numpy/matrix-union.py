import numpy as np

aMatrix = np.eye(3)
bMatrix = np.eye(3)

unionMatrix = np.vstack((aMatrix, bMatrix))

print((aMatrix, bMatrix))
