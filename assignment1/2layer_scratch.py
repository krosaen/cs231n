import numpy as np


def sample_images(num_pixels, num_images):
    return np.clip(
        (np.ones((num_images, num_pixels)) * 100 + np.random.randn(num_images, num_pixels) * 50).astype(np.int32),
        0, 255)

N = 10
D = 16
H = 4
C = 3

X = sample_images(D, N)

W1 = 1e-4 * np.random.randn(D, H)
b1 = np.ones(H) * 0.01

print(X)
print(W1)

out1 = X.dot(W1) + b1.T
print("out1")
print(out1)

print("out1_relu")
out1_relu = np.maximum(out1, np.zeros_like(out1))
print(out1_relu)

print("")
print(np.max(out1_relu, axis=1).reshape(out1_relu.shape[0], -1))