import numpy as np


def sample_images(num_images, num_pixels):
    return np.clip(
        (np.ones((num_images, num_pixels)) * 100 + np.random.randn(num_images, num_pixels) * 50).astype(np.int32),
        0, 255)

N = 10
D = 16
H = 4
C = 3

X = sample_images(N, D)

W1 = 1e-4 * np.random.randn(D, H)
b1 = np.ones(H) * 0.01

W2 = 1e-4 * np.random.randn(H, C)
b2 = np.ones(C) * 0.01


print(X)
print(W1)

out1 = X.dot(W1) + b1.T
print("out1")
print(out1)

print("out1_relu")
out1_relu = np.maximum(out1, np.zeros_like(out1))
print(out1_relu)

scores = out1_relu.dot(W2) + b2.T

print("scores")
print(scores)

y = np.random.random_integers(0, C-1, size=N)

print(y)

print(scores[range(N), y])

y_m = np.zeros((N, C))
y_m[range(N), y] = 1

print(y_m)

print("----")

print(scores[range(N), y])

print("----")

print(scores * y_m)


