import cv2
import numpy as np


def perform_flash(source, a=5, target=-1, perform_gamma_correction=True):
    rows, cols, _ = source.shape

    v = np.max(source, axis=2)
    vd = np.copy(v)
    vd[vd == 0] = 1e-9
    result = source / (a * np.exp(np.mean(np.log(vd))) + np.tile(np.expand_dims(vd, axis=2), (1, 1, 3)))

    if perform_gamma_correction:
        result **= 1.0 / 2.2

    if target >= 0:
        result *= target / np.mean((0.299 * result[:, :, 2] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 0]))
    else:
        result *= 255.0 / np.max(result)

    return result


def perform_storm(source, a=5, target=-1, kernels=(1, 4, 16, 64, 256), perform_gamma_correction=True):
    rows, cols, _ = source.shape

    v = np.max(source, axis=2)
    vd = np.copy(v)
    vd[vd == 0] = 1e-9
    lv = np.log(vd)
    result = sum([source / np.tile(
        np.expand_dims(a * np.exp(cv2.boxFilter(lv, -1, (int(min(rows // kernel, cols // kernel)),) * 2)) + vd, axis=2),
        (1, 1, 3)) for kernel in kernels])

    if perform_gamma_correction:
        result **= 1.0 / 2.2

    if target >= 0:
        result *= target / np.mean((0.299 * result[:, :, 2] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 0]))
    else:
        result *= 255.0 / np.max(result)

    return result
