import numpy as np


def rotation_matrix(
    omega: float, phi: float, kappa: float, full_circle: float = 2 * np.pi
) -> np.ndarray:
    """
    Calculates rotation matrix from euler rotation angles.

    Parameters
    ----------
    omega : float
        Rotation around x-axis.
    phi : float
        Rotation around y-axis.
    kappa : float
        Rotation around z-axis.
    full_circle : float
        Full circle of input angles.

    Returns
    -------
    _ : np.ndarray
        Rotation matrix.
    """
    omega = omega / full_circle * 2 * np.pi
    phi = phi / full_circle * 2 * np.pi
    kappa = kappa / full_circle * 2 * np.pi

    r11 = np.cos(phi) * np.cos(kappa)
    r12 = -np.cos(phi) * np.sin(kappa)
    r13 = np.sin(phi)
    r21 = (np.cos(omega) * np.sin(kappa)) + (np.sin(omega) * np.sin(phi) * np.cos(kappa))
    r22 = (np.cos(omega) * np.cos(kappa)) - (np.sin(omega) * np.sin(phi) * np.sin(kappa))
    r23 = -np.sin(omega) * np.cos(phi)
    r31 = (np.sin(omega) * np.sin(kappa)) - (np.cos(omega) * np.sin(phi) * np.cos(kappa))
    r32 = (np.sin(omega) * np.cos(kappa)) + (np.cos(omega) * np.sin(phi) * np.sin(kappa))
    r33 = np.cos(omega) * np.cos(phi)

    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])


def rotation_matrix_to_euler(rot_mat: np.ndarray, full_circle: float = 2 * np.pi) -> tuple:
    """
    Convert rotation matrix to euler angles (omega, phi, kappa)

    Parameters
    ----------
    rot_mat : np.ndarray
        Rotation matrix
    full_circle : float; default = 2 * np.pi
        Full circle of input angles.

    Returns
    -------
    _  : tuple(omega, phi, kappa)
        Euler angle describing same rotation as input matrix
    """
    rho = full_circle / (2 * np.pi)
    omega = np.arctan2(-rot_mat[1, 2], rot_mat[2, 2]) * rho
    phi = np.arcsin(rot_mat[0, 2]) * rho
    kappa = np.arctan2(-rot_mat[0, 1], rot_mat[0, 0]) * rho

    return (omega, phi, kappa)
