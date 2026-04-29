import numpy as np
import matplotlib.pyplot as plt


def display_values(
    iteration: int,
    x: float,
    y: float,
    z: float,
    omega: float,
    phi: float,
    kappa: float,
    digits: int = 3,
) -> None:
    """
    Displays intermediate results for a specific iteration

    Parameters
    ----------
    iteration : int
        Number (index) of iteration.
    x : float
        x coordinate of object point i.
    y : float
        y coordinate of object point i.
    z : float
        z coordinate of object point i.
    omega : float
        Rotation around x-axis.
    phi : float
        Rotation around y-axis.
    kappa : float
        Rotation around z-axis.
    digits : int
        Num digits.
    """
    omega = omega / np.pi * 180
    phi = phi / np.pi * 180
    kappa = kappa / np.pi * 180

    print(f"{iteration + 1}. Iteration")
    print(
        f"X0 = {round(x,digits)}cm  Y0 = {round(y,digits)}cm  Z0 = {round(z,digits)}cm  Omega = {round(omega,digits)}°  Phi = {round(phi,digits)}°  Kappa = {round(kappa,digits)}°"
    )


def add_text(u, v, text, offset, color="r", fontsize=12):
    """
    Wrapper around matplotlib.pyplot.text() to add text at a
    specific position for an array-like structure.

    Parameters
    ----------
    u : array-like
        Image coordinates x-axis.
    v : array-like
        Image coordinates y-axis.
    text : array-like (float or int)
        Text to be added.
    offset : int
        Offset to be added to text position.
    color : str; default = 'r'
        Text color.
    fontsize : int; default = 12.
        Fontsize.
    """
    assert len(u) == len(v) == len(text)
    assert isinstance(text[0], int) or isinstance(text[0], float)

    for i, t in enumerate(text):
        plt.text(u[i] + offset, v[i] - offset, str(int(t)), color=color, fontsize=fontsize)


def rotation_matrix_from_DLT(L: float, x: np.ndarray, x0: float, y0: float, cx: float, cy: float) -> tuple:
    """
    Builds a rotation matrix from DLT coefficients and interior orientation.

    Parameters
    ----------
    L : float
        DLT scale factor. Because the DLT solution has an arbitrary sign, this
        value is used with the sign supplied by the caller.
    x : np.ndarray
        Vector with the 11 DLT coefficients.
    x0, y0 : float
        Principal point coordinates.
    cx, cy : float
        Camera constants/principal distances in x and y direction.

    Returns
    -------
    tuple
        Rotation matrix and its nine coefficients.
    """
    x = np.asarray(x).ravel()

    r11 = L * (x0 * x[8] - x[0]) / cx
    r12 = L * (y0 * x[8] - x[4]) / cy
    r13 = L * x[8]
    r21 = L * (x0 * x[9] - x[1]) / cx
    r22 = L * (y0 * x[9] - x[5]) / cy
    r23 = L * x[9]
    r31 = L * (x0 * x[10] - x[2]) / cx
    r32 = L * (y0 * x[10] - x[6]) / cy
    r33 = L * x[10]

    R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    return R, r11, r12, r13, r21, r22, r23, r31, r32, r33
