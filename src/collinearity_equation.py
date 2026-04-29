import numpy as np


def auxilary_terms_coll_equation(
    Xi: float, Yi: float, Zi: float, X0: float, Y0: float, Z0: float, R: np.ndarray
) -> tuple:
    """
    Calculation of auxilary terms of collinearity equation.

    Notation
    --------
    Since all code is written in english the used notations differs from the notation in the script.

    script  code    description

    kx      n_x     numerator of x collinearity equation
    ky      n_y     numerator of y collinearity equation
    N       d       denominator of both equations

    Parameters
    ----------
    Xi : float
        x coordinate of object point i.
    Yi : float
        y coordinate of object point i.
    Zi : float
        z coordinate of object point i.
    X0 : float
        x coordinate of camera projection center.
    Y0 : float
        y coordinate of camera projection center.
    Z0 : float
        z coordinate of camera projection center.
    R : np.ndarray (3x3)
        Rotation matrix defining orientation of camera in object coordinate
        system at time of image capturing.

    Returns
    -------
    _ : (n_x, n_y, d)
        Auxilaray terms defining numerator of x collinearity equation (n_x),
        numerator of y collinearity equation (n_y) and denominator of both equations (d).
    """
    n_x = R[0, 0] * (Xi - X0) + R[1, 0] * (Yi - Y0) + R[2, 0] * (Zi - Z0)
    n_y = R[0, 1] * (Xi - X0) + R[1, 1] * (Yi - Y0) + R[2, 1] * (Zi - Z0)
    d = R[0, 2] * (Xi - X0) + R[1, 2] * (Yi - Y0) + R[2, 2] * (Zi - Z0)

    return n_x, n_y, d


def collinearity_equation(
    Xi: float,
    Yi: float,
    Zi: float,
    X0: float,
    Y0: float,
    Z0: float,
    x0: float,
    y0: float,
    pd: float,
    R: np.ndarray,
) -> tuple:
    """
    Transformation of object coordinates of point i (Xi, Yi, Zi) in to image
    space by collinearity equation.

    Notation
    --------
    Since all code is written in english the used notations differs from the notation in the script.

    script  code    description

    kx      n_x     numerator of x collinearity equation
    ky      n_y     numerator of y collinearity equation
    N       d       denominator of both equations

    Parameters
    ----------
    Xi : float
        x coordinate of object point i.
    Yi : float
        y coordinate of object point i.
    Zi : float
        z coordinate of object point i.
    X0 : float
        x coordinate of camera projection center.
    Y0 : float
        y coordinate of camera projection center.
    Z0 : float
        z coordinate of camera projection center.
    x0 : float
        x cooridnate of principal point.
    y0 : float
        y cooridnate of principal point.
    pd : float
        principal distance (sometimes also called camera constant)
    R : np.ndarray (3x3)
        Rotation matrix defining orientation of camera in object coordinate
        system at time of image capturing.

    Returns
    -------
    _ : (x,y)
        Image coordinates.
    """
    n_x, n_y, d = auxilary_terms_coll_equation(Xi, Yi, Zi, X0, Y0, Z0, R)
    x = x0 + (-pd * (n_x / d))
    y = y0 + (-pd * (n_y / d))

    return (x, y)


def part_derivatives_coll_equation(
    Xi: float,
    Yi: float,
    Zi: float,
    X0: float,
    Y0: float,
    Z0: float,
    pd: float,
    kappa: float,
    R: np.ndarray,
) -> tuple:
    """
    Calculates the partial derivatives of the collinearity equation for the
    parameters of the exterior orientation of the camera.

    Notation
    --------
    Since all code is written in english the used notations differs from the notation in the script.

    script  code    description

    kx      n_x     numerator of x collinearity equation
    ky      n_y     numerator of y collinearity equation
    N       d       denominator of both equations
    ck      pd      principal distance / camera constant

    Parameters
    ----------
    Xi : float
        x coordinate of object point i.
    Yi : float
        y coordinate of object point i.
    Zi : float
        z coordinate of object point i.
    X0 : float
        x coordinate of camera projection center.
    Y0 : float
        y coordinate of camera projection center.
    Z0 : float
        z coordinate of camera projection center.
    pd : float
        principal distance (also called camera constant)
    kappa : float
        rotation angle around z-axis of object coordinate system of camera
        at time of image capturing.
    R : np.ndarray (3x3)
        Rotation matrix defining orientation of camera in object coordinate
        system at time of image capturing.

    Returns
    -------
    _ : [x_X0, x_Y0, x_Z0, x_omega, x_phi, x_kappa, y_X0, y_Y0, y_Z0, y_omega, y_phi, y_kappa]
        Partial derivatives of the collinearity equation for the parameters
        of the exterior orientation of the camera.
    """
    n_x, n_y, d = auxilary_terms_coll_equation(Xi, Yi, Zi, X0, Y0, Z0, R)

    x_X0 = (-pd / d**2) * (R[0, 2] * n_x - R[0, 0] * d)
    x_Y0 = (-pd / d**2) * (R[1, 2] * n_x - R[1, 0] * d)
    x_Z0 = (-pd / d**2) * (R[2, 2] * n_x - R[2, 0] * d)
    x_omega = (-pd / d) * (
        (n_x / d) * (R[2, 2] * (Yi - Y0) - R[1, 2] * (Zi - Z0))
        - (R[2, 0] * (Yi - Y0))
        + (R[1, 0] * (Zi - Z0))
    )
    x_phi = (-pd / d) * (n_x / d * (n_y * np.sin(kappa) - n_x * np.cos(kappa)) - d * np.cos(kappa))
    x_kappa = (-pd / d) * n_y

    y_X0 = (-pd / d**2) * (R[0, 2] * n_y - R[0, 1] * d)
    y_Y0 = (-pd / d**2) * (R[1, 2] * n_y - R[1, 1] * d)
    y_Z0 = (-pd / d**2) * (R[2, 2] * n_y - R[2, 1] * d)
    y_omega = (-pd / d) * (
        (n_y / d) * (R[2, 2] * (Yi - Y0) - R[1, 2] * (Zi - Z0))
        - (R[2, 1] * (Yi - Y0))
        + (R[1, 1] * (Zi - Z0))
    )
    y_phi = (-pd / d) * (n_y / d * (n_x * np.cos(kappa) + n_y * np.sin(kappa)) + d * np.sin(kappa))
    y_kappa = pd / d * n_x

    return (
        x_X0,
        x_Y0,
        x_Z0,
        x_omega,
        x_phi,
        x_kappa,
        y_X0,
        y_Y0,
        y_Z0,
        y_omega,
        y_phi,
        y_kappa,
    )
