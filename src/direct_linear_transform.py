import numpy as np


def direct_linear_transform(img_points: np.ndarray, obj_points: np.ndarray) -> tuple:
    """
    Calculates the exterior orientation of a perspective camera by direct linear transform
    (projection center (X0, Y0, Z0) and orientation as rotation matrix)

    Parameters
    ----------
    img_points : np.ndarray
        Array of observed image coordinates of form (N, (x, y)).
    obj_points : np.ndarray
        Array of corresponding object points of form (N, (X, Y, Z)).

    Returns
    -------
    tuple
        Projection center as array (X, Y, Z)T, rotation matrix as array and
        interiorior orientation as list[x0, y0, cx, cy].

    """
    A = np.empty((img_points.shape[0] * 2, 11), dtype=np.float64)
    f_vec = np.empty((img_points.shape[0] * 2, 1), dtype=np.float64)

    # setup design-matrix (A-Matrix)
    for i, p in enumerate(img_points):
        x, y = p
        X, Y, Z = obj_points[i, :]

        A[2 * i : 2 * i + 2, :] = np.array(
            [
                [X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z],
                [0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z],
            ]
        )

        f_vec[2 * i : 2 * i + 2, :] = np.array([[x], [y]])

    # least square adjustment
    # Use solve instead of an explicit inverse: numerically cleaner, same result.
    x = np.linalg.solve(A.T @ A, A.T @ f_vec).ravel()

    # helper variable
    # The DLT scale is ambiguous (+/-). The sign below matches the rotationa
    # convention used in the collinearity equations and in the unit tests.
    L = -1.0 / np.sqrt(x[8] ** 2 + x[9] ** 2 + x[10] ** 2)

    # interior camera orientationa
    x0 = L**2 * (x[0] * x[8] + x[1] * x[9] + x[2] * x[10])
    y0 = L**2 * (x[4] * x[8] + x[5] * x[9] + x[6] * x[10])
    cx = np.sqrt(L**2 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) - x0**2)
    cy = np.sqrt(L**2 * (x[4] ** 2 + x[5] ** 2 + x[6] ** 2) - y0**2)

    # coefficient of rotation matrix
    r11 = L * (x0 * x[8] - x[0]) / cx
    r12 = L * (y0 * x[8] - x[4]) / cy
    r13 = L * x[8]
    r21 = L * (x0 * x[9] - x[1]) / cx
    r22 = L * (y0 * x[9] - x[5]) / cy
    r23 = L * x[9]
    r31 = L * (x0 * x[10] - x[2]) / cx
    r32 = L * (y0 * x[10] - x[6]) / cy
    r33 = L * x[10]
    # rotation matrix
    R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

    # projection center (exterior orientation)
    M = np.array(
        [
            [x[0], x[1], x[2]],
            [x[4], x[5], x[6]],
            [x[8], x[9], x[10]],
        ]
    )
    proj_center = -np.linalg.solve(M, np.array([[x[3]], [x[7]], [1.0]]))

    return R, proj_center, [x0, y0, cx, cy]
