import numpy as np

def normalize(vectors):
    # type: (numpy.ndarray) -> numpy.ndarray
    """Normalize vectors.

    Parameters:
        vectors (numpy.ndarray): The vectors, as rows.

    Returns:
        numpy.ndarray: The normalized vec.
    """
    flat = (len(vectors.shape) == 1)
    if flat:
        vectors = vectors[np.newaxis, :]
    result = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    if flat:
        return result[0]
    else:
        return result


def project(vectors, bases, change_coords=False):
    # type: (numpy.ndarray, numpy.ndarray, bool) -> numpy.ndarray
    """Project the vectors on to the subspace formed by the bases.

    Parameters:
        vectors (numpy.ndarray): The vectors to project, as rows.
        bases (numpy.ndarray): The bases to project on to.
        change_coords (bool): If True, the result will be in the coordinate
            system defined by the bases. Defaults to False.

    Returns:
        numpy.ndarray: The projection.
    """
    flat = (len(vectors.shape) == 1)
    if flat:
        vectors = vectors[np.newaxis, :]
    if len(bases.shape) == 1:
        bases = bases[np.newaxis, :]
    try:
        result = np.linalg.inv(bases @ bases.T) @ bases @ vectors.T
    except ValueError:
        breakpoint()
    if not change_coords:
        result = bases.T @ result
    result = result.T
    if flat:
        return result[0]
    else:
        return result


def reject(vectors, bases):
    # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """Reject the vectors from the subspace formed by the bases.

    Parameters:
        vectors (numpy.ndarray): The vector to reject.
        bases (numpy.ndarray): The vector to reject from.

    Returns:
        numpy.ndarray: The rejection.
    """
    return vectors - project(vectors, bases)


def recenter(vectors):
    # type: (numpy.ndarray) -> numpy.ndarray
    """Redefine vectors as coming from their centroid.

    Parameters:
        vectors (numpy.ndarray): The vectors, as rows

    Returns:
        numpy.ndarray: The new vectors.
    """
    centroid = np.mean(vectors, axis=0)
    extrusion = np.repeat(centroid[np.newaxis, :], [vectors.shape[0]], axis=0)
    return vectors - extrusion
