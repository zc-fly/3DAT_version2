"""Module for operations related to geometrical analysis."""
import logging
import functools
from typing import List, Tuple, Optional

import numpy as np
from scipy import optimize, spatial
from scipy.spatial import qhull
from .camera_model import CameraModel

logger = logging.getLogger(__name__)

FIELD_OF_PLAY_XY_MARGIN = 1.20  # 20% around FieldOfPlay
BOUND_MIN_Z, BOUND_MAX_Z = -1.0, 5.0  # There shouldn't be anyone under the ground
BOUND_MIN_NORM, BOUND_MAX_NORM = -10.0, 10.0  # Typical values for normalization values


def add_bounding_box_padding(bounding_box: List[float], percentage: float = 10.0) -> List[float]:
    """Add padding around a given bounding box.

    :param bounding_box: Single bounding box in a format [x, y, w, h]
    :param percentage: (optional) how much bounding box should be padded
    :return: padded bounding box in a format [x, y, w, h]
    """
    scale_factor = (100.0 + percentage) / 100.0
    x, y, w, h = bounding_box
    new_width = w * scale_factor
    new_height = h * scale_factor
    shift_x = (new_width - w) / 2.0
    shift_y = (new_height - h) / 2.0
    return [x - shift_x, y - shift_y, new_width, new_height]


def get_half_space(points: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Build a half space (A * x + b <= 0) based on three points.

    :param points: tuple of three points on which plane should be built
    :return: half space where A * x + b <= 0 described as [-Ax, -Ay, -Az, b]
    """
    v_1 = points[0] - points[2]
    v_2 = points[1] - points[2]

    A = np.cross(v_1, v_2)  # Normal vector of plane
    b = np.dot(A, points[2])

    return np.hstack((-A, b))


def get_cone(bounding_box: List[float], camera: CameraModel) \
        -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build cone described as corners and half spaces.

    :param bounding_box: represented as list of [x, y, w, h]
    :param camera: object of calibrated camera model
    :return: tuple of real world positions for cone corners and half spaces describing cone
    """
    x, y, w, h = bounding_box
    pixel_corners = np.array([
        [x, y],  # Upper Left
        [x + w, y],  # Upper Right
        [x, y + h],  # Lower Left
        [x + w, y + h],  # Lower Right
    ])

    real_world_corners = [camera.image_to_world(corner) for corner in pixel_corners]
    upper_left, upper_right, lower_left, lower_right = real_world_corners
    half_spaces = [
        get_half_space((upper_left, upper_right, camera.position)),
        get_half_space((lower_left, upper_left, camera.position)),
        get_half_space((lower_right, lower_left, camera.position)),
        get_half_space((upper_right, lower_right, camera.position)),
    ]
    return real_world_corners, half_spaces


def get_polyhedron(half_spaces: List[np.ndarray], field_of_play: np.ndarray = None) \
        -> Optional[spatial.HalfspaceIntersection]:
    """Build a polyhedron (3D shape) based on given half spaces describing it.

    :param half_spaces: list of half spaces, where each one is in form of A * x + b <= 0 and
                        described as [-Ax, -Ay, -Az, b]
    :param field_of_play: 3x4 numpy array with FieldOfPlay
    :return: Halfspace Intersection object from SciPy library
    """
    if len(half_spaces) <= 4:
        return None  # We cannot create a Polyhedron without at least 5 half spaces

    try:
        inequalities = np.array(half_spaces)
        inequalities_shape = list(inequalities.shape)

        # For our use-case, objective function does not matter too much as we want to achieve
        # any feasible point inside of the polyhedron
        objective_func = np.zeros((inequalities_shape[1], ))
        objective_func[-1] = -1

        # Normal vector allows us to search for a point inside of the polyhedron instead of
        # polyhedron's vertices
        norm_vector = np.reshape(
            np.linalg.norm(inequalities[:, :-1], axis=1), (inequalities_shape[0], 1))
        A = np.hstack((inequalities[:, :-1], norm_vector))
        b = -inequalities[:, -1:]

        # Search for interior point within bounds based on FieldOfPlay. Limitations to
        # number of iterations allow us to reduce compute time
        bounds = get_bounds(field_of_play)
        result = optimize.linprog(
            objective_func, A_ub=A, b_ub=b, method='interior-point', bounds=bounds,
            options={'tol': 1e-5, 'maxiter': 10}
        )
        interior_point = result.x[:-1]
        return spatial.HalfspaceIntersection(inequalities, interior_point)
    except qhull.QhullError:
        return None  # Feasible point was not inside of the polyhedron
    except ValueError:
        logger.warning('Could not calculate feasible point. Probably, we could find it with '
                       'increased tolerance.')
        return None


def get_bounds(field_of_play: np.ndarray = None) -> List[Tuple]:
    """Get bounds for optimization algorithm finding feasible point.

    By default, we should find feasible point anywhere in the 3D space. However, SciPy
    works slightly faster once we tell him that our point is somewhere on the FieldOfPlay as
    it reduces number of iterations in the optimization algorithm.

    :param field_of_play: (optional) 3x4 numpy array with FieldOfPlay
    :return: list of bounds for feasible points in each axis
    """
    bounds: List[Tuple] = [(None, None)] * 4
    if field_of_play is not None:
        x_min, y_min, _ = field_of_play.min(axis=0) * FIELD_OF_PLAY_XY_MARGIN
        x_max, y_max, _ = field_of_play.max(axis=0) * FIELD_OF_PLAY_XY_MARGIN
        bounds = [(x_min, x_max), (y_min, y_max), (BOUND_MIN_Z, BOUND_MAX_Z),
                  (BOUND_MIN_NORM, BOUND_MAX_NORM)]
    return bounds


def get_vertices_for_polyhedron(polyhedron: spatial.HalfspaceIntersection) -> np.ndarray:
    """Compute vertices that describe given polyhedron.

    :param polyhedron: Halfspace Intersection object from SciPy library
    :return: numpy array (Nx3 matrix) for N vertices (X, Y, Z)
    """
    return polyhedron.intersections


def get_polyhedron_height(polyhedron: spatial.HalfspaceIntersection) -> float:
    """Compute height of a given polyhedron.

    :param polyhedron: Halfspace Intersection object from SciPy library
    :return: polyhedron's height
    """
    vertices = get_vertices_for_polyhedron(polyhedron)

    try:
        if vertices.size == 0 or vertices.shape[0] < 2:
            return 0.0  # There are not enough vertices for height calculation

        height = max(vertices[:, 2]) - min(vertices[:, 2])
        return height
    except AttributeError:
        return 0.0


def get_polyhedron_volume(polyhedron: spatial.HalfspaceIntersection) -> float:
    """Compute volume of a given polyhedron.

    In our case, polyhedrons are always convex shapes. That's because it is a result of
    camera cones overlapping. In other cases, below implementation won't work due to assumption
    of convexity.

    :param polyhedron: Halfspace Intersection object from SciPy library
    :return: polyhedron's volume
    """
    vertices = get_vertices_for_polyhedron(polyhedron)
    if vertices.size == 0 or vertices.shape[0] <= 4:
        return 0.0  # There are not enough vertices for volume calculation

    try:
        convex_hull = qhull.ConvexHull(vertices)
        return convex_hull.volume  # pylint: disable=no-member  # It is created in runtime
    except qhull.QhullError:
        # Convex Hull may raise an exception in case of emtpy list of vertices or polyhedron
        # that is not convex (shouldn't happen in our case)
        return 0.0


def compute_intersection_over_union(polyhedron_a: spatial.HalfspaceIntersection,
                                    polyhedron_b: spatial.HalfspaceIntersection) \
        -> float:
    """Calculate Intersection over Union metric between two Polyhedrons.

    :param polyhedron_a: First Polyhedron
    :param polyhedron_b: Second Polyhedron
    :return: Intersection over Union (IoU) metric computed for both Polyhedrons
    """
    half_spaces_a = list(polyhedron_a.halfspaces)
    half_spaces_b = list(polyhedron_b.halfspaces)
    intersection_half_spaces = half_spaces_a + half_spaces_b
    intersection_polyhedron = get_polyhedron(intersection_half_spaces)
    if not intersection_polyhedron:
        return 0.0  # Polyhedrons does not overlap

    volume_a = get_polyhedron_volume(polyhedron_a)
    volume_b = get_polyhedron_volume(polyhedron_b)
    intersection_volume = get_polyhedron_volume(intersection_polyhedron)
    union_volume = volume_a + volume_b - intersection_volume

    try:
        return intersection_volume / union_volume
    except ZeroDivisionError:
        return 0.0  # Both polyhedrons seems to be empty if they union was equal to zero


def compute_location_error(polyhedron_a: spatial.HalfspaceIntersection,
                           polyhedron_b: spatial.HalfspaceIntersection) -> float:
    """Calculate location error between two bounding boxes in XY-axes.

    :return: distance between centers of two polyhedrons
    """
    vertices_a = get_vertices_for_polyhedron(polyhedron_a)
    vertices_b = get_vertices_for_polyhedron(polyhedron_b)
    if vertices_a.size == 0 or vertices_b.size == 0:
        return 0.0

    location_xy_a = np.mean(vertices_a, axis=0)[:2]  # only X, Y axes
    location_xy_b = np.mean(vertices_b, axis=0)[:2]  # only X, Y axes
    return np.linalg.norm(location_xy_a - location_xy_b)


def compute_flat_iou(box_1: List[float], box_2: List[float]) -> float:
    """Calculates intersection over union for two 2D bounding boxes

    :param box_1: first bounding box
    :param box_2: second bounding box
    :return: intersection over union metric
    """
    x_1, y_1, w_1, h_1 = box_1
    x_2, y_2, w_2, h_2 = box_2

    # Looking for intersection:
    xy_intersection = (max(x_1, x_2), max(y_1, y_2))
    xy_max = min(x_1 + w_1, x_2 + w_2), min(y_1 + h_1, y_2 + h_2)
    wh_intersection = xy_max[0] - xy_intersection[0], xy_max[1] - xy_intersection[1]
    if wh_intersection[0] > 0 and wh_intersection[1] > 0:
        vol_intersection = wh_intersection[0] * wh_intersection[1]
    else:
        vol_intersection = 0.0  # the two boxes do not intersect

    vol_union = w_1 * h_1 + w_2 * h_2 - vol_intersection

    try:
        return vol_intersection / vol_union
    except ZeroDivisionError:
        return 0.0  # both boxes where empty


def polyhedron_to_box(polyhedron: spatial.HalfspaceIntersection,
                      camera: CameraModel) -> List[float]:
    """Calculates projection of a polyhedron in real world coordinates onto cameras image.

    :param polyhedron: Polyhedron in real world coordinates
    :param camera: object of calibrated camera model
    :return: minimal rectangel bounding box containing the projection, in form of [x, y, w, h]
    """
    vertices = get_vertices_for_polyhedron(polyhedron)
    pixel_points = [camera.world_to_image(vertex) for vertex in vertices]
    x_max, y_max = np.amax(pixel_points, axis=0)
    x_min, y_min = np.amin(pixel_points, axis=0)
    bounding_box = [x_min, y_min, x_max - x_min, y_max - y_min]
    return bounding_box


def project_2d_point_to_ground_plane(position2d: List[int], field_of_play: np.ndarray,
                                     camera: CameraModel) -> List[float]:
    """Project 2D Position on the image into 3D Position on the ground level (Field of Play).

    :param position2d: position 2D on the image
    :param field_of_play: 3x4 numpy array with FieldOfPlay
    :param camera: object of calibrated camera model
    :return: position 3D on the ground plane
    """
    plane_normal = np.cross(field_of_play[1] - field_of_play[0],
                            field_of_play[2] - field_of_play[0])
    plane_point = field_of_play[0]  # Any vertex should be fine

    ray_start_point = camera.position
    ray_end_point = camera.image_to_world(position2d, z=1.0, vector_length=1)
    ray_to_athlete = ray_end_point - ray_start_point

    plane_to_camera = ray_start_point - plane_point
    segment_length = -(np.dot(plane_normal, plane_to_camera) /
                       np.dot(plane_normal, ray_to_athlete))
    position_3d = ray_start_point + segment_length * ray_to_athlete
    return position_3d


def distance_from_point_to_ray(point: np.ndarray, ray_start: np.ndarray, ray_direction: np.ndarray)\
        -> float:
    """Return distance from a given point to ray.

    :param point: point as numpy array
    :param ray_start: ray starting point as numpy array
    :param ray_direction: ray direction as numpy array
    :return: distance as float
    """
    return np.linalg.norm(np.cross(ray_direction, point - ray_start))


def triangulation_optimization_func(rays: List[Tuple[np.ndarray, np.ndarray]],
                                    proposed_point: np.ndarray) -> np.ndarray:
    """Optimize distance from a point to all rays.

    :param rays: list of tuples with starting points and vectors from a given point
    :param proposed_point: point, which was proposed by triangulation
    :return: array of distances for each ray
    """
    return np.array([
        distance_from_point_to_ray(proposed_point, ray_start, ray_direction)
        for ray_start, ray_direction in rays
    ])


def triangulate_rays(rays: List[Tuple[np.ndarray, np.ndarray]],
                     starting_point: np.ndarray = None) -> np.ndarray:
    """Triangulate given rays and return optimal 3D Point.

    :param rays: list of tuples with starting points and vectors from a given point
    :param starting_point: (optional) starting point, which can be used for faster
    :return: point that is in the closest distance to all given rays
    """
    method = 'lm' if len(rays) >= 3 else 'trf'  # LM is faster but works only if m >= n
    starting_point = starting_point if starting_point is not None else np.array((0, 0, 0))
    optimization_func = functools.partial(triangulation_optimization_func, rays)
    result = optimize.least_squares(optimization_func, starting_point, xtol=1e-04,
                                    loss='linear', method=method)
    return result.x
