"""Storm-centered radar images."""

import os
import copy
import glob
import numpy
from scipy.interpolate import interp1d as scipy_interp1d
import netCDF4
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PADDING_VALUE = 0.
GRID_SPACING_TOLERANCE_DEG = 1e-4
AZ_SHEAR_GRID_SPACING_MULTIPLIER = 2

LABEL_FILE_EXTENSION = '.nc'
ELEVATION_COLUMN = 'elevation_m_asl'

GRIDRAD_TIME_INTERVAL_SEC = 300
TIME_FORMAT = '%Y-%m-%d-%H%M%S'
TIME_FORMAT_REGEX = (
    '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9]'
)

FIRST_STORM_ROW_KEY = 'first_storm_image_row'
LAST_STORM_ROW_KEY = 'last_storm_image_row'
FIRST_STORM_COLUMN_KEY = 'first_storm_image_column'
LAST_STORM_COLUMN_KEY = 'last_storm_image_column'
NUM_TOP_PADDING_ROWS_KEY = 'num_padding_rows_at_top'
NUM_BOTTOM_PADDING_ROWS_KEY = 'num_padding_rows_at_bottom'
NUM_LEFT_PADDING_COLS_KEY = 'num_padding_columns_at_left'
NUM_RIGHT_PADDING_COLS_KEY = 'num_padding_columns_at_right'

ROTATED_NON_SHEAR_LATITUDES_COLUMN = 'rotated_lat_matrix_non_shear_deg'
ROTATED_NON_SHEAR_LONGITUDES_COLUMN = 'rotated_lng_matrix_non_shear_deg'
ROTATED_SHEAR_LATITUDES_COLUMN = 'rotated_lat_matrix_for_shear_deg'
ROTATED_SHEAR_LONGITUDES_COLUMN = 'rotated_lng_matrix_for_shear_deg'

STORM_IMAGE_MATRIX_KEY = 'storm_image_matrix'
FULL_IDS_KEY = 'full_storm_id_strings'
VALID_TIMES_KEY = 'valid_times_unix_sec'
RADAR_FIELD_NAME_KEY = 'radar_field_name'
RADAR_HEIGHT_KEY = 'radar_height_m_agl'
ROTATED_GRIDS_KEY = 'rotated_grids'
ROTATED_GRID_SPACING_KEY = 'rotated_grid_spacing_metres'
LABEL_VALUES_KEY = 'label_values'

RADAR_FIELD_NAMES_KEY = 'radar_field_names'
RADAR_HEIGHTS_KEY = 'radar_heights_m_agl'
IMAGE_FILE_NAMES_KEY = 'image_file_name_matrix'
FIELD_NAME_BY_PAIR_KEY = 'field_name_by_pair'
HEIGHT_BY_PAIR_KEY = 'height_by_pair_m_agl'

ROW_DIMENSION_KEY = 'grid_row'
COLUMN_DIMENSION_KEY = 'grid_column'
CHARACTER_DIMENSION_KEY = 'storm_id_character'
STORM_OBJECT_DIMENSION_KEY = 'storm_object'

STORM_COLUMNS_NEEDED = [
    tracking_utils.FULL_ID_COLUMN, tracking_utils.VALID_TIME_COLUMN,
    tracking_utils.SPC_DATE_COLUMN, tracking_utils.CENTROID_LATITUDE_COLUMN,
    tracking_utils.CENTROID_LONGITUDE_COLUMN,
    tracking_utils.EAST_VELOCITY_COLUMN, tracking_utils.NORTH_VELOCITY_COLUMN
]

# Highest and lowest points in continental U.S.
LOWEST_POINT_IN_CONUS_M_ASL = -100.
HIGHEST_POINT_IN_CONUS_M_ASL = 4500.

DEFAULT_NUM_IMAGE_ROWS = 32
DEFAULT_NUM_IMAGE_COLUMNS = 32
DEFAULT_ROTATED_GRID_SPACING_METRES = 1500.
DEFAULT_RADAR_HEIGHTS_M_AGL = numpy.linspace(1000, 12000, num=12, dtype=int)

DEFAULT_MYRORSS_MRMS_FIELD_NAMES = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME,
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME,
    radar_utils.REFL_NAME, radar_utils.REFL_COLUMN_MAX_NAME,
    radar_utils.REFL_0CELSIUS_NAME, radar_utils.REFL_M10CELSIUS_NAME,
    radar_utils.REFL_M20CELSIUS_NAME, radar_utils.REFL_LOWEST_ALTITUDE_NAME,
    radar_utils.MESH_NAME, radar_utils.SHI_NAME, radar_utils.VIL_NAME
]
DEFAULT_GRIDRAD_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.DIVERGENCE_NAME
]
AZIMUTHAL_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
]


def _check_extraction_args(
        num_storm_image_rows, num_storm_image_columns, rotate_grids,
        rotated_grid_spacing_metres, radar_field_names, radar_source,
        radar_heights_m_agl=None, reflectivity_heights_m_agl=None):
    """Checks input args for extraction of storm-centered radar images.

    Specifically, this method checks input args for
    `extract_storm_images_myrorss_or_mrms` or `extract_storm_images_gridrad`.

    :param num_storm_image_rows: Number of rows in each storm-centered image.
        Must be even.
    :param num_storm_image_columns: Number columns in each storm-centered image.
        Must be even.
    :param rotate_grids: Boolean flag.  If True, each grid will be rotated so
        that storm motion is in the +x-direction; thus, storm-centered grids
        will be equidistant.  If False, each storm-centered grid will be a
        contiguous rectangle extracted from the full grid; thus, storm-centered
        grids will be lat-long.
    :param rotated_grid_spacing_metres: [used only if rotate_grids = True]
        Spacing between grid points in adjacent rows or columns.
    :param radar_field_names: 1-D list with names of radar fields.
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_heights_m_agl: [may be None]
        1-D list of radar heights (metres above ground level).  One storm-
        centered image will be created for each tuple of storm object, field in
        `radar_field_names`, and height in `radar_heights_m_agl`.
    :param reflectivity_heights_m_agl: [may be None]
        1-D list of reflectivity heights (metres above ground level).  One
        storm-centered image will be created for each pair of storm object and
        field in `radar_field_names` (other than "reflectivity_dbz").  One
        storm-centered image will also be created for each pair of storm object
        and "reflectivity_dbz" at height in `reflectivity_heights_m_agl`.
    :raises: ValueError: if `num_storm_image_rows` or `num_storm_image_columns`
        is not even.
    """

    error_checking.assert_is_integer(num_storm_image_rows)
    error_checking.assert_is_greater(num_storm_image_rows, 0)

    if num_storm_image_rows != rounder.round_to_nearest(
            num_storm_image_rows, 2):
        error_string = (
            'Number of rows per storm-centered image ({0:d}) should be even.'
        ).format(num_storm_image_rows)

        raise ValueError(error_string)

    error_checking.assert_is_integer(num_storm_image_columns)
    error_checking.assert_is_greater(num_storm_image_columns, 0)

    if num_storm_image_columns != rounder.round_to_nearest(
            num_storm_image_columns, 2):
        error_string = (
            'Number of columns per storm-centered image ({0:d}) should be even.'
        ).format(num_storm_image_columns)

        raise ValueError(error_string)

    error_checking.assert_is_boolean(rotate_grids)
    if rotate_grids:
        error_checking.assert_is_greater(rotated_grid_spacing_metres, 0.)

    error_checking.assert_is_string_list(radar_field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(radar_field_names), num_dimensions=1)

    radar_utils.check_data_source(radar_source)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        error_checking.assert_is_geq_numpy_array(radar_heights_m_agl, 0)
        error_checking.assert_is_numpy_array(
            numpy.array(radar_heights_m_agl), num_dimensions=1)

    elif reflectivity_heights_m_agl is not None:
        error_checking.assert_is_geq_numpy_array(
            reflectivity_heights_m_agl, 0)
        error_checking.assert_is_numpy_array(
            numpy.array(reflectivity_heights_m_agl), num_dimensions=1)


def _check_grid_spacing(
        new_metadata_dict, orig_lat_spacing_deg, orig_lng_spacing_deg):
    """Ensures consistency between grid spacing in new and original radar files.

    :param new_metadata_dict: Dictionary created by
        `myrorss_and_mrms_io.read_metadata_from_raw_file` or
        `gridrad_io.read_metadata_from_full_grid_file`.
    :param orig_lat_spacing_deg: Spacing (deg N) between meridionally adjacent
        grid points in original file.
    :param orig_lng_spacing_deg: Spacing (deg E) between zonally adjacent grid
        points in original file.
    :return: orig_lat_spacing_deg: See above.
    :return: orig_lng_spacing_deg: See above.
    :raises: ValueError: if grid spacings are inconsistent.
    """

    is_field_az_shear = (
        radar_utils.FIELD_NAME_COLUMN in new_metadata_dict and
        new_metadata_dict[radar_utils.FIELD_NAME_COLUMN] in
        AZIMUTHAL_SHEAR_FIELD_NAMES
    )

    if is_field_az_shear:
        new_lat_spacing_deg = (
            AZ_SHEAR_GRID_SPACING_MULTIPLIER *
            new_metadata_dict[radar_utils.LAT_SPACING_COLUMN])
        new_lng_spacing_deg = (
            AZ_SHEAR_GRID_SPACING_MULTIPLIER *
            new_metadata_dict[radar_utils.LNG_SPACING_COLUMN])
    else:
        new_lat_spacing_deg = new_metadata_dict[radar_utils.LAT_SPACING_COLUMN]
        new_lng_spacing_deg = new_metadata_dict[radar_utils.LNG_SPACING_COLUMN]

    new_lat_spacing_deg = rounder.round_to_nearest(
        new_lat_spacing_deg, GRID_SPACING_TOLERANCE_DEG)
    new_lng_spacing_deg = rounder.round_to_nearest(
        new_lng_spacing_deg, GRID_SPACING_TOLERANCE_DEG)

    if orig_lat_spacing_deg is None:
        orig_lat_spacing_deg = new_lat_spacing_deg + 0.
        orig_lng_spacing_deg = new_lng_spacing_deg + 0.

    if (orig_lat_spacing_deg != new_lat_spacing_deg or
            orig_lng_spacing_deg != new_lng_spacing_deg):
        error_string = (
            'Original file has grid spacing of {0:.4f} deg N, {1:.4f} deg E.  '
            'New file has spacing of {2:.4f} deg N, {3:.4f} deg E.'
        ).format(orig_lat_spacing_deg, orig_lng_spacing_deg,
                 new_lat_spacing_deg, new_lng_spacing_deg)

        raise ValueError(error_string)

    return orig_lat_spacing_deg, orig_lng_spacing_deg


def _check_storm_images(
        storm_image_matrix, full_id_strings, valid_times_unix_sec,
        radar_field_name, radar_height_m_agl, rotated_grids,
        rotated_grid_spacing_metres=None):
    """Checks storm-centered radar images for errors.

    L = number of storm objects
    M = number of rows in each image
    N = number of columns in each image

    :param storm_image_matrix: L-by-M-by-N numpy array of storm-centered radar
        measurements.
    :param full_id_strings: length-L list of full storm IDs.
    :param valid_times_unix_sec: length-L numpy array of storm times.
    :param radar_field_name: Name of radar field.
    :param radar_height_m_agl: Height (metres above ground level) of radar
        field.
    :param rotated_grids: Boolean flag.  If True, each grid is rotated so that
        storm motion is in the +x-direction.
    :param rotated_grid_spacing_metres: [used iff `rotate_grids = True`]
        Spacing between grid points in adjacent rows or columns.
    """

    error_checking.assert_is_numpy_array_without_nan(storm_image_matrix)
    error_checking.assert_is_numpy_array(storm_image_matrix, num_dimensions=3)

    num_storm_objects = storm_image_matrix.shape[0]
    these_expected_dim = numpy.array([num_storm_objects], dtype=int)

    error_checking.assert_is_string_list(full_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(full_id_strings), exact_dimensions=these_expected_dim
    )

    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec, exact_dimensions=these_expected_dim)

    radar_utils.check_field_name(radar_field_name)
    error_checking.assert_is_geq(radar_height_m_agl, 0)

    error_checking.assert_is_boolean(rotated_grids)
    if rotated_grids:
        error_checking.assert_is_greater(rotated_grid_spacing_metres, 0.)


def _find_input_heights_needed(
        storm_elevations_m_asl, desired_radar_heights_m_agl, radar_source):
    """Finds radar heights needed, in metres above sea level.

    :param storm_elevations_m_asl: 1-D numpy array of storm elevations (metres
        above sea level).
    :param desired_radar_heights_m_agl: 1-D numpy array of desired radar heights
        (metres above ground level).
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :return: desired_radar_heights_m_asl: 1-D numpy array of desired radar
        heights (metres above sea level).
    """

    min_radar_height_m_asl = (
        numpy.min(storm_elevations_m_asl) +
        numpy.min(desired_radar_heights_m_agl)
    )
    max_radar_height_m_asl = (
        numpy.max(storm_elevations_m_asl) +
        numpy.max(desired_radar_heights_m_agl)
    )

    desired_radar_heights_m_asl = radar_utils.get_valid_heights(
        data_source=radar_source, field_name=radar_utils.REFL_NAME)
    good_indices = numpy.where(numpy.logical_and(
        desired_radar_heights_m_asl >= min_radar_height_m_asl,
        desired_radar_heights_m_asl <= max_radar_height_m_asl
    ))[0]

    if 0 not in good_indices:
        these_indices = numpy.array([numpy.min(good_indices) - 1], dtype=int)
        good_indices = numpy.concatenate((good_indices, these_indices))

    max_possible_index = len(desired_radar_heights_m_asl) - 1
    if max_possible_index not in good_indices:
        these_indices = numpy.array([numpy.max(good_indices) + 1], dtype=int)
        good_indices = numpy.concatenate((good_indices, these_indices))

    return desired_radar_heights_m_asl[numpy.sort(good_indices)]


def _fields_and_heights_to_pairs(
        radar_field_names, reflectivity_heights_m_agl, radar_source):
    """Converts lists of fields and reflectivity heights to field-height pairs.

    C = number of field/height pairs

    :param radar_field_names: 1-D list with names of radar fields.
    :param reflectivity_heights_m_agl: 1-D numpy array of reflectivity heights
        (only for the field "reflectivity_dbz", in metres above ground level).
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :return: field_name_by_pair: length-C list with names of radar fields.
    :return: height_by_pair_m_agl: length-C numpy array of heights (metres above
        ground level).
    """

    error_checking.assert_is_numpy_array(
        numpy.array(radar_field_names), num_dimensions=1)

    field_name_by_pair = []
    height_by_pair_m_agl = []

    for this_field_name in radar_field_names:
        if this_field_name == radar_utils.REFL_NAME:
            error_checking.assert_is_geq_numpy_array(
                reflectivity_heights_m_agl, 0)
            error_checking.assert_is_numpy_array(
                reflectivity_heights_m_agl, num_dimensions=1)

            field_name_by_pair += (
                [this_field_name] * len(reflectivity_heights_m_agl))
            height_by_pair_m_agl += reflectivity_heights_m_agl.tolist()
        else:
            this_height_m_agl = radar_utils.get_valid_heights(
                data_source=radar_source, field_name=this_field_name)[0]

            field_name_by_pair.append(this_field_name)
            height_by_pair_m_agl.append(this_height_m_agl)

    height_by_pair_m_agl = numpy.round(
        numpy.array(height_by_pair_m_agl)
    ).astype(int)

    return field_name_by_pair, height_by_pair_m_agl


def _get_relevant_storm_objects(
        storm_object_table, valid_time_unix_sec, valid_spc_date_string):
    """Returns indices of relevant storm objects (at the given time & SPC date).

    :param storm_object_table: See doc for
        `extract_storm_images_myrorss_or_mrms` or
        `extract_storm_images_gridrad`.
    :param valid_time_unix_sec: Will find storm objects with this valid time.
    :param valid_spc_date_string: Will find storm objects on this SPC date
        (format "yyyymmdd").
    :return: relevant_indices: 1-D numpy array with indices of relevant storm
        objects.
    """

    relevant_flags = numpy.logical_and(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values ==
        valid_time_unix_sec,
        storm_object_table[tracking_utils.SPC_DATE_COLUMN].values ==
        valid_spc_date_string
    )

    return numpy.where(relevant_flags)[0]


def _rotate_grid_one_storm_object(
        centroid_latitude_deg, centroid_longitude_deg, eastward_motion_m_s01,
        northward_motion_m_s01, num_storm_image_rows, num_storm_image_columns,
        storm_grid_spacing_metres):
    """Generates lat-long coordinates for rotated, storm-centered grid.

    The grid is rotated so that storm motion is in the +x-direction.

    m = number of rows in storm-centered grid (must be even)
    n = number of columns in storm-centered grid (must be even)

    :param centroid_latitude_deg: Latitude (deg N) of storm centroid.
    :param centroid_longitude_deg: Longitude (deg E) of storm centroid.
    :param eastward_motion_m_s01: Eastward component of storm motion (metres per
        second).
    :param northward_motion_m_s01: Northward component of storm motion.
    :param num_storm_image_rows: m in the above discussion.
    :param num_storm_image_columns: n in the above discussion.
    :param storm_grid_spacing_metres: Spacing between grid points in adjacent
        rows or columns.
    :return: grid_point_lat_matrix_deg: m-by-n numpy array with latitudes
        (deg N) of grid points.
    :return: grid_point_lng_matrix_deg: m-by-n numpy array with longitudes
        (deg E) of grid points.
    """

    storm_bearing_deg = geodetic_utils.xy_to_scalar_displacements_and_bearings(
        x_displacements_metres=numpy.array([eastward_motion_m_s01]),
        y_displacements_metres=numpy.array([northward_motion_m_s01])
    )[-1][0]

    this_max_displacement_metres = storm_grid_spacing_metres * (
        num_storm_image_columns / 2 - 0.5)
    this_min_displacement_metres = -1 * this_max_displacement_metres
    x_prime_displacements_metres = numpy.linspace(
        this_min_displacement_metres, this_max_displacement_metres,
        num=num_storm_image_columns)

    this_max_displacement_metres = storm_grid_spacing_metres * (
        num_storm_image_rows / 2 - 0.5)
    this_min_displacement_metres = -1 * this_max_displacement_metres
    y_prime_displacements_metres = numpy.linspace(
        this_min_displacement_metres, this_max_displacement_metres,
        num=num_storm_image_rows)

    (x_prime_displ_matrix_metres, y_prime_displ_matrix_metres
    ) = grids.xy_vectors_to_matrices(
        x_unique_metres=x_prime_displacements_metres,
        y_unique_metres=y_prime_displacements_metres)

    (x_displacement_matrix_metres, y_displacement_matrix_metres
    ) = geodetic_utils.rotate_displacement_vectors(
        x_displacements_metres=x_prime_displ_matrix_metres,
        y_displacements_metres=y_prime_displ_matrix_metres,
        ccw_rotation_angle_deg=-(storm_bearing_deg - 90))

    (scalar_displacement_matrix_metres, bearing_matrix_deg
    ) = geodetic_utils.xy_to_scalar_displacements_and_bearings(
        x_displacements_metres=x_displacement_matrix_metres,
        y_displacements_metres=y_displacement_matrix_metres)

    start_latitude_matrix_deg = numpy.full(
        (num_storm_image_rows, num_storm_image_columns), centroid_latitude_deg)
    start_longitude_matrix_deg = numpy.full(
        (num_storm_image_rows, num_storm_image_columns), centroid_longitude_deg)

    return geodetic_utils.start_points_and_displacements_to_endpoints(
        start_latitudes_deg=start_latitude_matrix_deg,
        start_longitudes_deg=start_longitude_matrix_deg,
        scalar_displacements_metres=scalar_displacement_matrix_metres,
        geodetic_bearings_deg=bearing_matrix_deg)


def _rotate_grids_many_storm_objects(
        storm_object_table, num_storm_image_rows, num_storm_image_columns,
        storm_grid_spacing_metres, for_azimuthal_shear):
    """Creates rotated, storm-centered grid for each storm object.

    m = number of rows in each storm-centered grid
    n = number of columns in each storm-centered grid

    :param storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.
    storm_object_table.east_velocity_m_s01: Eastward storm velocity (metres per
        second).
    storm_object_table.north_velocity_m_s01: Northward storm velocity.
    :param num_storm_image_rows: m in the above discussion.
    :param num_storm_image_columns: n in the above discussion.
    :param storm_grid_spacing_metres: Spacing between grid points in adjacent
        rows or columns.
    :param for_azimuthal_shear: Boolean flag.  If True (False), the grids will
        be created will be used for azimuthal shear (other fields).  This option
        affects only the names of the output columns.
    :return: storm_object_table: Same as input, but with two new columns.
    storm_object_table.rotated_lat_matrix_non_shear_deg: m-by-n numpy array with
        latitudes (deg N) of grid points.
    storm_object_table.rotated_lng_matrix_non_shear_deg: m-by-n numpy array with
        longitudes (deg E) of grid points.
    If `for_azimuthal_shear = True`, these columns will be named
    "rotated_lat_matrix_for_shear_deg" and "rotated_lng_matrix_for_shear_deg",
    instead.
    """

    num_storm_objects = len(storm_object_table.index)
    list_of_latitude_matrices = [None] * num_storm_objects
    list_of_longitude_matrices = [None] * num_storm_objects

    for i in range(num_storm_objects):
        list_of_latitude_matrices[i], list_of_longitude_matrices[i] = (
            _rotate_grid_one_storm_object(
                centroid_latitude_deg=storm_object_table[
                    tracking_utils.CENTROID_LATITUDE_COLUMN].values[i],
                centroid_longitude_deg=storm_object_table[
                    tracking_utils.CENTROID_LONGITUDE_COLUMN].values[i],
                eastward_motion_m_s01=storm_object_table[
                    tracking_utils.EAST_VELOCITY_COLUMN].values[i],
                northward_motion_m_s01=storm_object_table[
                    tracking_utils.NORTH_VELOCITY_COLUMN].values[i],
                num_storm_image_rows=num_storm_image_rows,
                num_storm_image_columns=num_storm_image_columns,
                storm_grid_spacing_metres=storm_grid_spacing_metres)
        )

    if for_azimuthal_shear:
        argument_dict = {
            ROTATED_SHEAR_LATITUDES_COLUMN: list_of_latitude_matrices,
            ROTATED_SHEAR_LONGITUDES_COLUMN: list_of_longitude_matrices,
        }
    else:
        argument_dict = {
            ROTATED_NON_SHEAR_LATITUDES_COLUMN: list_of_latitude_matrices,
            ROTATED_NON_SHEAR_LONGITUDES_COLUMN: list_of_longitude_matrices,
        }

    return storm_object_table.assign(**argument_dict)


def _centroids_latlng_to_rowcol(
        centroid_latitudes_deg, centroid_longitudes_deg, nw_grid_point_lat_deg,
        nw_grid_point_lng_deg, lat_spacing_deg, lng_spacing_deg):
    """Converts storm centroids from lat-long to row-column coordinates.

    L = number of storm objects

    :param centroid_latitudes_deg: length-L numpy array with latitudes (deg N)
        of storm centroids.
    :param centroid_longitudes_deg: length-L numpy array with longitudes (deg E)
        of storm centroids.
    :param nw_grid_point_lat_deg: Latitude (deg N) of northwesternmost grid
        point.
    :param nw_grid_point_lng_deg: Longitude (deg E) of northwesternmost grid
        point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :return: centroid_rows: length-L numpy array with row indices (half-
        integers) of storm centroids.
    :return: centroid_columns: length-L numpy array with column indices (half-
        integers) of storm centroids.
    """

    center_row_indices, center_column_indices = radar_utils.latlng_to_rowcol(
        latitudes_deg=centroid_latitudes_deg,
        longitudes_deg=centroid_longitudes_deg,
        nw_grid_point_lat_deg=nw_grid_point_lat_deg,
        nw_grid_point_lng_deg=nw_grid_point_lng_deg,
        lat_spacing_deg=lat_spacing_deg, lng_spacing_deg=lng_spacing_deg)

    return (rounder.round_to_half_integer(center_row_indices),
            rounder.round_to_half_integer(center_column_indices))


def _get_unrotated_storm_image_coords(
        num_full_grid_rows, num_full_grid_columns, num_storm_image_rows,
        num_storm_image_columns, center_row, center_column):
    """Generates row-column coordinates for storm-centered grid.

    :param num_full_grid_rows: Number of rows in full grid.
    :param num_full_grid_columns: Number of columns in full grid.
    :param num_storm_image_rows: Number of rows in storm-centered image
        (subgrid).
    :param num_storm_image_columns: Number of columns in subgrid.
    :param center_row: Row index (half-integer) at center of subgrid.  If
        `center_row = k`, row k in the full grid is at the center of the
        subgrid.
    :param center_column: Column index (half-integer) at center of subgrid.  If
        `center_column = m`, column m in the full grid is at the center of the
        subgrid.
    :return: coord_dict: Dictionary with the following keys.
    coord_dict['first_storm_image_row']: First row in subgrid.  If
        `first_storm_image_row = i`, row 0 in the subgrid = row i in the full
        grid.
    coord_dict['last_storm_image_row']: Last row in subgrid.
    coord_dict['first_storm_image_column']: First column in subgrid.  If
        `first_storm_image_column = j`, column 0 in the subgrid = column j in
        the full grid.
    coord_dict['last_storm_image_column']: Last column in subgrid.
    coord_dict['num_padding_rows_at_top']: Number of padding rows at top of
        subgrid.  This will be non-zero iff the subgrid does not fit inside the
        full grid.
    coord_dict['num_padding_rows_at_bottom']: Number of padding rows at bottom
        of subgrid.
    coord_dict['num_padding_rows_at_left']: Number of padding rows at left side
        of subgrid.
    coord_dict['num_padding_rows_at_right']: Number of padding rows at right
        side of subgrid.
    """

    first_storm_image_row = int(numpy.ceil(
        center_row - num_storm_image_rows / 2
    ))
    last_storm_image_row = int(numpy.floor(
        center_row + num_storm_image_rows / 2
    ))
    first_storm_image_column = int(numpy.ceil(
        center_column - num_storm_image_columns / 2
    ))
    last_storm_image_column = int(numpy.floor(
        center_column + num_storm_image_columns / 2
    ))

    if first_storm_image_row < 0:
        num_padding_rows_at_top = 0 - first_storm_image_row
        first_storm_image_row = 0
    else:
        num_padding_rows_at_top = 0

    if last_storm_image_row > num_full_grid_rows - 1:
        num_padding_rows_at_bottom = last_storm_image_row - (
            num_full_grid_rows - 1)
        last_storm_image_row = num_full_grid_rows - 1
    else:
        num_padding_rows_at_bottom = 0

    if first_storm_image_column < 0:
        num_padding_columns_at_left = 0 - first_storm_image_column
        first_storm_image_column = 0
    else:
        num_padding_columns_at_left = 0

    if last_storm_image_column > num_full_grid_columns - 1:
        num_padding_columns_at_right = last_storm_image_column - (
            num_full_grid_columns - 1)
        last_storm_image_column = num_full_grid_columns - 1
    else:
        num_padding_columns_at_right = 0

    return {
        FIRST_STORM_ROW_KEY: first_storm_image_row,
        LAST_STORM_ROW_KEY: last_storm_image_row,
        FIRST_STORM_COLUMN_KEY: first_storm_image_column,
        LAST_STORM_COLUMN_KEY: last_storm_image_column,
        NUM_TOP_PADDING_ROWS_KEY: num_padding_rows_at_top,
        NUM_BOTTOM_PADDING_ROWS_KEY: num_padding_rows_at_bottom,
        NUM_LEFT_PADDING_COLS_KEY: num_padding_columns_at_left,
        NUM_RIGHT_PADDING_COLS_KEY: num_padding_columns_at_right
    }


def _subset_xy_grid_for_interp(
        field_matrix, grid_point_x_coords_metres, grid_point_y_coords_metres,
        query_x_coords_metres, query_y_coords_metres):
    """Subsets x-y grid before interpolation.

    Interpolation will be done from the x-y grid to the query points.

    M = number of rows in original grid
    N = number of columns in original grid
    m = number of rows in subset grid
    n = number of columns in subset grid

    :param field_matrix: M-by-N numpy array of gridded values, which will be
        interpolated to query points.
    :param grid_point_x_coords_metres: length-N numpy array of x-coordinates at
        grid points.  Assumed to be sorted in ascending order.
    :param grid_point_y_coords_metres: length-M numpy array of y-coordinates at
        grid points.  Assumed to be sorted in ascending order.
    :param query_x_coords_metres: numpy array (any dimensions) with
        x-coordinates of query points.
    :param query_y_coords_metres: numpy array (equivalent shape to
        `query_x_coords_metres`) with y-coordinates of query points.
    :return: subset_field_matrix: m-by-n numpy array of gridded values.
    :return: subset_gp_x_coords_metres: length-n numpy array of x-coordinates at
        grid points.
    :return: subset_gp_y_coords_metres: length-m numpy array of y-coordinates at
        grid points.
    """

    valid_x_indices = numpy.where(numpy.logical_and(
        grid_point_x_coords_metres >= numpy.min(query_x_coords_metres),
        grid_point_x_coords_metres <= numpy.max(query_x_coords_metres)))[0]
    first_valid_x_index = max([valid_x_indices[0] - 2, 0])
    last_valid_x_index = min([
        valid_x_indices[-1] + 2,
        len(grid_point_x_coords_metres) - 1
    ])

    valid_y_indices = numpy.where(numpy.logical_and(
        grid_point_y_coords_metres >= numpy.min(query_y_coords_metres),
        grid_point_y_coords_metres <= numpy.max(query_y_coords_metres)))[0]
    first_valid_y_index = max([valid_y_indices[0] - 2, 0])
    last_valid_y_index = min([
        valid_y_indices[-1] + 2,
        len(grid_point_y_coords_metres) - 1
    ])

    subset_field_matrix = field_matrix[
        first_valid_y_index:(last_valid_y_index + 1),
        first_valid_x_index:(last_valid_x_index + 1)
    ]

    return (
        subset_field_matrix,
        grid_point_x_coords_metres[
            first_valid_x_index:(last_valid_x_index + 1)
        ],
        grid_point_y_coords_metres[
            first_valid_y_index:(last_valid_y_index + 1)
        ]
    )


def _extract_rotated_storm_image(
        full_radar_matrix, full_grid_point_latitudes_deg,
        full_grid_point_longitudes_deg, rotated_gp_lat_matrix_deg,
        rotated_gp_lng_matrix_deg):
    """Extracts rotated, storm-centered image from full radar image.

    M = number of rows in full grid
    N = number of columns in full grid
    m = number of rows in storm-centered grid
    n = number of columns in storm-centered grid

    :param full_radar_matrix: M-by-N numpy array of radar values (one variable
        at one height and one time step).  Latitude should increase with row
        index, and longitude should increase with column index.
    :param full_grid_point_latitudes_deg: length-M numpy array with latitudes
        (deg N) of grid points.
    :param full_grid_point_longitudes_deg: length-N numpy array with longitudes
        (deg E) of grid points.
    :param rotated_gp_lat_matrix_deg: m-by-n numpy array with latitudes (deg N)
        of grid points.
    :param rotated_gp_lng_matrix_deg: m-by-n numpy array with longitudes (deg E)
        of grid points.
    :return: storm_centered_radar_matrix: m-by-n numpy array of radar values
        (same variable, height, and time step).
    """

    central_latitude_deg = numpy.mean(rotated_gp_lat_matrix_deg)
    central_longitude_deg = numpy.mean(rotated_gp_lng_matrix_deg)

    projection_object = projections.init_cylindrical_equidistant_projection(
        central_latitude_deg=central_latitude_deg,
        central_longitude_deg=central_longitude_deg,
        true_scale_latitude_deg=central_latitude_deg)

    (rotated_gp_x_matrix_metres, rotated_gp_y_matrix_metres
    ) = projections.project_latlng_to_xy(
        latitudes_deg=rotated_gp_lat_matrix_deg,
        longitudes_deg=rotated_gp_lng_matrix_deg,
        projection_object=projection_object)

    full_grid_points_x_metres, _ = projections.project_latlng_to_xy(
        latitudes_deg=numpy.full(
            full_grid_point_longitudes_deg.shape, central_latitude_deg),
        longitudes_deg=full_grid_point_longitudes_deg,
        projection_object=projection_object)

    _, full_grid_points_y_metres = projections.project_latlng_to_xy(
        latitudes_deg=full_grid_point_latitudes_deg,
        longitudes_deg=numpy.full(
            full_grid_point_latitudes_deg.shape, central_longitude_deg),
        projection_object=projection_object)

    (full_radar_matrix, full_grid_points_x_metres, full_grid_points_y_metres
    ) = _subset_xy_grid_for_interp(
        field_matrix=full_radar_matrix,
        grid_point_x_coords_metres=full_grid_points_x_metres,
        grid_point_y_coords_metres=full_grid_points_y_metres,
        query_x_coords_metres=rotated_gp_x_matrix_metres,
        query_y_coords_metres=rotated_gp_y_matrix_metres)

    storm_centered_radar_matrix = interp.interp_from_xy_grid_to_points(
        input_matrix=full_radar_matrix,
        sorted_grid_point_x_metres=full_grid_points_x_metres,
        sorted_grid_point_y_metres=full_grid_points_y_metres,
        query_x_coords_metres=rotated_gp_x_matrix_metres.ravel(),
        query_y_coords_metres=rotated_gp_y_matrix_metres.ravel(),
        method_string=interp.SPLINE_METHOD_STRING, spline_degree=1,
        extrapolate=True)
    storm_centered_radar_matrix = numpy.reshape(
        storm_centered_radar_matrix, rotated_gp_lat_matrix_deg.shape)

    invalid_x_flags = numpy.logical_or(
        rotated_gp_x_matrix_metres < numpy.min(full_grid_points_x_metres),
        rotated_gp_x_matrix_metres > numpy.max(full_grid_points_x_metres))
    invalid_y_flags = numpy.logical_or(
        rotated_gp_y_matrix_metres < numpy.min(full_grid_points_y_metres),
        rotated_gp_y_matrix_metres > numpy.max(full_grid_points_y_metres))
    invalid_indices = numpy.where(
        numpy.logical_or(invalid_x_flags, invalid_y_flags))

    storm_centered_radar_matrix[invalid_indices] = PADDING_VALUE
    return numpy.flipud(storm_centered_radar_matrix)


def _extract_unrotated_storm_image(
        full_radar_matrix, center_row, center_column, num_storm_image_rows,
        num_storm_image_columns):
    """Extracts storm-centered image from full radar image.

    M = number of rows in full grid
    N = number of columns in full grid
    m = number of rows in subgrid (must be even)
    n = number of columns in subgrid (must be even)

    The subgrid is rotated so that storm motion is in the +x-direction.

    :param full_radar_matrix: M-by-N numpy array of radar values (one variable
        at one height and one time step).
    :param center_row: Row index (half-integer) at center of subgrid.  If
        `center_row = i`, row i in the full grid is at the center of the
        subgrid.
    :param center_column: Column index (half-integer) at center of subgrid.  If
        `center_column = j`, column j in the full grid is at the center of the
        subgrid.
    :param num_storm_image_rows: m in the above discussion.
    :param num_storm_image_columns: n in the above discussion.
    :return: storm_centered_radar_matrix: m-by-n numpy array of radar values
        (same variable, height, and time step).
    """

    num_full_grid_rows = full_radar_matrix.shape[0]
    num_full_grid_columns = full_radar_matrix.shape[1]

    error_checking.assert_is_geq(center_row, -0.5)
    error_checking.assert_is_leq(center_row, num_full_grid_rows - 0.5)
    error_checking.assert_is_geq(center_column, -0.5)
    error_checking.assert_is_leq(center_column, num_full_grid_columns - 0.5)

    storm_image_coord_dict = _get_unrotated_storm_image_coords(
        num_full_grid_rows=num_full_grid_rows,
        num_full_grid_columns=num_full_grid_columns,
        num_storm_image_rows=num_storm_image_rows,
        num_storm_image_columns=num_storm_image_columns, center_row=center_row,
        center_column=center_column)

    storm_image_rows = numpy.linspace(
        storm_image_coord_dict[FIRST_STORM_ROW_KEY],
        storm_image_coord_dict[LAST_STORM_ROW_KEY],
        num=(storm_image_coord_dict[LAST_STORM_ROW_KEY] -
             storm_image_coord_dict[FIRST_STORM_ROW_KEY] + 1),
        dtype=int)
    storm_image_columns = numpy.linspace(
        storm_image_coord_dict[FIRST_STORM_COLUMN_KEY],
        storm_image_coord_dict[LAST_STORM_COLUMN_KEY],
        num=(storm_image_coord_dict[LAST_STORM_COLUMN_KEY] -
             storm_image_coord_dict[FIRST_STORM_COLUMN_KEY] + 1),
        dtype=int)

    storm_centered_radar_matrix = numpy.take(
        full_radar_matrix, storm_image_rows, axis=0)
    storm_centered_radar_matrix = numpy.take(
        storm_centered_radar_matrix, storm_image_columns, axis=1)

    pad_width_input_arg = (
        (storm_image_coord_dict[NUM_TOP_PADDING_ROWS_KEY],
         storm_image_coord_dict[NUM_BOTTOM_PADDING_ROWS_KEY]),
        (storm_image_coord_dict[NUM_LEFT_PADDING_COLS_KEY],
         storm_image_coord_dict[NUM_RIGHT_PADDING_COLS_KEY]))

    return numpy.pad(
        storm_centered_radar_matrix, pad_width=pad_width_input_arg,
        mode='constant', constant_values=PADDING_VALUE)


def _interp_storm_image_in_height(
        storm_image_matrix_3d, orig_heights_m_asl, new_heights_m_asl):
    """Interpolates 3-D storm-centered radar image to new heights.

    M = number of grid rows
    N = number of grid columns
    h = number of heights in original grid
    H = number of heights in new grid

    :param storm_image_matrix_3d: M-by-N-by-h numpy array of storm-centered
        radar measurements.
    :param orig_heights_m_asl: length-h numpy array of heights (metres above sea
        level).
    :param new_heights_m_asl: length-H numpy array of heights (metres above sea
        level).  Will interpolate from `orig_heights_m_asl` to these heights.
    :return: storm_image_matrix_3d: M-by-N-by-H numpy array of storm-centered
        radar measurements.
    """

    interp_object = scipy_interp1d(
        x=orig_heights_m_asl.astype(float), y=storm_image_matrix_3d,
        kind='linear', axis=-1, bounds_error=False, fill_value='extrapolate',
        assume_sorted=True)

    return interp_object(new_heights_m_asl.astype(float))


def _find_many_files_one_spc_date(
        top_directory_name, start_time_unix_sec, end_time_unix_sec,
        spc_date_string, radar_source, field_name_by_pair, height_by_pair_m_agl,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False):
    """Finds files with storm-centered MYRORSS or MRMS images for one SPC date.

    Each file should contain one radar field/height at one time step.

    T = number of time steps
    C = number of field/height pairs

    :param top_directory_name: See doc for `find_many_files_myrorss_or_mrms`.
    :param start_time_unix_sec: Start time.  This method will find all files
        from `start_time_unix_sec`...`end_time_unix_sec` that fit into the given
        SPC date.
    :param end_time_unix_sec: See above.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param radar_source: Source (either "myrorss" or "mrms").
    :param field_name_by_pair: length-C list with names of radar fields.
    :param height_by_pair_m_agl: length-C numpy array of radar heights (metres
        above ground level).
    :param raise_error_if_all_missing: See doc for
        `find_many_files_myrorss_or_mrms`.
    :param raise_error_if_any_missing: Same.
    :return: image_file_name_matrix: T-by-C numpy array of paths to files found.
    :return: valid_times_unix_sec: length-T numpy array of storm times.
    """

    height_by_pair_m_agl = numpy.round(height_by_pair_m_agl).astype(int)
    azimuthal_shear_flags = numpy.array(
        [this_field_name in AZIMUTHAL_SHEAR_FIELD_NAMES
         for this_field_name in field_name_by_pair])
    any_az_shear_fields = numpy.any(azimuthal_shear_flags)

    glob_index = -1
    num_field_height_pairs = len(field_name_by_pair)
    image_file_name_matrix = None
    valid_times_unix_sec = None

    for j in range(num_field_height_pairs):
        glob_now = (field_name_by_pair[j] in AZIMUTHAL_SHEAR_FIELD_NAMES
                    or not any_az_shear_fields)
        if not glob_now:
            continue

        glob_index = j + 0
        this_file_pattern = (
            '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}/{5:05d}_metres_agl/'
            'storm_images_{6:s}.nc'
        ).format(
            top_directory_name, radar_source, spc_date_string[:4],
            spc_date_string, field_name_by_pair[j], height_by_pair_m_agl[j],
            TIME_FORMAT_REGEX)

        all_file_names = glob.glob(this_file_pattern)
        image_file_names = []
        valid_times_unix_sec = []

        for this_file_name in all_file_names:
            this_pathless_file_name = os.path.split(this_file_name)[-1]
            this_extensionless_file_name = os.path.splitext(
                this_pathless_file_name)[0]
            this_time_string = this_extensionless_file_name.split('_')[-1]
            this_time_unix_sec = time_conversion.string_to_unix_sec(
                this_time_string, TIME_FORMAT)

            if start_time_unix_sec <= this_time_unix_sec <= end_time_unix_sec:
                image_file_names.append(this_file_name)
                valid_times_unix_sec.append(this_time_unix_sec)

        num_times = len(image_file_names)

        if num_times == 0:
            if raise_error_if_all_missing:
                error_string = (
                    'Cannot find files with "{0:s}" at {1:d} metres AGL on SPC '
                    'date "{2:s}".'
                ).format(
                    field_name_by_pair[j], height_by_pair_m_agl[j],
                    spc_date_string
                )

                raise ValueError(error_string)

            return None, None

        valid_times_unix_sec = numpy.array(valid_times_unix_sec, dtype=int)
        image_file_name_matrix = numpy.full(
            (num_times, num_field_height_pairs), '', dtype=object)
        image_file_name_matrix[:, j] = numpy.array(
            image_file_names, dtype=object)

        break

    for j in range(num_field_height_pairs):
        if j == glob_index:
            continue

        for i in range(len(valid_times_unix_sec)):
            image_file_name_matrix[i, j] = find_storm_image_file(
                top_directory_name=top_directory_name,
                unix_time_sec=valid_times_unix_sec[i],
                spc_date_string=spc_date_string, radar_source=radar_source,
                radar_field_name=field_name_by_pair[j],
                radar_height_m_agl=height_by_pair_m_agl[j],
                raise_error_if_missing=raise_error_if_any_missing)

            if not os.path.isfile(image_file_name_matrix[i, j]):
                image_file_name_matrix[i, j] = ''

    return image_file_name_matrix, valid_times_unix_sec


def downsize_storm_images(
        storm_image_matrix, radar_field_name, num_rows_to_keep=None,
        num_columns_to_keep=None):
    """Downsizes storm-centered radar images.

    :param storm_image_matrix: See doc for `_check_storm_images`.
    :param radar_field_name: Same.
    :param num_rows_to_keep: Number of rows to keep.  If `storm_image_matrix`
        contains azimuthal shear, this will be doubled.
    :param num_columns_to_keep: Number of columns to keep.  If
        `storm_image_matrix` contains azimuthal shear, this will be doubled.
    :return: storm_image_matrix: Same as input, but maybe with fewer rows and
        columns.
    :raises: ValueError: if downsized image cannot be centered in full image.
    """

    error_checking.assert_is_numpy_array_without_nan(storm_image_matrix)
    num_dimensions = len(storm_image_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 3)

    radar_utils.check_field_name(radar_field_name)
    if radar_field_name in AZIMUTHAL_SHEAR_FIELD_NAMES:
        if num_rows_to_keep is not None:
            num_rows_to_keep *= 2
        if num_columns_to_keep is not None:
            num_columns_to_keep *= 2

    num_rows_total = storm_image_matrix.shape[1]
    if num_rows_to_keep == num_rows_total:
        num_rows_to_keep = None

    num_columns_total = storm_image_matrix.shape[2]
    if num_columns_to_keep == num_columns_total:
        num_columns_to_keep = None

    if num_rows_to_keep is not None:
        error_checking.assert_is_integer(num_rows_to_keep)
        error_checking.assert_is_greater(num_rows_to_keep, 0)

        num_rows_leftover = num_rows_total - num_rows_to_keep
        if num_rows_leftover != rounder.round_to_nearest(num_rows_leftover, 2):
            error_string = (
                'Cannot downsize from {0:d} to {1:d} rows, because number of '
                'rows left over ({2:d}) is odd.'
            ).format(num_rows_total, num_rows_to_keep, num_rows_leftover)
            raise ValueError(error_string)

        first_row_to_keep = num_rows_leftover // 2
        last_row_to_keep = first_row_to_keep + num_rows_to_keep - 1
        storm_image_matrix = storm_image_matrix[
            :, first_row_to_keep:(last_row_to_keep + 1), ...
        ]

    if num_columns_to_keep is not None:
        error_checking.assert_is_integer(num_columns_to_keep)
        error_checking.assert_is_greater(num_columns_to_keep, 0)

        num_columns_leftover = num_columns_total - num_columns_to_keep
        if (num_columns_leftover !=
                rounder.round_to_nearest(num_columns_leftover, 2)):
            error_string = (
                'Cannot downsize from {0:d} to {1:d} columns, because number of'
                ' columns left over ({2:d}) is odd.'
            ).format(
                num_columns_total, num_columns_to_keep, num_columns_leftover
            )
            raise ValueError(error_string)

        first_column_to_keep = num_columns_leftover // 2
        last_column_to_keep = first_column_to_keep + num_columns_to_keep - 1
        storm_image_matrix = storm_image_matrix[
            :, :, first_column_to_keep:(last_column_to_keep + 1), ...
        ]

    return storm_image_matrix


def extract_storm_images_myrorss_or_mrms(
        storm_object_table, radar_source, top_radar_dir_name,
        top_output_dir_name, elevation_dir_name,
        num_storm_image_rows=DEFAULT_NUM_IMAGE_ROWS,
        num_storm_image_columns=DEFAULT_NUM_IMAGE_COLUMNS, rotate_grids=True,
        rotated_grid_spacing_metres=DEFAULT_ROTATED_GRID_SPACING_METRES,
        radar_field_names=DEFAULT_MYRORSS_MRMS_FIELD_NAMES,
        reflectivity_heights_m_agl=DEFAULT_RADAR_HEIGHTS_M_AGL):
    """Extracts storm-centered image for each field/height and storm object.

    L = number of storm objects
    C = number of field/height pairs
    T = number of unique storm times

    :param storm_object_table: L-row pandas DataFrame with the following
        columns.
    storm_object_table.full_id_string: Full storm ID.
    storm_object_table.valid_time_unix_sec: Valid time.
    storm_object_table.spc_date_string: SPC date (format "yyyymmdd").
    storm_object_table.centroid_latitude_deg: Latitude (deg N) of storm
        centroid.
    storm_object_table.centroid_longitude_deg: Longitude (deg E) of storm
        centroid.

    If `rotate_grids = True`, also need the following columns.

    storm_object_table.east_velocity_m_s01: Eastward storm velocity (metres per
        second).
    storm_object_table.north_velocity_m_s01: Northward storm velocity.
    :param radar_source: Data source (either "myrorss" or "mrms").
    :param top_radar_dir_name: Name of top-level input directory, containing
        radar data from the given source.
    :param top_output_dir_name: Name of top-level output directory, to which
        storm-centered images will be written.
    :param elevation_dir_name: Name of directory where elevations are stored
        (by the Python package "srtm").
    :param num_storm_image_rows: See doc for `_check_extraction_args`.
    :param num_storm_image_columns: Same.
    :param rotate_grids: Same.
    :param rotated_grid_spacing_metres: Same.
    :param radar_field_names: Same.
    :param reflectivity_heights_m_agl: Same.
    """

    _check_extraction_args(
        num_storm_image_rows=num_storm_image_rows,
        num_storm_image_columns=num_storm_image_columns,
        rotate_grids=rotate_grids,
        rotated_grid_spacing_metres=rotated_grid_spacing_metres,
        radar_field_names=radar_field_names, radar_source=radar_source,
        reflectivity_heights_m_agl=reflectivity_heights_m_agl)

    reflectivity_heights_m_agl = numpy.round(
        reflectivity_heights_m_agl
    ).astype(int)

    # Find radar heights needed, in metres above sea level.
    if radar_utils.REFL_NAME in radar_field_names:
        these_elevations_m_asl = numpy.array([
            LOWEST_POINT_IN_CONUS_M_ASL, HIGHEST_POINT_IN_CONUS_M_ASL
        ])

        reflectivity_heights_m_asl = _find_input_heights_needed(
            storm_elevations_m_asl=these_elevations_m_asl,
            desired_radar_heights_m_agl=reflectivity_heights_m_agl,
            radar_source=radar_source)
    else:
        reflectivity_heights_m_agl = numpy.array([], dtype=int)
        reflectivity_heights_m_asl = numpy.array([], dtype=int)

    num_refl_heights_agl = len(reflectivity_heights_m_agl)
    num_refl_heights_asl = len(reflectivity_heights_m_asl)

    # Find input files.
    valid_spc_date_strings = (
        storm_object_table[tracking_utils.SPC_DATE_COLUMN].values.tolist()
    )

    input_file_dict = myrorss_and_mrms_io.find_many_raw_files(
        desired_times_unix_sec=
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values.astype(int),
        spc_date_strings=valid_spc_date_strings, data_source=radar_source,
        field_names=radar_field_names, top_directory_name=top_radar_dir_name,
        reflectivity_heights_m_asl=reflectivity_heights_m_asl)

    radar_file_name_matrix = input_file_dict[
        myrorss_and_mrms_io.RADAR_FILE_NAMES_KEY]
    valid_times_unix_sec = input_file_dict[myrorss_and_mrms_io.UNIQUE_TIMES_KEY]
    valid_spc_dates_unix_sec = input_file_dict[
        myrorss_and_mrms_io.SPC_DATES_AT_UNIQUE_TIMES_KEY]
    field_name_by_pair = input_file_dict[
        myrorss_and_mrms_io.FIELD_NAME_BY_PAIR_KEY]
    height_by_pair_m_asl = input_file_dict[
        myrorss_and_mrms_io.HEIGHT_BY_PAIR_KEY]

    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in valid_times_unix_sec
    ]

    any_azimuthal_shear = [
        f in AZIMUTHAL_SHEAR_FIELD_NAMES for f in field_name_by_pair
    ]
    any_non_azimuthal_shear = [
        f not in AZIMUTHAL_SHEAR_FIELD_NAMES for f in field_name_by_pair
    ]

    num_times = len(valid_time_strings)
    num_field_height_pairs = len(field_name_by_pair)
    latitude_spacing_deg = None
    longitude_spacing_deg = None

    for i in range(num_times):
        these_storm_indices = _get_relevant_storm_objects(
            storm_object_table=storm_object_table,
            valid_time_unix_sec=valid_times_unix_sec[i],
            valid_spc_date_string=valid_spc_date_strings[i])

        this_storm_object_table = storm_object_table.iloc[these_storm_indices]

        print('Finding storm elevations at {0:s}...'.format(
            valid_time_strings[i]
        ))

        these_elevations_m_asl = geodetic_utils.get_elevations(
            latitudes_deg=this_storm_object_table[
                tracking_utils.CENTROID_LATITUDE_COLUMN].values,
            longitudes_deg=this_storm_object_table[
                tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
            working_dir_name=elevation_dir_name)

        this_storm_object_table = this_storm_object_table.assign(**{
            ELEVATION_COLUMN: these_elevations_m_asl
        })

        if rotate_grids:
            if any_azimuthal_shear:
                print((
                    'Creating rotated {0:.1f}-metre grids for storms at '
                    '{1:s}...'
                ).format(
                    rotated_grid_spacing_metres / 2, valid_time_strings[i]
                ))

                this_storm_object_table = _rotate_grids_many_storm_objects(
                    storm_object_table=this_storm_object_table,
                    num_storm_image_rows=num_storm_image_rows * 2,
                    num_storm_image_columns=num_storm_image_columns * 2,
                    storm_grid_spacing_metres=rotated_grid_spacing_metres / 2,
                    for_azimuthal_shear=True)

            if any_non_azimuthal_shear:
                print((
                    'Creating rotated {0:.1f}-metre grids for storms at '
                    '{1:s}...'
                ).format(
                    rotated_grid_spacing_metres, valid_time_strings[i]
                ))

                this_storm_object_table = _rotate_grids_many_storm_objects(
                    storm_object_table=this_storm_object_table,
                    num_storm_image_rows=num_storm_image_rows,
                    num_storm_image_columns=num_storm_image_columns,
                    storm_grid_spacing_metres=rotated_grid_spacing_metres,
                    for_azimuthal_shear=False)

        this_num_storms = len(this_storm_object_table.index)
        this_refl_matrix_sea_relative_dbz = numpy.full(
            (this_num_storms, num_storm_image_rows, num_storm_image_columns,
             num_refl_heights_asl),
            numpy.nan
        )

        for j in range(num_field_height_pairs):
            print((
                'Extracting storm-centered images for "{0:s}" at {1:d} '
                'metres ASL and {2:s}...'
            ).format(
                field_name_by_pair[j],
                int(numpy.round(height_by_pair_m_asl[j])),
                valid_time_strings[i]
            ))

            this_metadata_dict = (
                myrorss_and_mrms_io.read_metadata_from_raw_file(
                    netcdf_file_name=radar_file_name_matrix[i, j],
                    data_source=radar_source)
            )

            this_sparse_grid_table = (
                myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                    netcdf_file_name=radar_file_name_matrix[i, j],
                    field_name_orig=this_metadata_dict[
                        myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                    data_source=radar_source,
                    sentinel_values=this_metadata_dict[
                        radar_utils.SENTINEL_VALUE_COLUMN]
                )
            )

            (this_full_radar_matrix, these_full_latitudes_deg,
             these_full_longitudes_deg
            ) = radar_s2f.sparse_to_full_grid(
                sparse_grid_table=this_sparse_grid_table,
                metadata_dict=this_metadata_dict)

            this_full_radar_matrix[
                numpy.isnan(this_full_radar_matrix)
            ] = PADDING_VALUE

            if rotate_grids:
                this_full_radar_matrix = numpy.flipud(this_full_radar_matrix)
                these_full_latitudes_deg = these_full_latitudes_deg[::-1]

            latitude_spacing_deg, longitude_spacing_deg = _check_grid_spacing(
                new_metadata_dict=this_metadata_dict,
                orig_lat_spacing_deg=latitude_spacing_deg,
                orig_lng_spacing_deg=longitude_spacing_deg)

            if field_name_by_pair[j] in AZIMUTHAL_SHEAR_FIELD_NAMES:
                this_num_image_rows = (
                    num_storm_image_rows * AZ_SHEAR_GRID_SPACING_MULTIPLIER
                )
                this_num_image_columns = (
                    num_storm_image_columns * AZ_SHEAR_GRID_SPACING_MULTIPLIER
                )
            else:
                this_num_image_rows = num_storm_image_rows + 0
                this_num_image_columns = num_storm_image_columns + 0

            this_storm_image_matrix = numpy.full(
                (this_num_storms, this_num_image_rows, this_num_image_columns),
                numpy.nan
            )

            if rotate_grids:
                for k in range(this_num_storms):
                    if field_name_by_pair[j] in AZIMUTHAL_SHEAR_FIELD_NAMES:
                        this_rotated_lat_matrix_deg = this_storm_object_table[
                            ROTATED_SHEAR_LATITUDES_COLUMN].values[k]
                        this_rotated_lng_matrix_deg = this_storm_object_table[
                            ROTATED_SHEAR_LONGITUDES_COLUMN].values[k]
                    else:
                        this_rotated_lat_matrix_deg = this_storm_object_table[
                            ROTATED_NON_SHEAR_LATITUDES_COLUMN].values[k]
                        this_rotated_lng_matrix_deg = this_storm_object_table[
                            ROTATED_NON_SHEAR_LONGITUDES_COLUMN].values[k]

                    this_storm_image_matrix[
                        k, :, :
                    ] = _extract_rotated_storm_image(
                        full_radar_matrix=this_full_radar_matrix,
                        full_grid_point_latitudes_deg=these_full_latitudes_deg,
                        full_grid_point_longitudes_deg=
                        these_full_longitudes_deg,
                        rotated_gp_lat_matrix_deg=this_rotated_lat_matrix_deg,
                        rotated_gp_lng_matrix_deg=this_rotated_lng_matrix_deg)
            else:
                these_center_rows, these_center_columns = (
                    _centroids_latlng_to_rowcol(
                        centroid_latitudes_deg=this_storm_object_table[
                            tracking_utils.CENTROID_LATITUDE_COLUMN].values,
                        centroid_longitudes_deg=this_storm_object_table[
                            tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
                        nw_grid_point_lat_deg=this_metadata_dict[
                            radar_utils.NW_GRID_POINT_LAT_COLUMN],
                        nw_grid_point_lng_deg=this_metadata_dict[
                            radar_utils.NW_GRID_POINT_LNG_COLUMN],
                        lat_spacing_deg=this_metadata_dict[
                            radar_utils.LAT_SPACING_COLUMN],
                        lng_spacing_deg=this_metadata_dict[
                            radar_utils.LNG_SPACING_COLUMN]
                    )
                )

                for k in range(this_num_storms):
                    this_storm_image_matrix[k, :, :] = (
                        _extract_unrotated_storm_image(
                            full_radar_matrix=this_full_radar_matrix,
                            center_row=these_center_rows[k],
                            center_column=these_center_columns[k],
                            num_storm_image_rows=this_num_image_rows,
                            num_storm_image_columns=this_num_image_columns)
                    )

            if field_name_by_pair[j] == radar_utils.REFL_NAME:
                this_height_index = numpy.where(
                    height_by_pair_m_asl[j] == reflectivity_heights_m_asl
                )[0][0]

                this_refl_matrix_sea_relative_dbz[..., this_height_index] = (
                    this_storm_image_matrix
                )

                continue

            this_image_file_name = find_storm_image_file(
                top_directory_name=top_output_dir_name,
                unix_time_sec=valid_times_unix_sec[i],
                spc_date_string=time_conversion.time_to_spc_date_string(
                    valid_spc_dates_unix_sec[i]),
                radar_source=radar_source,
                radar_field_name=field_name_by_pair[j],
                radar_height_m_agl=height_by_pair_m_asl[j],
                raise_error_if_missing=False)

            print((
                'Writing storm-centered images to: "{0:s}"...'
            ).format(this_image_file_name))

            write_storm_images(
                netcdf_file_name=this_image_file_name,
                storm_image_matrix=this_storm_image_matrix,
                full_id_strings=this_storm_object_table[
                    tracking_utils.FULL_ID_COLUMN].values.tolist(),
                valid_times_unix_sec=this_storm_object_table[
                    tracking_utils.VALID_TIME_COLUMN].values.astype(int),
                radar_field_name=field_name_by_pair[j],
                radar_height_m_agl=height_by_pair_m_asl[j],
                rotated_grids=rotate_grids,
                rotated_grid_spacing_metres=rotated_grid_spacing_metres)

        if radar_utils.REFL_NAME in radar_field_names:
            print('Interpolating reflectivity to desired heights above ground '
                  'level...')

            this_refl_matrix_ground_relative_dbz = numpy.full(
                (this_num_storms, num_storm_image_rows, num_storm_image_columns,
                 num_refl_heights_agl),
                numpy.nan
            )

            for k in range(this_num_storms):
                these_heights_m_asl = (
                    this_storm_object_table[ELEVATION_COLUMN].values[k]
                    + reflectivity_heights_m_agl
                )

                this_refl_matrix_ground_relative_dbz[k, ...] = (
                    _interp_storm_image_in_height(
                        storm_image_matrix_3d=this_refl_matrix_sea_relative_dbz[
                            k, ...],
                        orig_heights_m_asl=reflectivity_heights_m_asl,
                        new_heights_m_asl=these_heights_m_asl)
                )

        for j in range(num_refl_heights_agl):
            this_image_file_name = find_storm_image_file(
                top_directory_name=top_output_dir_name,
                unix_time_sec=valid_times_unix_sec[i],
                spc_date_string=time_conversion.time_to_spc_date_string(
                    valid_spc_dates_unix_sec[i]),
                radar_source=radar_source,
                radar_field_name=radar_utils.REFL_NAME,
                radar_height_m_agl=reflectivity_heights_m_agl[j],
                raise_error_if_missing=False)

            print((
                'Writing storm-centered images to: "{0:s}"...'
            ).format(this_image_file_name))

            write_storm_images(
                netcdf_file_name=this_image_file_name,
                storm_image_matrix=this_refl_matrix_ground_relative_dbz[..., j],
                full_id_strings=this_storm_object_table[
                    tracking_utils.FULL_ID_COLUMN].values.tolist(),
                valid_times_unix_sec=this_storm_object_table[
                    tracking_utils.VALID_TIME_COLUMN].values.astype(int),
                radar_field_name=radar_utils.REFL_NAME,
                radar_height_m_agl=reflectivity_heights_m_agl[j],
                rotated_grids=rotate_grids,
                rotated_grid_spacing_metres=rotated_grid_spacing_metres)

        print('\n')


def extract_storm_images_gridrad(
        storm_object_table, top_radar_dir_name, top_output_dir_name,
        elevation_dir_name, num_storm_image_rows=DEFAULT_NUM_IMAGE_ROWS,
        num_storm_image_columns=DEFAULT_NUM_IMAGE_COLUMNS, rotate_grids=True,
        rotated_grid_spacing_metres=DEFAULT_ROTATED_GRID_SPACING_METRES,
        radar_field_names=DEFAULT_GRIDRAD_FIELD_NAMES,
        radar_heights_m_agl=DEFAULT_RADAR_HEIGHTS_M_AGL,
        new_version=False):
    """Extracts storm-centered image for each field, height, and storm object.

    L = number of storm objects
    F = number of radar fields
    H = number of radar heights
    T = number of unique storm times

    :param storm_object_table: See doc for
        `extract_storm_images_myrorss_or_mrms`.
    :param top_radar_dir_name: Same.
    :param top_output_dir_name: Same.
    :param elevation_dir_name: Same.
    :param num_storm_image_rows: Same.
    :param num_storm_image_columns: Same.
    :param rotate_grids: Same.
    :param rotated_grid_spacing_metres: Same.
    :param radar_field_names: length-F list with names of radar fields.
    :param radar_heights_m_agl: length-H numpy array of radar heights (metres
        above ground level).
    """

    _check_extraction_args(
        num_storm_image_rows=num_storm_image_rows,
        num_storm_image_columns=num_storm_image_columns,
        rotate_grids=rotate_grids,
        rotated_grid_spacing_metres=rotated_grid_spacing_metres,
        radar_field_names=radar_field_names,
        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
        radar_heights_m_agl=radar_heights_m_agl)

    radar_heights_m_agl = numpy.round(radar_heights_m_agl).astype(int)

    these_elevations_m_asl = numpy.array([
        LOWEST_POINT_IN_CONUS_M_ASL, HIGHEST_POINT_IN_CONUS_M_ASL
    ])

    radar_heights_m_asl = _find_input_heights_needed(
        storm_elevations_m_asl=these_elevations_m_asl,
        desired_radar_heights_m_agl=radar_heights_m_agl,
        radar_source=radar_utils.GRIDRAD_SOURCE_ID)

    num_heights_agl = len(radar_heights_m_agl)
    num_heights_asl = len(radar_heights_m_asl)

    valid_times_unix_sec = numpy.unique(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
    )
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in valid_times_unix_sec
    ]
    valid_spc_date_strings = [
        time_conversion.time_to_spc_date_string(t) for t in valid_times_unix_sec
    ]

    # Find input files.
    num_times = len(valid_times_unix_sec)
    radar_file_names = [None] * num_times

    for i in range(num_times):
        radar_file_names[i] = gridrad_io.find_file(
            unix_time_sec=valid_times_unix_sec[i],
            top_directory_name=top_radar_dir_name, raise_error_if_missing=True,new_version=new_version)

    num_times = len(valid_times_unix_sec)
    num_fields = len(radar_field_names)
    latitude_spacing_deg = None
    longitude_spacing_deg = None

    for i in range(num_times):
        this_metadata_dict = gridrad_io.read_metadata_from_full_grid_file(
            radar_file_names[i]
        )

        latitude_spacing_deg, longitude_spacing_deg = _check_grid_spacing(
            new_metadata_dict=this_metadata_dict,
            orig_lat_spacing_deg=latitude_spacing_deg,
            orig_lng_spacing_deg=longitude_spacing_deg)

        these_storm_indices = _get_relevant_storm_objects(
            storm_object_table=storm_object_table,
            valid_time_unix_sec=valid_times_unix_sec[i],
            valid_spc_date_string=valid_spc_date_strings[i])

        this_storm_object_table = storm_object_table.iloc[these_storm_indices]

        print('Finding storm elevations at {0:s}...'.format(
            valid_time_strings[i]
        ))

        these_elevations_m_asl = geodetic_utils.get_elevations(
            latitudes_deg=this_storm_object_table[
                tracking_utils.CENTROID_LATITUDE_COLUMN].values,
            longitudes_deg=this_storm_object_table[
                tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
            working_dir_name=elevation_dir_name)

        this_storm_object_table = this_storm_object_table.assign(**{
            ELEVATION_COLUMN: these_elevations_m_asl
        })

        if rotate_grids:
            print((
                'Creating rotated {0:.1f}-metre grids for storms at '
                '{1:s}...'
            ).format(
                rotated_grid_spacing_metres, valid_time_strings[i]
            ))

            this_storm_object_table = _rotate_grids_many_storm_objects(
                storm_object_table=this_storm_object_table,
                num_storm_image_rows=num_storm_image_rows,
                num_storm_image_columns=num_storm_image_columns,
                storm_grid_spacing_metres=rotated_grid_spacing_metres,
                for_azimuthal_shear=False)

        this_num_storms = len(this_storm_object_table.index)

        for j in range(num_fields):
            print('Reading "{0:s}" from file: "{1:s}"...'.format(
                radar_field_names[j], radar_file_names[i]
            ))

            (this_full_radar_matrix_3d, these_full_heights_m_asl,
             these_full_latitudes_deg, these_full_longitudes_deg
            ) = gridrad_io.read_field_from_full_grid_file(
                netcdf_file_name=radar_file_names[i],
                field_name=radar_field_names[j],
                metadata_dict=this_metadata_dict)

            this_full_radar_matrix_3d[
                numpy.isnan(this_full_radar_matrix_3d)
            ] = PADDING_VALUE

            if not rotate_grids:
                this_full_radar_matrix_3d = numpy.flip(
                    this_full_radar_matrix_3d, axis=1)
                these_full_latitudes_deg = these_full_latitudes_deg[::-1]

            this_storm_image_matrix_sea_relative = numpy.full(
                (this_num_storms, num_storm_image_rows, num_storm_image_columns,
                 num_heights_asl),
                numpy.nan
            )

            for k in range(num_heights_asl):
                this_height_index = numpy.where(
                    these_full_heights_m_asl == radar_heights_m_asl[k]
                )[0][0]

                this_full_radar_matrix_2d = this_full_radar_matrix_3d[
                    this_height_index, ...]

                print((
                    'Extracting storm-centered images for "{0:s}" at {1:d} '
                    'metres ASL and {2:s}...'
                ).format(
                    radar_field_names[j],
                    int(numpy.round(radar_heights_m_asl[k])),
                    valid_time_strings[i]
                ))

                if rotate_grids:
                    for m in range(this_num_storms):
                        this_storm_image_matrix_sea_relative[
                            m, ..., k
                        ] = _extract_rotated_storm_image(
                            full_radar_matrix=this_full_radar_matrix_2d,
                            full_grid_point_latitudes_deg=
                            these_full_latitudes_deg,
                            full_grid_point_longitudes_deg=
                            these_full_longitudes_deg,
                            rotated_gp_lat_matrix_deg=this_storm_object_table[
                                ROTATED_NON_SHEAR_LATITUDES_COLUMN].values[m],
                            rotated_gp_lng_matrix_deg=this_storm_object_table[
                                ROTATED_NON_SHEAR_LONGITUDES_COLUMN].values[m]
                        )
                else:
                    these_center_rows, these_center_columns = (
                        _centroids_latlng_to_rowcol(
                            centroid_latitudes_deg=this_storm_object_table[
                                tracking_utils.CENTROID_LATITUDE_COLUMN].values,
                            centroid_longitudes_deg=this_storm_object_table[
                                tracking_utils.CENTROID_LONGITUDE_COLUMN
                            ].values,
                            nw_grid_point_lat_deg=this_metadata_dict[
                                radar_utils.NW_GRID_POINT_LAT_COLUMN],
                            nw_grid_point_lng_deg=this_metadata_dict[
                                radar_utils.NW_GRID_POINT_LNG_COLUMN],
                            lat_spacing_deg=this_metadata_dict[
                                radar_utils.LAT_SPACING_COLUMN],
                            lng_spacing_deg=this_metadata_dict[
                                radar_utils.LNG_SPACING_COLUMN]
                        )
                    )

                    for m in range(this_num_storms):
                        this_storm_image_matrix_sea_relative[m, ..., k] = (
                            _extract_unrotated_storm_image(
                                full_radar_matrix=this_full_radar_matrix_2d,
                                center_row=these_center_rows[m],
                                center_column=these_center_columns[m],
                                num_storm_image_rows=num_storm_image_rows,
                                num_storm_image_columns=num_storm_image_columns)
                        )

            print((
                'Interpolating "{0:s}" to desired heights above ground level...'
            ).format(radar_field_names[j]))

            this_storm_image_matrix_ground_relative = numpy.full(
                (this_num_storms, num_storm_image_rows, num_storm_image_columns,
                 num_heights_agl),
                numpy.nan
            )

            for m in range(this_num_storms):
                these_heights_m_asl = (
                    this_storm_object_table[ELEVATION_COLUMN].values[m]
                    + radar_heights_m_agl
                )

                this_storm_image_matrix_ground_relative[m, ...] = (
                    _interp_storm_image_in_height(
                        storm_image_matrix_3d=
                        this_storm_image_matrix_sea_relative[m, ...],
                        orig_heights_m_asl=radar_heights_m_asl,
                        new_heights_m_asl=these_heights_m_asl)
                )

            for k in range(num_heights_agl):
                this_image_file_name = find_storm_image_file(
                    top_directory_name=top_output_dir_name,
                    unix_time_sec=valid_times_unix_sec[i],
                    spc_date_string=valid_spc_date_strings[i],
                    radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                    radar_field_name=radar_field_names[j],
                    radar_height_m_agl=radar_heights_m_agl[k],
                    raise_error_if_missing=False)

                print((
                    'Writing storm-centered images to: "{0:s}"...'
                ).format(this_image_file_name))

                write_storm_images(
                    netcdf_file_name=this_image_file_name,
                    storm_image_matrix=this_storm_image_matrix_ground_relative[
                        ..., k],
                    full_id_strings=this_storm_object_table[
                        tracking_utils.FULL_ID_COLUMN].values.tolist(),
                    valid_times_unix_sec=this_storm_object_table[
                        tracking_utils.VALID_TIME_COLUMN].values.astype(int),
                    radar_field_name=radar_field_names[j],
                    radar_height_m_agl=radar_heights_m_agl[k],
                    rotated_grids=rotate_grids,
                    rotated_grid_spacing_metres=rotated_grid_spacing_metres)

            print('\n')


def write_storm_images(
        netcdf_file_name, storm_image_matrix, full_id_strings,
        valid_times_unix_sec, radar_field_name, radar_height_m_agl,
        rotated_grids=False, rotated_grid_spacing_metres=None,
        num_storm_objects_per_chunk=1):
    """Writes storm-centered radar images to NetCDF file.

    This file will contain images for one radar field/height.

    :param netcdf_file_name: Path to output file.
    :param storm_image_matrix: See doc for `_check_storm_images`.
    :param full_id_strings: Same.
    :param valid_times_unix_sec: Same.
    :param radar_field_name: Same.
    :param radar_height_m_agl: Same.
    :param rotated_grids: Same.
    :param rotated_grid_spacing_metres: Same.
    :param num_storm_objects_per_chunk: Number of storm objects per NetCDF
        chunk.  To use default chunking, make this `None`.
    """

    _check_storm_images(
        storm_image_matrix=storm_image_matrix, full_id_strings=full_id_strings,
        valid_times_unix_sec=valid_times_unix_sec,
        radar_field_name=radar_field_name,
        radar_height_m_agl=radar_height_m_agl, rotated_grids=rotated_grids,
        rotated_grid_spacing_metres=rotated_grid_spacing_metres)

    if num_storm_objects_per_chunk is not None:
        error_checking.assert_is_integer(num_storm_objects_per_chunk)
        error_checking.assert_is_geq(num_storm_objects_per_chunk, 1)

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    netcdf_dataset = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    netcdf_dataset.setncattr(RADAR_FIELD_NAME_KEY, radar_field_name)
    netcdf_dataset.setncattr(RADAR_HEIGHT_KEY, radar_height_m_agl)
    netcdf_dataset.setncattr(ROTATED_GRIDS_KEY, int(rotated_grids))

    if rotated_grids:
        netcdf_dataset.setncattr(
            ROTATED_GRID_SPACING_KEY, rotated_grid_spacing_metres)

    num_storm_objects = storm_image_matrix.shape[0]
    num_id_characters = 1

    for i in range(num_storm_objects):
        num_id_characters = max([
            num_id_characters, len(full_id_strings[i])
        ])

    netcdf_dataset.createDimension(
        STORM_OBJECT_DIMENSION_KEY, num_storm_objects
    )
    netcdf_dataset.createDimension(
        ROW_DIMENSION_KEY, storm_image_matrix.shape[1]
    )
    netcdf_dataset.createDimension(
        COLUMN_DIMENSION_KEY, storm_image_matrix.shape[2]
    )
    netcdf_dataset.createDimension(CHARACTER_DIMENSION_KEY, num_id_characters)

    netcdf_dataset.createVariable(
        FULL_IDS_KEY, datatype='S1',
        dimensions=(STORM_OBJECT_DIMENSION_KEY, CHARACTER_DIMENSION_KEY)
    )

    string_type = 'S{0:d}'.format(num_id_characters)
    full_ids_char_array = netCDF4.stringtochar(numpy.array(
        full_id_strings, dtype=string_type
    ))
    netcdf_dataset.variables[FULL_IDS_KEY][:] = numpy.array(full_ids_char_array)

    netcdf_dataset.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32,
        dimensions=STORM_OBJECT_DIMENSION_KEY
    )
    netcdf_dataset.variables[VALID_TIMES_KEY][:] = valid_times_unix_sec

    if num_storm_objects_per_chunk is None:
        chunk_size_tuple = None
    else:
        chunk_size_tuple = (
            (num_storm_objects_per_chunk,) + storm_image_matrix.shape[1:]
        )

    netcdf_dataset.createVariable(
        STORM_IMAGE_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(STORM_OBJECT_DIMENSION_KEY, ROW_DIMENSION_KEY,
                    COLUMN_DIMENSION_KEY),
        chunksizes=chunk_size_tuple
    )

    netcdf_dataset.variables[STORM_IMAGE_MATRIX_KEY][:] = storm_image_matrix
    netcdf_dataset.close()


def read_storm_images(
        netcdf_file_name, return_images=True, full_id_strings_to_keep=None,
        valid_times_to_keep_unix_sec=None, num_rows_to_keep=None,
        num_columns_to_keep=None):
    """Reads storm-centered radar images from NetCDF file.

    This file should contain images for one radar field/height.

    If `full_id_strings_to_keep is None or
        valid_times_to_keep_unix_sec is None`,
    this method will return all storm objects in the file.  Otherwise, will
    return only a subset of storm objects.

    If `num_rows_to_keep is None or num_columns_to_keep is None`, this method
    will return full images.  Otherwise, will crop images but keep the same
    center.

    L = number of storm objects to return

    :param netcdf_file_name: Path to input file.
    :param return_images: Boolean flag.  If True, will return metadata and
        images.  If False, will return only metadata.
    :param full_id_strings_to_keep: [used iff `return_images = True`]
        length-L list of full storm IDs.
    :param valid_times_to_keep_unix_sec: [used iff `return_images = True`]
        length-L numpy array of storm times.
    :param num_rows_to_keep: [used iff `return_images = True`]
        See doc for `downsize_storm_images`.
    :param num_columns_to_keep: Same.
    :return: storm_image_dict: Dictionary with the following keys.
    storm_image_dict['storm_image_matrix']: See doc for `_check_storm_images`.
    storm_image_dict['full_storm_id_strings']: Same.
    storm_image_dict['valid_times_unix_sec']: Same.
    storm_image_dict['radar_field_name']: Same.
    storm_image_dict['radar_height_m_agl']: Same.
    storm_image_dict['rotated_grids']: Same.
    storm_image_dict['rotated_grid_spacing_key']: Same.
    """

    error_checking.assert_is_boolean(return_images)
    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    radar_field_name = str(getattr(netcdf_dataset, RADAR_FIELD_NAME_KEY))
    radar_height_m_agl = getattr(netcdf_dataset, RADAR_HEIGHT_KEY)
    rotated_grids = bool(getattr(netcdf_dataset, ROTATED_GRIDS_KEY))

    if rotated_grids:
        rotated_grid_spacing_metres = getattr(
            netcdf_dataset, ROTATED_GRID_SPACING_KEY)
    else:
        rotated_grid_spacing_metres = None

    num_storm_objects = netcdf_dataset.variables[FULL_IDS_KEY].shape[0]

    if num_storm_objects == 0:
        full_id_strings = []
        valid_times_unix_sec = numpy.array([], dtype=int)
    else:
        full_id_strings = netCDF4.chartostring(
            netcdf_dataset.variables[FULL_IDS_KEY][:]
        )
        full_id_strings = [str(f) for f in full_id_strings]

        valid_times_unix_sec = numpy.array(
            netcdf_dataset.variables[VALID_TIMES_KEY][:], dtype=int
        )

    if not return_images:
        return {
            FULL_IDS_KEY: full_id_strings,
            VALID_TIMES_KEY: valid_times_unix_sec,
            RADAR_FIELD_NAME_KEY: radar_field_name,
            RADAR_HEIGHT_KEY: radar_height_m_agl,
            ROTATED_GRIDS_KEY: rotated_grids,
            ROTATED_GRID_SPACING_KEY: rotated_grid_spacing_metres
        }

    filter_storms = not(
        full_id_strings_to_keep is None or valid_times_to_keep_unix_sec is None
    )

    if filter_storms:
        indices_to_keep = tracking_utils.find_storm_objects(
            all_id_strings=full_id_strings,
            all_times_unix_sec=valid_times_unix_sec,
            id_strings_to_keep=full_id_strings_to_keep,
            times_to_keep_unix_sec=valid_times_to_keep_unix_sec)

        full_id_strings = [full_id_strings[i] for i in indices_to_keep]
        valid_times_unix_sec = valid_times_unix_sec[indices_to_keep]
    else:
        indices_to_keep = numpy.linspace(
            0, num_storm_objects - 1, num=num_storm_objects, dtype=int)

    if len(indices_to_keep):
        storm_image_matrix = numpy.array(
            netcdf_dataset.variables[STORM_IMAGE_MATRIX_KEY][
                indices_to_keep, ...]
        )
    else:
        num_rows = netcdf_dataset.dimensions[ROW_DIMENSION_KEY].size
        num_columns = netcdf_dataset.dimensions[COLUMN_DIMENSION_KEY].size
        storm_image_matrix = numpy.full((0, num_rows, num_columns), 0.)

    netcdf_dataset.close()

    storm_image_matrix = downsize_storm_images(
        storm_image_matrix=storm_image_matrix,
        radar_field_name=radar_field_name, num_rows_to_keep=num_rows_to_keep,
        num_columns_to_keep=num_columns_to_keep)

    return {
        STORM_IMAGE_MATRIX_KEY: storm_image_matrix,
        FULL_IDS_KEY: full_id_strings,
        VALID_TIMES_KEY: valid_times_unix_sec,
        RADAR_FIELD_NAME_KEY: radar_field_name,
        RADAR_HEIGHT_KEY: radar_height_m_agl,
        ROTATED_GRIDS_KEY: rotated_grids,
        ROTATED_GRID_SPACING_KEY: rotated_grid_spacing_metres
    }


def find_storm_image_file(
        top_directory_name, spc_date_string, radar_source, radar_field_name,
        radar_height_m_agl, unix_time_sec=None, raise_error_if_missing=True):
    """Finds file with storm-centered radar images.

    If `unix_time_sec is None`, this method finds a file with images for one SPC
    date.  Otherwise, finds a file with images for one time step.

    :param top_directory_name: Name of top-level directory with storm-centered
        images.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :param radar_height_m_agl: Radar height (metres above ground level).
    :param unix_time_sec: [may be None] Time step.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: storm_image_file_name: Path to image file.  If file is missing and
        `raise_error_if_missing = False`, this is the *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    # Check input args.
    error_checking.assert_is_string(top_directory_name)
    time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    radar_utils.check_data_source(radar_source)
    radar_utils.check_field_name(radar_field_name)

    radar_height_m_agl = int(numpy.round(radar_height_m_agl))
    error_checking.assert_is_geq(radar_height_m_agl, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    # Find file.
    if unix_time_sec is None:
        storm_image_file_name = (
            '{0:s}/{1:s}/{2:s}/{3:s}/{4:05d}_metres_agl/storm_images_{5:s}.nc'
        ).format(
            top_directory_name, radar_source, spc_date_string[:4],
            radar_field_name, radar_height_m_agl, spc_date_string
        )
    else:
        storm_image_file_name = (
            '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}/{5:05d}_metres_agl/'
            'storm_images_{6:s}.nc'
        ).format(
            top_directory_name, radar_source, spc_date_string[:4],
            spc_date_string, radar_field_name, radar_height_m_agl,
            time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT)
        )

    if raise_error_if_missing and not os.path.isfile(storm_image_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            storm_image_file_name)

        raise ValueError(error_string)

    return storm_image_file_name


def image_file_name_to_time(storm_image_file_name):
    """Parses time from name of storm-image file.

    :param storm_image_file_name: Path to input file.
    :return: unix_time_sec: Valid time.  If the file contains data for one SPC
        date (rather than one time step), this will be None.
    :return: spc_date_string: SPC date (format "yyyymmdd").
    """

    directory_name, pathless_file_name = os.path.split(storm_image_file_name)
    extensionless_file_name, _ = os.path.splitext(pathless_file_name)
    time_string = extensionless_file_name.split('_')[-1]

    try:
        time_conversion.spc_date_string_to_unix_sec(time_string)
        return None, time_string
    except:
        pass

    unix_time_sec = time_conversion.string_to_unix_sec(time_string, TIME_FORMAT)
    spc_date_string = directory_name.split('/')[-3]
    time_conversion.spc_date_string_to_unix_sec(spc_date_string)

    return unix_time_sec, spc_date_string


def image_file_name_to_field(storm_image_file_name):
    """Parses radar field from name of storm-image file.

    :param storm_image_file_name: Path to input file.
    :return: radar_field_name: Name of radar field.
    :raises: ValueError: if radar field cannot be parsed from file name.
    """

    subdirectory_names = os.path.split(storm_image_file_name)[0].split('/')

    for this_subdir_name in subdirectory_names:
        try:
            radar_utils.check_field_name(this_subdir_name)
            return this_subdir_name
        except:
            pass

    error_string = 'Cannot parse radar field from file name: "{0:s}"'.format(
        storm_image_file_name)
    raise ValueError(error_string)


def image_file_name_to_height(storm_image_file_name):
    """Parses radar height from name of storm-image file.

    :param storm_image_file_name: Path to input file.
    :return: radar_height_m_agl: Radar height (metres above ground level).
    :raises: ValueError: if radar height cannot be parsed from file name.
    """

    keyword = '_metres_agl'
    subdirectory_names = os.path.split(storm_image_file_name)[0].split('/')

    for this_subdir_name in subdirectory_names:
        if keyword in this_subdir_name:
            return int(this_subdir_name.replace(keyword, ''))

    error_string = 'Cannot parse radar height from file name: "{0:s}"'.format(
        storm_image_file_name)
    raise ValueError(error_string)


def find_storm_label_file(
        storm_image_file_name, top_label_dir_name, label_name,
        raise_error_if_missing=True, warn_if_missing=True):
    """Finds file with storm-hazard labels.

    :param storm_image_file_name: Path to file with storm-centered radar images.
    :param top_label_dir_name: Name of top-level directory with hazard labels.
    :param label_name: Name of hazard labels.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :param warn_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = False` and `warn_if_missing = True`, this
        method will print a warning.
    :return: storm_label_file_name: Path to file with hazard labels.  If file is
        missing and `raise_error_if_missing = False`, this is the *expected*
        path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_boolean(warn_if_missing)
    unix_time_sec, spc_date_string = image_file_name_to_time(
        storm_image_file_name)

    target_param_dict = target_val_utils.target_name_to_params(label_name)

    storm_label_file_name = target_val_utils.find_target_file(
        top_directory_name=top_label_dir_name,
        event_type_string=target_param_dict[target_val_utils.EVENT_TYPE_KEY],
        spc_date_string=spc_date_string, unix_time_sec=unix_time_sec,
        raise_error_if_missing=raise_error_if_missing)

    if not os.path.isfile(storm_label_file_name) and warn_if_missing:
        warning_string = (
            'POTENTIAL PROBLEM.  Cannot find file.  Expected at: "{0:s}"'
        ).format(storm_label_file_name)

        print(warning_string)

    return storm_label_file_name


def find_many_files_myrorss_or_mrms(
        top_directory_name, radar_source, radar_field_names,
        start_time_unix_sec, end_time_unix_sec, one_file_per_time_step=True,
        reflectivity_heights_m_agl=None, raise_error_if_all_missing=True,
        raise_error_if_any_missing=False):
    """Finds many files with storm-centered images from MYRORSS or MRMS data.

    T = number of "file times"
    If `one_file_per_time_step = True`, T = number of time steps
    Else, T = number of SPC dates

    C = number of field/height pairs

    :param top_directory_name: Name of top-level directory for storm-centered
        images.
    :param radar_source: See doc for `_fields_and_heights_to_pairs`.
    :param radar_field_names: Same.
    :param start_time_unix_sec: Start time.  This method will find files for all
        times from `start_time_unix_sec`...`end_time_unix_sec`.  If
        `one_file_per_time_step = False`, start time can be any time on the
        first SPC date.
    :param end_time_unix_sec: See above.
    :param one_file_per_time_step: Boolean flag.  If True, this method will seek
        one file per field/height and time step.  If False, will seek one file
        per field/height and SPC date.
    :param reflectivity_heights_m_agl: See doc for
    `_fields_and_heights_to_pairs`.
    :param raise_error_if_all_missing: Boolean flag.  If no files are found and
        `raise_error_if_all_missing = True`, this method will error out.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing = True`, will error out.
    :return: file_dict: Dictionary with the following keys.
    file_dict['image_file_name_matrix']: T-by-C numpy array of paths to image
        files.
    file_dict['valid_times_unix_sec']: length-T numpy array of valid times.  If
        `one_file_per_time_step = False`, valid_times_unix_sec[i] is just a time
        within the [i]th SPC date.
    file_dict['field_name_by_pair']: length-C list with names of radar fields.
    file_dict['height_by_pair_m_agl']: length-C numpy array of radar heights
        (metres above ground level).
    :raises: ValueError: if no files are found and
        `raise_error_if_all_missing = True`.
    """

    field_name_by_pair, height_by_pair_m_agl = _fields_and_heights_to_pairs(
        radar_field_names=radar_field_names,
        reflectivity_heights_m_agl=reflectivity_heights_m_agl,
        radar_source=radar_source)

    first_spc_date_string = time_conversion.time_to_spc_date_string(
        start_time_unix_sec)
    last_spc_date_string = time_conversion.time_to_spc_date_string(
        end_time_unix_sec)

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    error_checking.assert_is_boolean(one_file_per_time_step)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(raise_error_if_any_missing)

    file_dict = {
        FIELD_NAME_BY_PAIR_KEY: field_name_by_pair,
        HEIGHT_BY_PAIR_KEY: height_by_pair_m_agl
    }

    if one_file_per_time_step:
        image_file_name_matrix = None
        valid_times_unix_sec = None

        for i in range(len(spc_date_strings)):
            print('Finding storm-image files for SPC date "{0:s}"...'.format(
                spc_date_strings[i]
            ))

            this_file_name_matrix, these_times_unix_sec = (
                _find_many_files_one_spc_date(
                    top_directory_name=top_directory_name,
                    start_time_unix_sec=start_time_unix_sec,
                    end_time_unix_sec=end_time_unix_sec,
                    spc_date_string=spc_date_strings[i],
                    radar_source=radar_source,
                    field_name_by_pair=field_name_by_pair,
                    height_by_pair_m_agl=height_by_pair_m_agl,
                    raise_error_if_all_missing=False,
                    raise_error_if_any_missing=raise_error_if_any_missing)
            )

            if this_file_name_matrix is None:
                continue

            if image_file_name_matrix is None:
                image_file_name_matrix = copy.deepcopy(this_file_name_matrix)
                valid_times_unix_sec = these_times_unix_sec + 0
            else:
                image_file_name_matrix = numpy.concatenate(
                    (image_file_name_matrix, this_file_name_matrix), axis=0
                )
                valid_times_unix_sec = numpy.concatenate((
                    valid_times_unix_sec, these_times_unix_sec
                ))

        if raise_error_if_all_missing and image_file_name_matrix is None:
            start_time_string = time_conversion.unix_sec_to_string(
                start_time_unix_sec, TIME_FORMAT)
            end_time_string = time_conversion.unix_sec_to_string(
                end_time_unix_sec, TIME_FORMAT)

            error_string = 'Cannot find any files from {0:s} to {1:s}.'.format(
                start_time_string, end_time_string)
            raise ValueError(error_string)

        file_dict.update({
            IMAGE_FILE_NAMES_KEY: image_file_name_matrix,
            VALID_TIMES_KEY: valid_times_unix_sec
        })

        return file_dict

    image_file_name_matrix = None
    valid_spc_date_strings = None
    valid_times_unix_sec = None
    num_field_height_pairs = len(field_name_by_pair)

    for j in range(num_field_height_pairs):
        print((
            'Finding storm-image files for "{0:s}" at {1:d} metres AGL...'
        ).format(
            field_name_by_pair[j], height_by_pair_m_agl[j]
        ))

        if j == 0:
            image_file_names = []
            valid_spc_date_strings = []

            for i in range(len(spc_date_strings)):
                this_file_name = find_storm_image_file(
                    top_directory_name=top_directory_name,
                    spc_date_string=spc_date_strings[i],
                    radar_source=radar_source,
                    radar_field_name=field_name_by_pair[j],
                    radar_height_m_agl=height_by_pair_m_agl[j],
                    raise_error_if_missing=raise_error_if_any_missing)

                if not os.path.isfile(this_file_name):
                    continue

                image_file_names.append(this_file_name)
                valid_spc_date_strings.append(spc_date_strings[i])

            num_times = len(image_file_names)

            if num_times == 0:
                if raise_error_if_all_missing:
                    error_string = (
                        'Cannot find any files from SPC dates "{0:s}" to '
                        '"{1:s}".'
                    ).format(spc_date_strings[0], spc_date_strings[-1])

                    raise ValueError(error_string)

                file_dict.update({
                    IMAGE_FILE_NAMES_KEY: None, VALID_TIMES_KEY: None
                })

                return file_dict

            image_file_name_matrix = numpy.full(
                (num_times, num_field_height_pairs), '', dtype=object
            )
            image_file_name_matrix[:, j] = numpy.array(
                image_file_names, dtype=object)

            valid_times_unix_sec = numpy.array([
                time_conversion.spc_date_string_to_unix_sec(s)
                for s in valid_spc_date_strings
            ], dtype=int)

        else:
            for i in range(len(valid_spc_date_strings)):
                image_file_name_matrix[i, j] = find_storm_image_file(
                    top_directory_name=top_directory_name,
                    spc_date_string=valid_spc_date_strings[i],
                    radar_source=radar_source,
                    radar_field_name=field_name_by_pair[j],
                    radar_height_m_agl=height_by_pair_m_agl[j],
                    raise_error_if_missing=raise_error_if_any_missing)

                if not os.path.isfile(image_file_name_matrix[i, j]):
                    image_file_name_matrix[i, j] = ''

    file_dict.update({
        IMAGE_FILE_NAMES_KEY: image_file_name_matrix,
        VALID_TIMES_KEY: valid_times_unix_sec
    })

    return file_dict


def find_many_files_gridrad(
        top_directory_name, radar_field_names, radar_heights_m_agl,
        start_time_unix_sec, end_time_unix_sec, one_file_per_time_step=True,
        raise_error_if_all_missing=True):
    """Finds many files with storm-centered images from GridRad data.

    T = number of "file times"
    If `one_file_per_time_step = True`, T = number of time steps
    Else, T = number of SPC dates

    F = number of radar fields
    H = number of radar heights

    :param top_directory_name: Name of top-level directory for storm-centered
        images.
    :param radar_field_names: length-F list with names of radar fields.
    :param radar_heights_m_agl: length-H numpy array of radar heights (metres
        above ground level).
    :param start_time_unix_sec: See doc for `find_many_files_myrorss_or_mrms`.
    :param end_time_unix_sec: Same.
    :param one_file_per_time_step: Same.
    :param raise_error_if_all_missing: Same.
    :return: file_dict: Dictionary with the following keys.
    file_dict['image_file_name_matrix']: T-by-F-by-H numpy array of paths to
        image files.
    file_dict['valid_times_unix_sec']: length-T numpy array of valid times.  If
        `one_file_per_time_step = False`, valid_times_unix_sec[i] is just a time
        within the [i]th SPC date.
    file_dict['radar_field_names']: Same as input.
    file_dict['radar_heights_m_agl']: Same as input.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(radar_field_names), num_dimensions=1
    )

    for this_field_name in radar_field_names:
        radar_utils.check_field_name(this_field_name)

    error_checking.assert_is_numpy_array(radar_heights_m_agl, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(radar_heights_m_agl, 0)
    radar_heights_m_agl = numpy.round(radar_heights_m_agl).astype(int)

    error_checking.assert_is_boolean(one_file_per_time_step)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    if one_file_per_time_step:
        all_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=start_time_unix_sec,
            end_time_unix_sec=end_time_unix_sec,
            time_interval_sec=GRIDRAD_TIME_INTERVAL_SEC, include_endpoint=True)

        good_indices = numpy.where(numpy.logical_and(
            all_times_unix_sec >= start_time_unix_sec,
            all_times_unix_sec <= end_time_unix_sec
        ))[0]

        all_times_unix_sec = all_times_unix_sec[good_indices]

        all_spc_date_strings = [
            time_conversion.time_to_spc_date_string(t)
            for t in all_times_unix_sec
        ]
    else:
        first_spc_date_string = time_conversion.time_to_spc_date_string(
            start_time_unix_sec)
        last_spc_date_string = time_conversion.time_to_spc_date_string(
            end_time_unix_sec)

        all_spc_date_strings = time_conversion.get_spc_dates_in_range(
            first_spc_date_string=first_spc_date_string,
            last_spc_date_string=last_spc_date_string)

        all_times_unix_sec = numpy.array([
            time_conversion.spc_date_string_to_unix_sec(s)
            for s in all_spc_date_strings
        ], dtype=int)

    file_dict = {
        RADAR_FIELD_NAMES_KEY: radar_field_names,
        RADAR_HEIGHTS_KEY: radar_heights_m_agl
    }

    image_file_name_matrix = None
    valid_times_unix_sec = None
    valid_spc_date_strings = None
    num_fields = len(radar_field_names)
    num_heights = len(radar_heights_m_agl)

    for j in range(num_fields):
        for k in range(num_heights):
            print((
                'Finding storm-image files for "{0:s}" at {1:d} metres AGL...'
            ).format(
                radar_field_names[j], radar_heights_m_agl[k]
            ))

            if j == 0 and k == 0:
                image_file_names = []
                valid_times_unix_sec = []
                valid_spc_date_strings = []

                for i in range(len(all_times_unix_sec)):
                    if one_file_per_time_step:
                        this_time_unix_sec = all_times_unix_sec[i]
                    else:
                        this_time_unix_sec = None

                    this_file_name = find_storm_image_file(
                        top_directory_name=top_directory_name,
                        unix_time_sec=this_time_unix_sec,
                        spc_date_string=all_spc_date_strings[i],
                        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                        radar_field_name=radar_field_names[j],
                        radar_height_m_agl=radar_heights_m_agl[k],
                        raise_error_if_missing=False)

                    if not os.path.isfile(this_file_name):
                        continue

                    image_file_names.append(this_file_name)
                    valid_times_unix_sec.append(all_times_unix_sec[i])
                    valid_spc_date_strings.append(all_spc_date_strings[i])

                num_times = len(image_file_names)

                if num_times == 0:
                    if raise_error_if_all_missing:
                        if one_file_per_time_step:
                            start_time_string = (
                                time_conversion.unix_sec_to_string(
                                    start_time_unix_sec, TIME_FORMAT)
                            )
                            end_time_string = (
                                time_conversion.unix_sec_to_string(
                                    end_time_unix_sec, TIME_FORMAT)
                            )

                            error_string = (
                                'Cannot find any files from {0:s} to {1:s}.'
                            ).format(start_time_string, end_time_string)

                            raise ValueError(error_string)

                        error_string = (
                            'Cannot find any files from SPC dates "{0:s}" to '
                            '"{1:s}".'
                        ).format(
                            all_spc_date_strings[0], all_spc_date_strings[-1]
                        )

                        raise ValueError(error_string)

                    file_dict.update({
                        IMAGE_FILE_NAMES_KEY: None, VALID_TIMES_KEY: None
                    })

                    return file_dict

                image_file_name_matrix = numpy.full(
                    (num_times, num_fields, num_heights), '', dtype=object
                )
                image_file_name_matrix[:, j, k] = numpy.array(
                    image_file_names, dtype=object)
                valid_times_unix_sec = numpy.array(
                    valid_times_unix_sec, dtype=int)
            else:
                for i in range(len(valid_times_unix_sec)):
                    if one_file_per_time_step:
                        this_time_unix_sec = valid_times_unix_sec[i]
                    else:
                        this_time_unix_sec = None

                    image_file_name_matrix[i, j, k] = find_storm_image_file(
                        top_directory_name=top_directory_name,
                        unix_time_sec=this_time_unix_sec,
                        spc_date_string=valid_spc_date_strings[i],
                        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                        radar_field_name=radar_field_names[j],
                        radar_height_m_agl=radar_heights_m_agl[k],
                        raise_error_if_missing=True)

    file_dict.update({
        IMAGE_FILE_NAMES_KEY: image_file_name_matrix,
        VALID_TIMES_KEY: valid_times_unix_sec
    })

    return file_dict
