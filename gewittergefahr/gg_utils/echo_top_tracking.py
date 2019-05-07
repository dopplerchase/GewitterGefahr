"""Implements the echo-top-based storm-tracking algorithm.

This algorithm is discussed in Section 3c of Homeyer et al. (2017).  The main
advantage of this algorithm (in my experience) over segmotion (Lakshmanan and
Smith 2010) is that it provides more intuitive and longer storm tracks.  The
main disadvantage of the echo-top-based algorithm (in my experience) is that it
provides only storm centers, not objects.  In other words, the echo-top-based
algorithm does not provide the bounding polygons.

--- REFERENCES ---

Haberlie, A. and W. Ashley, 2018: "A method for identifying midlatitude
    mesoscale convective systems in radar mosaics, part II: Tracking". Journal
    of Applied Meteorology and Climatology, in press,
    doi:10.1175/JAMC-D-17-0294.1.

Homeyer, C.R., and J.D. McAuliffe, and K.M. Bedka, 2017: "On the development of
    above-anvil cirrus plumes in extratropical convection". Journal of the
    Atmospheric Sciences, 74 (5), 1617-1633.

Lakshmanan, V., and T. Smith, 2010: "Evaluating a storm tracking algorithm".
    26th Conference on Interactive Information Processing Systems, Atlanta, GA,
    American Meteorological Society.
"""

import os.path
import warnings
import numpy
import pandas
from scipy.ndimage.filters import gaussian_filter
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import track_reanalysis
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import dilation
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import echo_classification as echo_classifn
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DEGREES_LAT_TO_METRES = 60 * 1852
CENTRAL_PROJ_LATITUDE_DEG = 35.
CENTRAL_PROJ_LONGITUDE_DEG = 265.

VALID_RADAR_FIELD_NAMES = [
    radar_utils.ECHO_TOP_15DBZ_NAME, radar_utils.ECHO_TOP_18DBZ_NAME,
    radar_utils.ECHO_TOP_20DBZ_NAME, radar_utils.ECHO_TOP_25DBZ_NAME,
    radar_utils.ECHO_TOP_40DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME
]

VALID_RADAR_SOURCE_NAMES = [
    radar_utils.MYRORSS_SOURCE_ID, radar_utils.MRMS_SOURCE_ID
]

DEFAULT_MIN_ECHO_TOP_KM = 4.
DEFAULT_SMOOTHING_RADIUS_DEG_LAT = 0.024
DEFAULT_HALF_WIDTH_FOR_MAX_FILTER_DEG_LAT = 0.06

DEFAULT_MIN_INTERMAX_DISTANCE_METRES = 0.1 * DEGREES_LAT_TO_METRES
DEFAULT_MIN_SIZE_PIXELS = 0
DEFAULT_MAX_LINK_TIME_SECONDS = 305
DEFAULT_MAX_VELOCITY_DIFF_M_S01 = 10.
DEFAULT_MAX_LINK_DISTANCE_M_S01 = (
    0.125 * DEGREES_LAT_TO_METRES / DEFAULT_MAX_LINK_TIME_SECONDS
)

DEFAULT_MAX_JOIN_TIME_SEC = 610
DEFAULT_MAX_JOIN_ERROR_M_S01 = 20.
DEFAULT_NUM_SECONDS_FOR_VELOCITY = 915
DEFAULT_MIN_REANALYZED_DURATION_SEC = 890

DUMMY_TRACKING_SCALE_METRES2 = int(numpy.round(numpy.pi * 1e8))  # 10-km radius

LATITUDES_KEY = 'latitudes_deg'
LONGITUDES_KEY = 'longitudes_deg'
MAX_VALUES_KEY = 'max_values'
X_COORDS_KEY = 'x_coords_metres'
Y_COORDS_KEY = 'y_coords_metres'
VALID_TIME_KEY = 'unix_time_sec'
CURRENT_TO_PREV_MATRIX_KEY = 'current_to_previous_matrix'

CENTROID_X_COLUMN = 'centroid_x_metres'
CENTROID_Y_COLUMN = 'centroid_y_metres'

GRID_POINT_ROWS_KEY = 'grid_point_rows_array_list'
GRID_POINT_COLUMNS_KEY = 'grid_point_columns_array_list'
GRID_POINT_LATITUDES_KEY = 'grid_point_lats_array_list_deg'
GRID_POINT_LONGITUDES_KEY = 'grid_point_lngs_array_list_deg'
POLYGON_OBJECTS_ROWCOL_KEY = 'polygon_objects_rowcol'
POLYGON_OBJECTS_LATLNG_KEY = 'polygon_objects_latlng_deg'


def _check_radar_field(radar_field_name):
    """Error-checks radar field.

    :param radar_field_name: Field name (string).
    :raises: ValueError: if `radar_field_name not in VALID_RADAR_FIELD_NAMES`.
    """

    error_checking.assert_is_string(radar_field_name)

    if radar_field_name not in VALID_RADAR_FIELD_NAMES:
        error_string = (
            '\n{0:s}\nValid radar fields (listed above) do not include "{1:s}".'
        ).format(str(VALID_RADAR_FIELD_NAMES), radar_field_name)

        raise ValueError(error_string)


def _check_radar_source(radar_source_name):
    """Error-checks source of radar data.

    :param radar_source_name: Data source (string).
    :raises: ValueError: if `radar_source_name not in VALID_RADAR_SOURCE_NAMES`.
    """

    error_checking.assert_is_string(radar_source_name)

    if radar_source_name not in VALID_RADAR_SOURCE_NAMES:
        error_string = (
            '\n{0:s}\nValid radar sources (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_RADAR_SOURCE_NAMES), radar_source_name)

        raise ValueError(error_string)


def _gaussian_smooth_radar_field(radar_matrix, e_folding_radius_pixels,
                                 cutoff_radius_pixels=None):
    """Applies Gaussian smoother to radar field.  NaN's are treated as zero.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param radar_matrix: M-by-N numpy array of data values.
    :param e_folding_radius_pixels: e-folding radius.
    :param cutoff_radius_pixels: Cutoff radius.  If
        `cutoff_radius_pixels is None`, will default to
        `3 * e_folding_radius_pixels`.
    :return: smoothed_radar_matrix: Smoothed version of input.
    """

    e_folding_radius_pixels = float(e_folding_radius_pixels)
    if cutoff_radius_pixels is None:
        cutoff_radius_pixels = 3 * e_folding_radius_pixels

    radar_matrix[numpy.isnan(radar_matrix)] = 0.

    smoothed_radar_matrix = gaussian_filter(
        input=radar_matrix, sigma=e_folding_radius_pixels, order=0,
        mode='constant', cval=0.,
        truncate=cutoff_radius_pixels / e_folding_radius_pixels)

    smoothed_radar_matrix[
        numpy.absolute(smoothed_radar_matrix) < TOLERANCE
    ] = numpy.nan

    return smoothed_radar_matrix


def _find_local_maxima(radar_matrix, radar_metadata_dict,
                       neigh_half_width_pixels):
    """Finds local maxima in radar field.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    P = number of local maxima

    :param radar_matrix: M-by-N numpy array of data values.
    :param radar_metadata_dict: Dictionary created by
        `myrorss_and_mrms_io.read_metadata_from_raw_file`.
    :param neigh_half_width_pixels: Half-width of neighbourhood for max filter.
    :return: local_max_dict_simple: Dictionary with the following keys.
    local_max_dict_simple['latitudes_deg']: length-P numpy array with latitudes
        (deg N) of local maxima.
    local_max_dict_simple['longitudes_deg']: length-P numpy array with
        longitudes (deg E) of local maxima.
    local_max_dict_simple['max_values']: length-P numpy array with magnitudes of
        local maxima.
    """

    filtered_radar_matrix = dilation.dilate_2d_matrix(
        input_matrix=radar_matrix, percentile_level=100.,
        half_width_in_pixels=neigh_half_width_pixels)

    max_index_arrays = numpy.where(
        numpy.absolute(filtered_radar_matrix - radar_matrix) < TOLERANCE
    )

    max_row_indices = max_index_arrays[0]
    max_column_indices = max_index_arrays[1]

    max_latitudes_deg, max_longitudes_deg = radar_utils.rowcol_to_latlng(
        grid_rows=max_row_indices, grid_columns=max_column_indices,
        nw_grid_point_lat_deg=
        radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN],
        nw_grid_point_lng_deg=
        radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
        lat_spacing_deg=radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
        lng_spacing_deg=radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN]
    )

    max_values = radar_matrix[max_row_indices, max_column_indices]

    sort_indices = numpy.argsort(-max_values)
    max_values = max_values[sort_indices]
    max_latitudes_deg = max_latitudes_deg[sort_indices]
    max_longitudes_deg = max_longitudes_deg[sort_indices]

    return {
        LATITUDES_KEY: max_latitudes_deg,
        LONGITUDES_KEY: max_longitudes_deg,
        MAX_VALUES_KEY: max_values
    }


def _remove_redundant_local_maxima(local_max_dict, projection_object,
                                   min_intermax_distance_metres):
    """Removes redundant local maxima at one time.

    P = number of local maxima retained

    :param local_max_dict: Dictionary with at least the following keys.
    local_max_dict['latitudes_deg']: See doc for `_find_local_maxima`.
    local_max_dict['longitudes_deg']: Same.
    local_max_dict['max_values']: Same.

    :param projection_object: Instance of `pyproj.Proj` (used to convert local
        maxima from lat-long to x-y coordinates).
    :param min_intermax_distance_metres: Minimum distance between any pair of
        local maxima.
    :return: local_max_dict: Same as input, except that no pair of maxima is
        within `min_intermax_distance_metres`.  Also contains additional columns
        listed below.
    local_max_dict['x_coords_metres']: length-P numpy array with x-coordinates
        of local maxima.
    local_max_dict['y_coords_metres']: length-P numpy array with y-coordinates
        of local maxima.
    """

    x_coords_metres, y_coords_metres = projections.project_latlng_to_xy(
        local_max_dict[LATITUDES_KEY], local_max_dict[LONGITUDES_KEY],
        projection_object=projection_object,
        false_easting_metres=0., false_northing_metres=0.)

    local_max_dict.update({
        X_COORDS_KEY: x_coords_metres,
        Y_COORDS_KEY: y_coords_metres
    })

    num_maxima = len(x_coords_metres)
    keep_max_flags = numpy.full(num_maxima, True, dtype=bool)

    for i in range(num_maxima):
        if not keep_max_flags[i]:
            continue

        these_distances_metres = numpy.sqrt(
            (x_coords_metres - x_coords_metres[i]) ** 2 +
            (y_coords_metres - y_coords_metres[i]) ** 2
        )

        these_distances_metres[i] = numpy.inf
        these_redundant_indices = numpy.where(
            these_distances_metres < min_intermax_distance_metres
        )[0]

        if len(these_redundant_indices) == 0:
            continue

        these_redundant_indices = numpy.concatenate((
            these_redundant_indices, numpy.array([i], dtype=int)
        ))
        keep_max_flags[these_redundant_indices] = False

        this_best_index = numpy.argmax(
            local_max_dict[MAX_VALUES_KEY][these_redundant_indices]
        )
        this_best_index = these_redundant_indices[this_best_index]
        keep_max_flags[this_best_index] = True

    indices_to_keep = numpy.where(keep_max_flags)[0]

    for this_key in local_max_dict:
        if isinstance(local_max_dict[this_key], list):
            local_max_dict[this_key] = [
                local_max_dict[this_key][k] for k in indices_to_keep
            ]
        elif isinstance(local_max_dict[this_key], numpy.ndarray):
            local_max_dict[this_key] = local_max_dict[this_key][
                indices_to_keep]

    return local_max_dict


def _check_time_period(
        first_spc_date_string, last_spc_date_string, first_time_unix_sec,
        last_time_unix_sec):
    """Error-checks time period.

    :param first_spc_date_string: First SPC date in period (format "yyyymmdd").
    :param last_spc_date_string: Last SPC date in period.
    :param first_time_unix_sec: First time in period.  If
        `first_time_unix_sec is None`, defaults to first time on first SPC date.
    :param last_time_unix_sec: Last time in period.  If
        `last_time_unix_sec is None`, defaults to last time on last SPC date.
    :return: spc_date_strings: 1-D list of SPC dates (format "yyyymmdd").
    :return: first_time_unix_sec: Same as input, but may have been replaced with
        default.
    :return: last_time_unix_sec: Same as input, but may have been replaced with
        default.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    if first_time_unix_sec is None:
        first_time_unix_sec = time_conversion.string_to_unix_sec(
            first_spc_date_string, time_conversion.SPC_DATE_FORMAT
        ) + time_conversion.MIN_SECONDS_INTO_SPC_DATE

    if last_time_unix_sec is None:
        last_time_unix_sec = time_conversion.string_to_unix_sec(
            last_spc_date_string, time_conversion.SPC_DATE_FORMAT
        ) + time_conversion.MAX_SECONDS_INTO_SPC_DATE

    error_checking.assert_is_greater(last_time_unix_sec, first_time_unix_sec)

    assert time_conversion.is_time_in_spc_date(
        first_time_unix_sec, first_spc_date_string)
    assert time_conversion.is_time_in_spc_date(
        last_time_unix_sec, last_spc_date_string)

    return spc_date_strings, first_time_unix_sec, last_time_unix_sec


def _find_input_radar_files(
        top_radar_dir_name, radar_field_name, radar_source_name,
        first_spc_date_string, last_spc_date_string, first_time_unix_sec,
        last_time_unix_sec):
    """Finds radar files (inputs to `run_tracking` -- basically main method).

    T = number of files found

    :param top_radar_dir_name: Name of top-level directory with radar files.
        Files therein will be found by
        `myrorss_and_mrms_io.find_raw_files_one_spc_date`.
    :param radar_field_name: Field name (must be accepted by
        `_check_radar_field`).
    :param radar_source_name: Data source (must be accepted by
        `_check_radar_source`).
    :param first_spc_date_string: See doc for `_check_time_period`.
    :param last_spc_date_string: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :return: radar_file_names: length-T list of paths to radar files.
    :return: valid_times_unix_sec: length-T numpy array of valid times.
    """

    _check_radar_field(radar_field_name)
    _check_radar_source(radar_source_name)

    spc_date_strings, first_time_unix_sec, last_time_unix_sec = (
        _check_time_period(
            first_spc_date_string=first_spc_date_string,
            last_spc_date_string=last_spc_date_string,
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec)
    )

    radar_file_names = []
    valid_times_unix_sec = numpy.array([], dtype=int)
    num_spc_dates = len(spc_date_strings)

    for i in range(num_spc_dates):
        these_file_names = myrorss_and_mrms_io.find_raw_files_one_spc_date(
            spc_date_string=spc_date_strings[i],
            field_name=radar_field_name, data_source=radar_source_name,
            top_directory_name=top_radar_dir_name, raise_error_if_missing=True)

        if i == 0:
            this_first_time_unix_sec = first_time_unix_sec + 0
        else:
            this_first_time_unix_sec = time_conversion.get_start_of_spc_date(
                spc_date_strings[i])

        if i == num_spc_dates - 1:
            this_last_time_unix_sec = last_time_unix_sec + 0
        else:
            this_last_time_unix_sec = time_conversion.get_end_of_spc_date(
                spc_date_strings[i])

        these_times_unix_sec = numpy.array([
            myrorss_and_mrms_io.raw_file_name_to_time(f)
            for f in these_file_names
        ], dtype=int)

        good_indices = numpy.where(numpy.logical_and(
            these_times_unix_sec >= this_first_time_unix_sec,
            these_times_unix_sec <= this_last_time_unix_sec
        ))[0]

        radar_file_names += [these_file_names[k] for k in good_indices]
        valid_times_unix_sec = numpy.concatenate((
            valid_times_unix_sec, these_times_unix_sec[good_indices]
        ))

    sort_indices = numpy.argsort(valid_times_unix_sec)
    valid_times_unix_sec = valid_times_unix_sec[sort_indices]
    radar_file_names = [radar_file_names[k] for k in sort_indices]

    return radar_file_names, valid_times_unix_sec


def _find_input_tracking_files(
        top_tracking_dir_name, first_spc_date_string, last_spc_date_string,
        first_time_unix_sec, last_time_unix_sec):
    """Finds tracking files (inputs to `run_tracking` -- basically main method).

    T = number of SPC dates

    :param top_tracking_dir_name: Name of top-level directory with tracking
        files.  Files therein will be found by
        `storm_tracking_io.find_processed_files_one_spc_date`.
    :param first_spc_date_string: See doc for `_check_time_period`.
    :param last_spc_date_string: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :return: spc_date_strings: length-T list of SPC dates (format "yyyymmdd").
    :return: tracking_file_names_by_date: length-T list, where the [i]th element
        is a 1-D list of paths to tracking files for the [i]th date.
    :return: valid_times_by_date_unix_sec: length-T list, where the [i]th
        element is a 1-D numpy array of valid times for the [i]th date.
    """

    spc_date_strings, first_time_unix_sec, last_time_unix_sec = (
        _check_time_period(
            first_spc_date_string=first_spc_date_string,
            last_spc_date_string=last_spc_date_string,
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec)
    )

    num_spc_dates = len(spc_date_strings)
    tracking_file_names_by_date = [['']] * num_spc_dates
    valid_times_by_date_unix_sec = [numpy.array([], dtype=int)] * num_spc_dates

    for i in range(num_spc_dates):
        these_file_names = tracking_io.find_files_one_spc_date(
            spc_date_string=spc_date_strings[i],
            source_name=tracking_utils.SEGMOTION_NAME,
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2
        )[0]

        if i == 0:
            this_first_time_unix_sec = first_time_unix_sec + 0
        else:
            this_first_time_unix_sec = time_conversion.get_start_of_spc_date(
                spc_date_strings[i])

        if i == num_spc_dates - 1:
            this_last_time_unix_sec = last_time_unix_sec + 0
        else:
            this_last_time_unix_sec = time_conversion.get_end_of_spc_date(
                spc_date_strings[i])

        these_times_unix_sec = numpy.array([
            tracking_io.file_name_to_time(f) for f in these_file_names
        ], dtype=int)

        sort_indices = numpy.argsort(these_times_unix_sec)
        these_file_names = [these_file_names[k] for k in sort_indices]
        these_times_unix_sec = these_times_unix_sec[sort_indices]

        good_indices = numpy.where(numpy.logical_and(
            these_times_unix_sec >= this_first_time_unix_sec,
            these_times_unix_sec <= this_last_time_unix_sec
        ))[0]

        tracking_file_names_by_date[i] = [
            these_file_names[k] for k in good_indices
        ]
        valid_times_by_date_unix_sec[i] = these_times_unix_sec[good_indices]

    return (spc_date_strings, tracking_file_names_by_date,
            valid_times_by_date_unix_sec)


def _local_maxima_to_polygons(
        local_max_dict, echo_top_matrix_km, min_echo_top_km,
        radar_metadata_dict, min_intermax_distance_metres):
    """Converts local maxima at one time from points to polygons.

    M = number of rows in grid (unique grid-point latitudes)
    N = number of columns in grid (unique grid-point longitudes)
    P = number of local maxima
    G_i = number of grid points in the [i]th polygon

    :param local_max_dict: Dictionary with the following keys.
    local_max_dict['latitudes_deg']: length-P numpy array of latitudes (deg N).
    local_max_dict['longitudes_deg']: length-P numpy array of longitudes
        (deg E).

    :param echo_top_matrix_km: M-by-N numpy array of echo tops (km above ground
        or sea level).
    :param min_echo_top_km: Minimum echo top (smaller values are not considered
        local maxima).
    :param radar_metadata_dict: Dictionary created by
        `myrorss_and_mrms_io.read_metadata_from_raw_file`.
    :param min_intermax_distance_metres: Minimum distance between any pair of
        local maxima.

    :return: local_max_dict: Same as input but with the following extra columns.
    local_max_dict['grid_point_rows_array_list']: length-P list, where the [i]th
        element is a numpy array (length G_i) with row indices of grid points in
        the [i]th polygon.
    local_max_dict['grid_point_columns_array_list']: Same but for columns.
    local_max_dict['grid_point_lats_array_list_deg']: Same but for latitudes
        (deg N).
    local_max_dict['grid_point_lngs_array_list_deg']: Same but for longitudes
        (deg E).
    local_max_dict['polygon_objects_rowcol']: length-P list of polygons
        (`shapely.geometry.Polygon` objects) with coordinates in row-column
        space.
    local_max_dict['polygon_objects_latlng']: length-P list of polygons
        (`shapely.geometry.Polygon` objects) with coordinates in lat-long space.
    """

    latitude_extent_deg = (
        radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN] *
        (radar_metadata_dict[radar_utils.NUM_LAT_COLUMN] - 1)
    )
    min_grid_point_latitude_deg = (
        radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN] -
        latitude_extent_deg
    )

    grid_point_latitudes_deg, grid_point_longitudes_deg = (
        grids.get_latlng_grid_points(
            min_latitude_deg=min_grid_point_latitude_deg,
            min_longitude_deg=
            radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
            lng_spacing_deg=radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN],
            num_rows=radar_metadata_dict[radar_utils.NUM_LAT_COLUMN],
            num_columns=radar_metadata_dict[radar_utils.NUM_LNG_COLUMN])
    )

    grid_point_latitudes_deg = grid_point_latitudes_deg[::-1]
    num_maxima = len(local_max_dict[LATITUDES_KEY])

    local_max_dict[GRID_POINT_ROWS_KEY] = [[]] * num_maxima
    local_max_dict[GRID_POINT_COLUMNS_KEY] = [[]] * num_maxima
    local_max_dict[GRID_POINT_LATITUDES_KEY] = [[]] * num_maxima
    local_max_dict[GRID_POINT_LONGITUDES_KEY] = [[]] * num_maxima
    local_max_dict[POLYGON_OBJECTS_ROWCOL_KEY] = numpy.full(
        num_maxima, numpy.nan, dtype=object)
    local_max_dict[POLYGON_OBJECTS_LATLNG_KEY] = numpy.full(
        num_maxima, numpy.nan, dtype=object)

    for i in range(num_maxima):
        this_echo_top_submatrix_km, this_row_offset, this_column_offset = (
            grids.extract_latlng_subgrid(
                data_matrix=echo_top_matrix_km,
                grid_point_latitudes_deg=grid_point_latitudes_deg,
                grid_point_longitudes_deg=grid_point_longitudes_deg,
                center_latitude_deg=local_max_dict[LATITUDES_KEY][i],
                center_longitude_deg=local_max_dict[LONGITUDES_KEY][i],
                max_distance_from_center_metres=min_intermax_distance_metres)
        )

        this_echo_top_submatrix_km[
            numpy.isnan(this_echo_top_submatrix_km)
        ] = 0.

        (local_max_dict[GRID_POINT_ROWS_KEY][i],
         local_max_dict[GRID_POINT_COLUMNS_KEY][i]
        ) = numpy.where(this_echo_top_submatrix_km >= min_echo_top_km)

        if not len(local_max_dict[GRID_POINT_ROWS_KEY][i]):
            this_row = numpy.floor(
                float(this_echo_top_submatrix_km.shape[0]) / 2
            )
            this_column = numpy.floor(
                float(this_echo_top_submatrix_km.shape[1]) / 2
            )

            local_max_dict[GRID_POINT_ROWS_KEY][i] = numpy.array(
                [this_row], dtype=int)
            local_max_dict[GRID_POINT_COLUMNS_KEY][i] = numpy.array(
                [this_column], dtype=int)

        local_max_dict[GRID_POINT_ROWS_KEY][i] = (
            local_max_dict[GRID_POINT_ROWS_KEY][i] + this_row_offset
        )
        local_max_dict[GRID_POINT_COLUMNS_KEY][i] = (
            local_max_dict[GRID_POINT_COLUMNS_KEY][i] + this_column_offset
        )

        these_vertex_rows, these_vertex_columns = (
            polygons.grid_points_in_poly_to_vertices(
                local_max_dict[GRID_POINT_ROWS_KEY][i],
                local_max_dict[GRID_POINT_COLUMNS_KEY][i])
        )

        (local_max_dict[GRID_POINT_LATITUDES_KEY][i],
         local_max_dict[GRID_POINT_LONGITUDES_KEY][i]
        ) = radar_utils.rowcol_to_latlng(
            local_max_dict[GRID_POINT_ROWS_KEY][i],
            local_max_dict[GRID_POINT_COLUMNS_KEY][i],
            nw_grid_point_lat_deg=
            radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN],
            nw_grid_point_lng_deg=
            radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=
            radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
            lng_spacing_deg=
            radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN])

        these_vertex_latitudes_deg, these_vertex_longitudes_deg = (
            radar_utils.rowcol_to_latlng(
                these_vertex_rows, these_vertex_columns,
                nw_grid_point_lat_deg=
                radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN],
                nw_grid_point_lng_deg=
                radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
                lat_spacing_deg=
                radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
                lng_spacing_deg=
                radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN])
        )

        local_max_dict[POLYGON_OBJECTS_ROWCOL_KEY][i] = (
            polygons.vertex_arrays_to_polygon_object(
                these_vertex_columns, these_vertex_rows)
        )
        local_max_dict[POLYGON_OBJECTS_LATLNG_KEY][i] = (
            polygons.vertex_arrays_to_polygon_object(
                these_vertex_longitudes_deg, these_vertex_latitudes_deg)
        )

    return local_max_dict


def _remove_small_polygons(local_max_dict, min_size_pixels):
    """Removes small polygons (storm objects) at one time.

    :param local_max_dict: Dictionary created by `_local_maxima_to_polygons`.
    :param min_size_pixels: Minimum size.
    :return: local_max_dict: Same as input but maybe with fewer storm objects.
    """

    if min_size_pixels == 0:
        return local_max_dict

    num_grid_cells_by_polygon = numpy.array(
        [len(r) for r in local_max_dict[GRID_POINT_ROWS_KEY]],
        dtype=int
    )

    indices_to_keep = numpy.where(
        num_grid_cells_by_polygon >= min_size_pixels
    )[0]

    for this_key in local_max_dict:
        if isinstance(local_max_dict[this_key], list):
            local_max_dict[this_key] = [
                local_max_dict[this_key][k] for k in indices_to_keep
            ]
        elif isinstance(local_max_dict[this_key], numpy.ndarray):
            local_max_dict[this_key] = local_max_dict[this_key][indices_to_keep]

    return local_max_dict


def _write_new_tracks(storm_object_table, top_output_dir_name,
                      valid_times_unix_sec):
    """Writes tracking files (one Pickle file per time step).

    These files are the main output of both `run_tracking` and
    `reanalyze_across_spc_dates`.

    :param storm_object_table: See doc for
        `storm_tracking_io.write_processed_file`.
    :param top_output_dir_name: Name of top-level directory.  File locations
        therein will be determined by `storm_tracking_io.find_processed_file`.
    :param valid_times_unix_sec: 1-D numpy array of valid times.  One file will
        be written for each.
    """

    for this_time_unix_sec in valid_times_unix_sec:
        this_file_name = tracking_io.find_file(
            top_tracking_dir_name=top_output_dir_name,
            valid_time_unix_sec=this_time_unix_sec,
            spc_date_string=time_conversion.time_to_spc_date_string(
                this_time_unix_sec),
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            raise_error_if_missing=False)

        print 'Writing new data to: "{0:s}"...'.format(this_file_name)

        tracking_io.write_file(
            storm_object_table=storm_object_table.loc[
                storm_object_table[tracking_utils.VALID_TIME_COLUMN] ==
                this_time_unix_sec
                ],
            pickle_file_name=this_file_name
        )


def _shuffle_tracking_data(
        storm_object_table_by_date, tracking_file_names_by_date,
        valid_times_by_date_unix_sec, current_date_index, top_output_dir_name):
    """Shuffles data into and out of memory.

    T = number of SPC dates

    :param storm_object_table_by_date: length-T list of pandas DataFrames.  If
        data for the [i]th date are currently out of memory,
        storm_object_table_by_date[i] = None.  If data for the [i]th date are
        currently in memory, storm_object_table_by_date[i] has columns listed in
        `storm_tracking_io.write_processed_file`.
    :param tracking_file_names_by_date: See doc for
        `_find_input_tracking_files`.
    :param valid_times_by_date_unix_sec: Same.
    :param current_date_index: Index of date currently being processed.  Must be
        in range 0...(T - 1).
    :param top_output_dir_name: Name of top-level output directory.  See doc for
        `_write_new_tracks`.
    :return: storm_object_table_by_date: Same as input, except that different
        items are in memory.
    """

    num_spc_dates = len(tracking_file_names_by_date)

    # Shuffle data out of memory.
    if current_date_index == num_spc_dates:
        for j in [num_spc_dates - 2, num_spc_dates - 1]:
            if j < 0:
                continue

            _write_new_tracks(
                storm_object_table=storm_object_table_by_date[j],
                top_output_dir_name=top_output_dir_name,
                valid_times_unix_sec=valid_times_by_date_unix_sec[j]
            )

            print '\n'
            storm_object_table_by_date[j] = pandas.DataFrame()

        return storm_object_table_by_date

    if current_date_index >= 2:
        _write_new_tracks(
            storm_object_table=storm_object_table_by_date[
                current_date_index - 2],
            top_output_dir_name=top_output_dir_name,
            valid_times_unix_sec=valid_times_by_date_unix_sec[
                current_date_index - 2]
        )

        print '\n'
        storm_object_table_by_date[current_date_index - 2] = pandas.DataFrame()

    # Shuffle data into memory.
    for j in [current_date_index - 1, current_date_index,
              current_date_index + 1]:

        if j < 0 or j >= num_spc_dates:
            continue
        if not storm_object_table_by_date[j].empty:
            continue

        storm_object_table_by_date[j] = tracking_io.read_many_files(
            tracking_file_names_by_date[j]
        )
        print '\n'

    return storm_object_table_by_date


def run_tracking(
        top_radar_dir_name, top_output_dir_name, first_spc_date_string,
        last_spc_date_string, first_time_unix_sec=None, last_time_unix_sec=None,
        echo_top_field_name=radar_utils.ECHO_TOP_40DBZ_NAME,
        radar_source_name=radar_utils.MYRORSS_SOURCE_ID,
        top_echo_classifn_dir_name=None,
        min_echo_top_km=DEFAULT_MIN_ECHO_TOP_KM,
        smoothing_radius_deg_lat=DEFAULT_SMOOTHING_RADIUS_DEG_LAT,
        half_width_for_max_filter_deg_lat=
        DEFAULT_HALF_WIDTH_FOR_MAX_FILTER_DEG_LAT,
        min_intermax_distance_metres=DEFAULT_MIN_INTERMAX_DISTANCE_METRES,
        min_polygon_size_pixels=DEFAULT_MIN_SIZE_PIXELS,
        max_link_time_seconds=DEFAULT_MAX_LINK_TIME_SECONDS,
        max_velocity_diff_m_s01=DEFAULT_MAX_VELOCITY_DIFF_M_S01,
        max_link_distance_m_s01=DEFAULT_MAX_LINK_DISTANCE_M_S01,
        min_track_duration_seconds=0,
        num_seconds_back_for_velocity=DEFAULT_NUM_SECONDS_FOR_VELOCITY):
    """Runs echo-top-tracking.  This is effectively the main method.

    :param top_radar_dir_name: See doc for `_find_input_radar_files`.
    :param top_output_dir_name: See doc for `_write_new_tracks`.
    :param first_spc_date_string: See doc for `_check_time_period`.
    :param last_spc_date_string: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :param echo_top_field_name: See doc for `_find_input_radar_files`.
    :param radar_source_name: Same.
    :param top_echo_classifn_dir_name: Name of top-level directory with
        echo-classification files.  Files therein will be found by
        `echo_classification.find_classification_file` and read by
        `echo_classification.read_classifications`.  Tracking will be performed
        only on convective pixels.  If `top_echo_classifn_dir_name is None`,
        tracking will be performed on all pixels.
    :param min_echo_top_km: See doc for `_local_maxima_to_polygons`.
    :param smoothing_radius_deg_lat: See doc for `_gaussian_smooth_radar_field`.
    :param half_width_for_max_filter_deg_lat: See doc for `_find_local_maxima`.
    :param min_intermax_distance_metres: See doc for
        `_remove_redundant_local_maxima`.
    :param min_polygon_size_pixels: See doc for `_remove_small_polygons`.
    :param max_link_time_seconds: See doc for
        `temporal_tracking.link_local_maxima_in_time`.
    :param max_velocity_diff_m_s01: Same.
    :param max_link_distance_m_s01: Same.
    :param min_track_duration_seconds: See doc for
        `temporal_tracking.remove_short_lived_storms`.
    :param num_seconds_back_for_velocity: See doc for
        `temporal_tracking.get_storm_velocities`.
    """

    if min_polygon_size_pixels is None:
        min_polygon_size_pixels = 0

    error_checking.assert_is_integer(min_polygon_size_pixels)
    error_checking.assert_is_geq(min_polygon_size_pixels, 0)
    error_checking.assert_is_greater(min_echo_top_km, 0.)

    radar_file_names, valid_times_unix_sec = _find_input_radar_files(
        top_radar_dir_name=top_radar_dir_name,
        radar_field_name=echo_top_field_name,
        radar_source_name=radar_source_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec)

    num_times = len(valid_times_unix_sec)
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in valid_times_unix_sec
    ]

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=CENTRAL_PROJ_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_PROJ_LONGITUDE_DEG)

    local_max_dict_by_time = [{}] * num_times
    keep_time_indices = []

    for i in range(num_times):
        if top_echo_classifn_dir_name is None:
            this_echo_classifn_file_name = None
            keep_time_indices.append(i)
        else:
            this_echo_classifn_file_name = (
                echo_classifn.find_classification_file(
                    top_directory_name=top_echo_classifn_dir_name,
                    valid_time_unix_sec=valid_times_unix_sec[i],
                    desire_zipped=True, allow_zipped_or_unzipped=True,
                    raise_error_if_missing=False)
            )

            if not os.path.isfile(this_echo_classifn_file_name):
                warning_string = (
                    'POTENTIAL PROBLEM.  Cannot find echo-classification file.'
                    '  Expected at: "{0:s}"'
                ).format(this_echo_classifn_file_name)

                warnings.warn(warning_string)
                local_max_dict_by_time[i] = None
                continue

            keep_time_indices.append(i)

        print 'Reading data from: "{0:s}"...'.format(radar_file_names[i])
        this_metadata_dict = myrorss_and_mrms_io.read_metadata_from_raw_file(
            netcdf_file_name=radar_file_names[i], data_source=radar_source_name)

        this_sparse_grid_table = (
            myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                netcdf_file_name=radar_file_names[i],
                field_name_orig=this_metadata_dict[
                    myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                data_source=radar_source_name,
                sentinel_values=this_metadata_dict[
                    radar_utils.SENTINEL_VALUE_COLUMN]
            )
        )

        this_echo_top_matrix_km = radar_s2f.sparse_to_full_grid(
            sparse_grid_table=this_sparse_grid_table,
            metadata_dict=this_metadata_dict,
            ignore_if_below=min_echo_top_km
        )[0]

        print 'Finding local maxima in "{0:s}" at {1:s}...'.format(
            echo_top_field_name, valid_time_strings[i])

        this_latitude_spacing_deg = this_metadata_dict[
            radar_utils.LAT_SPACING_COLUMN]

        this_echo_top_matrix_km = _gaussian_smooth_radar_field(
            radar_matrix=this_echo_top_matrix_km,
            e_folding_radius_pixels=
            smoothing_radius_deg_lat / this_latitude_spacing_deg
        )

        if this_echo_classifn_file_name is not None:
            print 'Reading data from: "{0:s}"...'.format(
                this_echo_classifn_file_name)

            this_convective_flag_matrix = echo_classifn.read_classifications(
                this_echo_classifn_file_name
            )[0]

            this_convective_flag_matrix = numpy.flip(
                this_convective_flag_matrix, axis=0)
            this_echo_top_matrix_km[this_convective_flag_matrix == False] = 0.

        this_half_width_pixels = int(numpy.round(
            half_width_for_max_filter_deg_lat / this_latitude_spacing_deg
        ))

        local_max_dict_by_time[i] = _find_local_maxima(
            radar_matrix=this_echo_top_matrix_km,
            radar_metadata_dict=this_metadata_dict,
            neigh_half_width_pixels=this_half_width_pixels)

        local_max_dict_by_time[i].update(
            {VALID_TIME_KEY: valid_times_unix_sec[i]}
        )

        local_max_dict_by_time[i] = _local_maxima_to_polygons(
            local_max_dict=local_max_dict_by_time[i],
            echo_top_matrix_km=this_echo_top_matrix_km,
            min_echo_top_km=min_echo_top_km,
            radar_metadata_dict=this_metadata_dict,
            min_intermax_distance_metres=min_intermax_distance_metres)

        local_max_dict_by_time[i] = _remove_small_polygons(
            local_max_dict=local_max_dict_by_time[i],
            min_size_pixels=min_polygon_size_pixels)

        local_max_dict_by_time[i] = _remove_redundant_local_maxima(
            local_max_dict=local_max_dict_by_time[i],
            projection_object=projection_object,
            min_intermax_distance_metres=min_intermax_distance_metres)

        if i == 0:
            this_current_to_prev_matrix = (
                temporal_tracking.link_local_maxima_in_time(
                    current_local_max_dict=local_max_dict_by_time[i],
                    previous_local_max_dict=None,
                    max_link_time_seconds=max_link_time_seconds,
                    max_velocity_diff_m_s01=max_velocity_diff_m_s01,
                    max_link_distance_m_s01=max_link_distance_m_s01)
            )
        else:
            print (
                'Linking local maxima at {0:s} with those at {1:s}...\n'
            ).format(valid_time_strings[i], valid_time_strings[i - 1])

            this_current_to_prev_matrix = (
                temporal_tracking.link_local_maxima_in_time(
                    current_local_max_dict=local_max_dict_by_time[i],
                    previous_local_max_dict=local_max_dict_by_time[i - 1],
                    max_link_time_seconds=max_link_time_seconds,
                    max_velocity_diff_m_s01=max_velocity_diff_m_s01,
                    max_link_distance_m_s01=max_link_distance_m_s01)
            )

        local_max_dict_by_time[i].update(
            {CURRENT_TO_PREV_MATRIX_KEY: this_current_to_prev_matrix}
        )

        if i == 0:
            local_max_dict_by_time[i] = (
                temporal_tracking.get_intermediate_velocities(
                    current_local_max_dict=local_max_dict_by_time[i],
                    previous_local_max_dict=None)
            )
        else:
            local_max_dict_by_time[i] = (
                temporal_tracking.get_intermediate_velocities(
                    current_local_max_dict=local_max_dict_by_time[i],
                    previous_local_max_dict=local_max_dict_by_time[i - 1])
            )

    keep_time_indices = numpy.array(keep_time_indices, dtype=int)
    valid_times_unix_sec = valid_times_unix_sec[keep_time_indices]
    local_max_dict_by_time = [
        local_max_dict_by_time[k] for k in keep_time_indices
    ]

    print SEPARATOR_STRING
    print 'Converting time series of "{0:s}" maxima to storm tracks...'.format(
        echo_top_field_name)
    storm_object_table = temporal_tracking.local_maxima_to_storm_tracks(
        local_max_dict_by_time)

    print 'Removing tracks that last < {0:d} seconds...'.format(
        int(min_track_duration_seconds)
    )
    storm_object_table = temporal_tracking.remove_short_lived_storms(
        storm_object_table=storm_object_table,
        min_duration_seconds=min_track_duration_seconds)

    print 'Computing storm ages...'
    storm_object_table = temporal_tracking.get_storm_ages(
        storm_object_table=storm_object_table,
        tracking_start_time_unix_sec=valid_times_unix_sec[0],
        tracking_end_time_unix_sec=valid_times_unix_sec[-1],
        max_link_time_seconds=max_link_time_seconds, max_join_time_seconds=0)

    print 'Computing storm velocities...'
    storm_object_table = temporal_tracking.get_storm_velocities(
        storm_object_table=storm_object_table,
        num_seconds_back=num_seconds_back_for_velocity)

    print SEPARATOR_STRING
    _write_new_tracks(
        storm_object_table=storm_object_table,
        top_output_dir_name=top_output_dir_name,
        valid_times_unix_sec=valid_times_unix_sec)


def reanalyze_across_spc_dates(
        top_input_dir_name, top_output_dir_name, first_spc_date_string,
        last_spc_date_string, first_time_unix_sec=None, last_time_unix_sec=None,
        tracking_start_time_unix_sec=None, tracking_end_time_unix_sec=None,
        max_link_time_seconds=DEFAULT_MAX_LINK_TIME_SECONDS,
        max_join_time_seconds=DEFAULT_MAX_JOIN_TIME_SEC,
        max_join_error_m_s01=DEFAULT_MAX_JOIN_ERROR_M_S01,
        min_track_duration_seconds=DEFAULT_MIN_REANALYZED_DURATION_SEC,
        num_seconds_back_for_velocity=DEFAULT_NUM_SECONDS_FOR_VELOCITY):
    """Reanalyzes tracks across SPC dates.

    :param top_input_dir_name: Name of top-level directory with original tracks
        (before reanalysis).  For more details, see doc for
        `_find_input_tracking_files`.
    :param top_output_dir_name: Name of top-level directory for new tracks
        (after reanalysis).  For more details, see doc for
        `_write_new_tracks`.
    :param first_spc_date_string: See doc for `_find_input_tracking_files`.
    :param last_spc_date_string: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :param tracking_start_time_unix_sec: First time in tracking period.  If
        `tracking_start_time_unix_sec is None`, defaults to
        `first_time_unix_sec`.
    :param tracking_end_time_unix_sec: Last time in tracking period.  If
        `tracking_end_time_unix_sec is None`, defaults to `last_time_unix_sec`.
    :param max_link_time_seconds: See doc for
        `temporal_tracking.link_local_maxima_in_time`.
    :param max_join_time_seconds: See doc for
        `track_reanalysis.join_collinear_tracks`.
    :param max_join_error_m_s01: Same.
    :param min_track_duration_seconds: See doc for
        `temporal_tracking.remove_short_lived_storms`.
    :param num_seconds_back_for_velocity: See doc for
        `temporal_tracking.get_storm_velocities`.
    """

    (spc_date_strings, tracking_file_names_by_date, valid_times_by_date_unix_sec
    ) = _find_input_tracking_files(
        top_tracking_dir_name=top_input_dir_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec)

    if (tracking_start_time_unix_sec is None
            or tracking_end_time_unix_sec is None):

        these_times_unix_sec = numpy.array(
            [numpy.min(t) for t in valid_times_by_date_unix_sec], dtype=int
        )
        tracking_start_time_unix_sec = numpy.min(these_times_unix_sec)

        these_times_unix_sec = numpy.array(
            [numpy.max(t) for t in valid_times_by_date_unix_sec], dtype=int
        )
        tracking_end_time_unix_sec = numpy.max(these_times_unix_sec)

    else:
        time_conversion.unix_sec_to_string(
            tracking_start_time_unix_sec, TIME_FORMAT)
        time_conversion.unix_sec_to_string(
            tracking_end_time_unix_sec, TIME_FORMAT)
        error_checking.assert_is_greater(
            tracking_end_time_unix_sec, tracking_start_time_unix_sec)

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=CENTRAL_PROJ_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_PROJ_LONGITUDE_DEG)

    num_spc_dates = len(spc_date_strings)

    if num_spc_dates == 1:
        storm_object_table = tracking_io.read_many_files(
            tracking_file_names_by_date[0])
        print SEPARATOR_STRING

        first_late_time_unix_sec = numpy.min(
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
        )
        last_late_time_unix_sec = numpy.max(
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
        )

        storm_object_table = track_reanalysis.join_collinear_tracks(
            storm_object_table=storm_object_table,
            first_late_time_unix_sec=first_late_time_unix_sec,
            last_late_time_unix_sec=last_late_time_unix_sec,
            max_join_time_seconds=max_join_time_seconds,
            max_join_error_m_s01=max_join_error_m_s01)
        print SEPARATOR_STRING

        print 'Removing tracks that last < {0:d} seconds...'.format(
            int(min_track_duration_seconds)
        )

        storm_object_table = temporal_tracking.remove_short_lived_storms(
            storm_object_table=storm_object_table,
            min_duration_seconds=min_track_duration_seconds)

        print 'Recomputing storm ages...'
        storm_object_table = temporal_tracking.get_storm_ages(
            storm_object_table=storm_object_table,
            tracking_start_time_unix_sec=tracking_start_time_unix_sec,
            tracking_end_time_unix_sec=tracking_end_time_unix_sec,
            max_link_time_seconds=max_link_time_seconds,
            max_join_time_seconds=max_join_time_seconds)

        these_x_coords_metres, these_y_coords_metres = (
            projections.project_latlng_to_xy(
                latitudes_deg=storm_object_table[
                    tracking_utils.CENTROID_LATITUDE_COLUMN].values,
                longitudes_deg=storm_object_table[
                    tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
                projection_object=projection_object,
                false_easting_metres=0., false_northing_metres=0.)
        )

        storm_object_table = storm_object_table.assign(**{
            CENTROID_X_COLUMN: these_x_coords_metres,
            CENTROID_Y_COLUMN: these_y_coords_metres
        })

        print 'Computing storm velocities...'
        storm_object_table = temporal_tracking.get_storm_velocities(
            storm_object_table=storm_object_table,
            num_seconds_back=num_seconds_back_for_velocity)

        _write_new_tracks(
            storm_object_table=storm_object_table,
            top_output_dir_name=top_output_dir_name,
            valid_times_unix_sec=valid_times_by_date_unix_sec[0])
        return

    storm_object_table_by_date = [pandas.DataFrame()] * num_spc_dates

    for i in range(num_spc_dates + 1):
        storm_object_table_by_date = _shuffle_tracking_data(
            tracking_file_names_by_date=tracking_file_names_by_date,
            valid_times_by_date_unix_sec=valid_times_by_date_unix_sec,
            storm_object_table_by_date=storm_object_table_by_date,
            current_date_index=i, top_output_dir_name=top_output_dir_name)
        print SEPARATOR_STRING

        if i == num_spc_dates:
            break

        if i != num_spc_dates - 1:
            indices_to_concat = numpy.array([i, i + 1], dtype=int)
            concat_storm_object_table = pandas.concat(
                [storm_object_table_by_date[k] for k in indices_to_concat],
                axis=0, ignore_index=True)

            this_first_time_unix_sec = numpy.min(
                storm_object_table_by_date[i + 1][
                    tracking_utils.VALID_TIME_COLUMN].values
            )
            this_last_time_unix_sec = numpy.max(
                storm_object_table_by_date[i + 1][
                    tracking_utils.VALID_TIME_COLUMN].values
            )

            concat_storm_object_table = track_reanalysis.join_collinear_tracks(
                storm_object_table=concat_storm_object_table,
                first_late_time_unix_sec=this_first_time_unix_sec,
                last_late_time_unix_sec=this_last_time_unix_sec,
                max_join_time_seconds=max_join_time_seconds,
                max_join_error_m_s01=max_join_error_m_s01)
            print SEPARATOR_STRING

            storm_object_table_by_date[i] = concat_storm_object_table.loc[
                concat_storm_object_table[tracking_utils.SPC_DATE_COLUMN] ==
                spc_date_strings[i]
            ]
            storm_object_table_by_date[i + 1] = concat_storm_object_table.loc[
                concat_storm_object_table[tracking_utils.SPC_DATE_COLUMN] ==
                spc_date_strings[i + 1]
            ]

        if i == 0:
            indices_to_concat = numpy.array([i, i + 1], dtype=int)
        elif i == num_spc_dates - 1:
            indices_to_concat = numpy.array([i - 1, i], dtype=int)
        else:
            indices_to_concat = numpy.array([i - 1, i, i + 1], dtype=int)

        concat_storm_object_table = pandas.concat(
            [storm_object_table_by_date[k] for k in indices_to_concat],
            axis=0, ignore_index=True)

        print 'Removing tracks that last < {0:d} seconds...'.format(
            int(min_track_duration_seconds)
        )
        concat_storm_object_table = temporal_tracking.remove_short_lived_storms(
            storm_object_table=concat_storm_object_table,
            min_duration_seconds=min_track_duration_seconds)

        print 'Recomputing storm ages...'
        concat_storm_object_table = temporal_tracking.get_storm_ages(
            storm_object_table=concat_storm_object_table,
            tracking_start_time_unix_sec=tracking_start_time_unix_sec,
            tracking_end_time_unix_sec=tracking_end_time_unix_sec,
            max_link_time_seconds=max_link_time_seconds,
            max_join_time_seconds=max_join_time_seconds)

        these_x_coords_metres, these_y_coords_metres = (
            projections.project_latlng_to_xy(
                latitudes_deg=concat_storm_object_table[
                    tracking_utils.CENTROID_LATITUDE_COLUMN].values,
                longitudes_deg=concat_storm_object_table[
                    tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
                projection_object=projection_object,
                false_easting_metres=0., false_northing_metres=0.)
        )

        concat_storm_object_table = concat_storm_object_table.assign({
            CENTROID_X_COLUMN: these_x_coords_metres,
            CENTROID_Y_COLUMN: these_y_coords_metres
        })

        print 'Computing storm velocities...'
        concat_storm_object_table = temporal_tracking.get_storm_velocities(
            storm_object_table=concat_storm_object_table,
            num_seconds_back=num_seconds_back_for_velocity)

        storm_object_table_by_date[i] = concat_storm_object_table.loc[
            concat_storm_object_table[tracking_utils.SPC_DATE_COLUMN] ==
            spc_date_strings[i]
        ]
        print SEPARATOR_STRING
