"""Links hazardous events to storm cells.

Currently the only "hazardous events" handled by this code are damaging
straight-line wind and tornadoes.

--- DEFINITIONS ---

"Storm cell" = a single thunderstorm (standard meteorological definition).  I
will use S to denote a storm cell.

"Storm object" = one thunderstorm at one time step (snapshot of a storm cell).
I will use s to denote a storm object.
"""

import copy
import pickle
import shutil
import os.path
import warnings
import numpy
import pandas
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

LARGE_INTEGER = int(1e10)

YEAR_FORMAT = '%Y'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WIND_EVENT_STRING = 'wind'
TORNADO_EVENT_STRING = 'tornado'
TORNADOGENESIS_EVENT_STRING = 'tornadogenesis'
VALID_EVENT_TYPE_STRINGS = [
    WIND_EVENT_STRING, TORNADO_EVENT_STRING, TORNADOGENESIS_EVENT_STRING
]

DEFAULT_MAX_TIME_BEFORE_STORM_SEC = 300
DEFAULT_MAX_TIME_AFTER_STORM_SEC = 300
DEFAULT_BOUNDING_BOX_PADDING_METRES = 1e5
DEFAULT_MAX_DISTANCE_FOR_WIND_METRES = 30000.
DEFAULT_MAX_DISTANCE_FOR_TORNADO_METRES = 30000.

REQUIRED_STORM_COLUMNS = [
    tracking_utils.PRIMARY_ID_COLUMN, tracking_utils.SECONDARY_ID_COLUMN,
    tracking_utils.FULL_ID_COLUMN, tracking_utils.VALID_TIME_COLUMN,
    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN,
    tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN,
    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN,
    tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN,
    tracking_utils.TRACKING_START_TIME_COLUMN,
    tracking_utils.TRACKING_END_TIME_COLUMN,
    tracking_utils.CELL_START_TIME_COLUMN, tracking_utils.CELL_END_TIME_COLUMN,
    tracking_utils.CENTROID_LATITUDE_COLUMN,
    tracking_utils.CENTROID_LONGITUDE_COLUMN,
    tracking_utils.LATLNG_POLYGON_COLUMN
]

REQUIRED_WIND_COLUMNS = [
    raw_wind_io.STATION_ID_COLUMN, raw_wind_io.LATITUDE_COLUMN,
    raw_wind_io.LONGITUDE_COLUMN, raw_wind_io.TIME_COLUMN,
    raw_wind_io.U_WIND_COLUMN, raw_wind_io.V_WIND_COLUMN
]

STORM_CENTROID_X_COLUMN = 'centroid_x_metres'
STORM_CENTROID_Y_COLUMN = 'centroid_y_metres'
STORM_VERTICES_X_COLUMN = 'vertices_x_metres'
STORM_VERTICES_Y_COLUMN = 'vertices_y_metres'

EVENT_TIME_COLUMN = 'unix_time_sec'
EVENT_LATITUDE_COLUMN = 'latitude_deg'
EVENT_LONGITUDE_COLUMN = 'longitude_deg'
EVENT_X_COLUMN = 'x_coord_metres'
EVENT_Y_COLUMN = 'y_coord_metres'
NEAREST_SECONDARY_ID_COLUMN = 'nearest_secondary_id_string'
NEAREST_TIME_COLUMN = 'nearest_storm_time_unix_sec'
LINKAGE_DISTANCE_COLUMN = 'linkage_distance_metres'

STORM_VERTEX_X_COLUMN = 'vertex_x_metres'
STORM_VERTEX_Y_COLUMN = 'vertex_y_metres'

LINKAGE_DISTANCES_COLUMN = 'linkage_distances_metres'
RELATIVE_EVENT_TIMES_COLUMN = 'relative_event_times_sec'
EVENT_LATITUDES_COLUMN = 'event_latitudes_deg'
EVENT_LONGITUDES_COLUMN = 'event_longitudes_deg'
MAIN_OBJECT_FLAGS_COLUMN = 'main_object_flags'

FUJITA_RATINGS_COLUMN = 'f_or_ef_scale_ratings'
TORNADO_IDS_COLUMN = 'tornado_id_strings'
WIND_STATION_IDS_COLUMN = 'wind_station_ids'
U_WINDS_COLUMN = 'u_winds_m_s01'
V_WINDS_COLUMN = 'v_winds_m_s01'

THESE_COLUMNS = [
    LINKAGE_DISTANCES_COLUMN, RELATIVE_EVENT_TIMES_COLUMN,
    EVENT_LATITUDES_COLUMN, EVENT_LONGITUDES_COLUMN, MAIN_OBJECT_FLAGS_COLUMN
]

REQUIRED_WIND_LINKAGE_COLUMNS = REQUIRED_STORM_COLUMNS + THESE_COLUMNS + [
    WIND_STATION_IDS_COLUMN, U_WINDS_COLUMN, V_WINDS_COLUMN
]

REQUIRED_TORNADO_LINKAGE_COLUMNS = REQUIRED_STORM_COLUMNS + THESE_COLUMNS + [
    FUJITA_RATINGS_COLUMN, TORNADO_IDS_COLUMN
]


def _check_input_args(
        tracking_file_names, max_time_before_storm_start_sec,
        max_time_after_storm_end_sec, bounding_box_padding_metres,
        storm_interp_time_interval_sec, max_link_distance_metres):
    """Error-checks input arguments.

    :param tracking_file_names: 1-D list of paths to storm-tracking files
        (readable by `storm_tracking_io.read_file`).
    :param max_time_before_storm_start_sec: Max difference between event (E)
        time and beginning of storm cell (S).  If E occurs more than
        `max_time_before_storm_start_sec` before beginning of S, E cannot be
        linked to S.
    :param max_time_after_storm_end_sec: Max difference between event (E) time
        and end of storm cell (S).  If E occurs more than
        `max_time_after_storm_end_sec` after end of S, E cannot be linked to S.
    :param bounding_box_padding_metres: Padding for bounding box around storm
        objects.  Events outside of this bounding box will be thrown out, which
        means that they cannot be linked to storms.  The purpose of the bounding
        box is to reduce the number of events that must be considered, thus
        reducing computing time.
    :param storm_interp_time_interval_sec: Discretization time for
        interpolation of storm positions.  Storms will be interpolated to each
        multiple of `storm_interp_time_interval_sec` between the first and
        last event times.  Setting `storm_interp_time_interval_sec` > 1
        reduces computing time, at the cost of a slight decrease in accuracy.
    :param max_link_distance_metres: Max linkage distance.  If event E is >
        `max_link_distance_metres` from the edge of the nearest storm, it will
        not be linked to any storm.
    """

    error_checking.assert_is_string_list(tracking_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(tracking_file_names), num_dimensions=1)

    error_checking.assert_is_integer(max_time_before_storm_start_sec)
    error_checking.assert_is_geq(max_time_before_storm_start_sec, 0)

    error_checking.assert_is_integer(max_time_after_storm_end_sec)
    error_checking.assert_is_geq(max_time_after_storm_end_sec, 0)

    error_checking.assert_is_integer(storm_interp_time_interval_sec)
    error_checking.assert_is_greater(storm_interp_time_interval_sec, 0)

    error_checking.assert_is_geq(max_link_distance_metres, 0.)
    error_checking.assert_is_geq(
        bounding_box_padding_metres, max_link_distance_metres)


def _get_bounding_box_for_storms(
        storm_object_table, padding_metres=DEFAULT_BOUNDING_BOX_PADDING_METRES):
    """Creates bounding box (with some padding) around all storm objects.

    :param storm_object_table: pandas DataFrame created by
        `_project_storms_latlng_to_xy`.
    :param padding_metres: Padding (will be added to each edge of bounding box).
    :return: x_limits_metres: length-2 numpy array with [min, max] x-coordinates
        of bounding box.
    :return: y_limits_metres: length-2 numpy array with [min, max] y-coordinates
        of bounding box.
    """

    all_x_coords_metres = numpy.array([])
    all_y_coords_metres = numpy.array([])
    num_storms = len(storm_object_table.index)

    for i in range(num_storms):
        all_x_coords_metres = numpy.concatenate((
            all_x_coords_metres,
            storm_object_table[STORM_VERTICES_X_COLUMN].values[i]
        ))

        all_y_coords_metres = numpy.concatenate((
            all_y_coords_metres,
            storm_object_table[STORM_VERTICES_Y_COLUMN].values[i]
        ))

    x_limits_metres = numpy.array([
        numpy.min(all_x_coords_metres) - padding_metres,
        numpy.max(all_x_coords_metres) + padding_metres
    ])

    y_limits_metres = numpy.array([
        numpy.min(all_y_coords_metres) - padding_metres,
        numpy.max(all_y_coords_metres) + padding_metres
    ])

    return x_limits_metres, y_limits_metres


def _project_storms_latlng_to_xy(storm_object_table, projection_object):
    """Projects storm positions from lat-long to x-y coordinates.

    This method projects both centroids and storm outlines.

    V = number of vertices in a given storm outline

    :param storm_object_table: pandas DataFrame created by `_read_input_storm_tracks`.
    :param projection_object: Instance of `pyproj.Proj`, defining an equidistant
        projection.
    :return: storm_object_table: Same as input, but with additional columns
        listed below.
    storm_object_table.centroid_x_metres: x-coordinate of centroid.
    storm_object_table.centroid_y_metres: y-coordinate of centroid.
    storm_object_table.vertices_x_metres: length-V numpy array with x-
        coordinates of vertices.
    storm_object_table.vertices_y_metres: length-V numpy array with y-
        coordinates of vertices.
    """

    centroids_x_metres, centroids_y_metres = projections.project_latlng_to_xy(
        latitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LATITUDE_COLUMN].values,
        longitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
        projection_object=projection_object)

    nested_array = storm_object_table[[
        tracking_utils.PRIMARY_ID_COLUMN, tracking_utils.PRIMARY_ID_COLUMN
    ]].values.tolist()

    storm_object_table = storm_object_table.assign(**{
        STORM_CENTROID_X_COLUMN: centroids_x_metres,
        STORM_CENTROID_Y_COLUMN: centroids_y_metres,
        STORM_VERTICES_X_COLUMN: nested_array,
        STORM_VERTICES_Y_COLUMN: nested_array
    })

    num_storm_objects = len(storm_object_table.index)

    for i in range(num_storm_objects):
        this_vertex_dict_latlng = polygons.polygon_object_to_vertex_arrays(
            storm_object_table[tracking_utils.LATLNG_POLYGON_COLUMN].values[i]
        )

        (storm_object_table[STORM_VERTICES_X_COLUMN].values[i],
         storm_object_table[STORM_VERTICES_Y_COLUMN].values[i]
        ) = projections.project_latlng_to_xy(
            latitudes_deg=this_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN],
            longitudes_deg=this_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN],
            projection_object=projection_object)

    return storm_object_table


def _project_events_latlng_to_xy(event_table, projection_object):
    """Projects event locations from lat-long to x-y coordinates.

    :param event_table: pandas DataFrame with at least the following columns.
    event_table.latitude_deg: Latitude (deg N).
    event_table.longitude_deg: Longitude (deg E).

    :param projection_object: Instance of `pyproj.Proj`, defining an equidistant
        projection.

    :return: event_table: Same as input, but with additional columns listed
        below.
    event_table.x_coord_metres: x-coordinate of event.
    event_table.y_coord_metres: y-coordinate of event.
    """

    x_coords_metres, y_coords_metres = projections.project_latlng_to_xy(
        event_table[EVENT_LATITUDE_COLUMN].values,
        event_table[EVENT_LONGITUDE_COLUMN].values,
        projection_object=projection_object)

    return event_table.assign(**{
        EVENT_X_COLUMN: x_coords_metres,
        EVENT_Y_COLUMN: y_coords_metres
    })


def _filter_events_by_bounding_box(
        event_table, x_limits_metres, y_limits_metres):
    """Removes events outside of bounding box.

    :param event_table: pandas DataFrame with at least the following columns.
    event_table.x_coord_metres: x-coordinate of event.
    event_table.y_coord_metres: y-coordinate of event.

    :return: x_limits_metres: length-2 numpy array with [min, max] x-coordinates
        of bounding box.
    :return: y_limits_metres: length-2 numpy array with [min, max] y-coordinates
        of bounding box.
    :return: event_table: Same as input, but possibly with fewer rows.
    """

    bad_x_flags = numpy.invert(numpy.logical_and(
        event_table[EVENT_X_COLUMN].values >= x_limits_metres[0],
        event_table[EVENT_X_COLUMN].values <= x_limits_metres[1]
    ))

    bad_y_flags = numpy.invert(numpy.logical_and(
        event_table[EVENT_Y_COLUMN].values >= y_limits_metres[0],
        event_table[EVENT_Y_COLUMN].values <= y_limits_metres[1]
    ))

    bad_row_indices = numpy.where(
        numpy.logical_or(bad_x_flags, bad_y_flags)
    )[0]

    return event_table.drop(
        event_table.index[bad_row_indices], axis=0, inplace=False
    )


def _filter_storms_by_time(storm_object_table, max_start_time_unix_sec,
                           min_end_time_unix_sec):
    """Filters storm cells by time.

    Any storm cell with start time > `max_start_time_unix_sec`, or end time <
    `min_end_time_unix_sec`, will be removed.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.cell_start_time_unix_sec: First time in corresponding
        storm cell.
    storm_object_table.cell_end_time_unix_sec: Last time in corresponding storm
        cell.

    :param max_start_time_unix_sec: Latest allowed start time.
    :param min_end_time_unix_sec: Earliest allowed end time.
    :return: storm_object_table: Same as input, but possibly with fewer rows.
    """

    bad_row_flags = numpy.invert(numpy.logical_and(
        storm_object_table[tracking_utils.CELL_START_TIME_COLUMN].values <=
        max_start_time_unix_sec,
        storm_object_table[tracking_utils.CELL_END_TIME_COLUMN].values >=
        min_end_time_unix_sec
    ))

    bad_row_indices = numpy.where(bad_row_flags)[0]

    return storm_object_table.drop(
        storm_object_table.index[bad_row_indices], axis=0, inplace=False
    )


def _interp_one_storm_in_time(storm_object_table_1cell, secondary_id_string,
                              target_time_unix_sec):
    """Interpolates one storm cell in time.

    The storm object nearest to the target time is advected (i.e., moved as a
    solid body, so its shape does not change) to the target time.  Radar data
    usually have a time interval of <= 5 minutes, so interpolation is usually
    over <= 2.5 minutes, and we assume that changes in shape over this time are
    negligible.

    V = number of vertices in a given storm outline

    :param storm_object_table_1cell: pandas DataFrame with at least the
        following columns.  Each row is one storm object, and this table should
        contain data for only one storm cell.  In other words, the ID for each
        storm object should be the same.

    storm_object_table_1cell.unix_time_sec: Valid time.
    storm_object_table_1cell.centroid_x_metres: x-coordinate of centroid.
    storm_object_table_1cell.centroid_y_metres: y-coordinate of centroid.
    storm_object_table_1cell.vertices_x_metres: length-V numpy array with x-
        coordinates of vertices.
    storm_object_table_1cell.vertices_y_metres: length-V numpy array with y-
        coordinates of vertices.

    :param secondary_id_string: Secondary storm ID.
    :param target_time_unix_sec: Target time.  Storm cell will be interpolated
        to this time.

    :return: interp_vertex_table_1object: pandas DataFrame with the following
        columns.  Each row is one vertex of the interpolated storm outline.
    interp_vertex_table_1object.secondary_id_string: Secondary storm ID (same as
        input).
    interp_vertex_table_1object.vertex_x_metres: x-coordinate of vertex.
    interp_vertex_table_1object.vertex_y_metres: y-coordinate of vertex.
    """

    valid_times_unix_sec, orig_to_unique_indices = numpy.unique(
        storm_object_table_1cell[tracking_utils.VALID_TIME_COLUMN].values,
        return_inverse=True)

    num_times = len(valid_times_unix_sec)
    x_centroids_metres = numpy.full(num_times, numpy.nan)
    y_centroids_metres = numpy.full(num_times, numpy.nan)
    x_vertices_by_time_metres = [numpy.array([], dtype=float)] * num_times
    y_vertices_by_time_metres = [numpy.array([], dtype=float)] * num_times

    for i in range(num_times):
        these_orig_indices = numpy.where(orig_to_unique_indices == i)[0]

        x_centroids_metres[i] = numpy.mean(
            storm_object_table_1cell[STORM_CENTROID_X_COLUMN].values[
                these_orig_indices]
        )

        y_centroids_metres[i] = numpy.mean(
            storm_object_table_1cell[STORM_CENTROID_Y_COLUMN].values[
                these_orig_indices]
        )

        for j in these_orig_indices:
            this_x_offset_metres = (
                x_centroids_metres[i] -
                storm_object_table_1cell[STORM_CENTROID_X_COLUMN].values[j]
            )

            this_y_offset_metres = (
                y_centroids_metres[i] -
                storm_object_table_1cell[STORM_CENTROID_Y_COLUMN].values[j]
            )

            these_x_vertices_metres = (
                this_x_offset_metres +
                storm_object_table_1cell[STORM_VERTICES_X_COLUMN].values[j]
            )

            x_vertices_by_time_metres[i] = numpy.concatenate((
                x_vertices_by_time_metres[i], these_x_vertices_metres
            ))

            these_y_vertices_metres = (
                this_y_offset_metres +
                storm_object_table_1cell[STORM_VERTICES_Y_COLUMN].values[j]
            )

            y_vertices_by_time_metres[i] = numpy.concatenate((
                y_vertices_by_time_metres[i], these_y_vertices_metres
            ))

    centroid_matrix = numpy.vstack((x_centroids_metres, y_centroids_metres))

    interp_centroid_vector = interp.interp_in_time(
        input_matrix=centroid_matrix,
        sorted_input_times_unix_sec=valid_times_unix_sec,
        query_times_unix_sec=numpy.array([target_time_unix_sec]),
        method_string=interp.LINEAR_METHOD_STRING, extrapolate=True)

    absolute_time_diffs_sec = numpy.absolute(
        valid_times_unix_sec - target_time_unix_sec
    )
    nearest_time_index = numpy.argmin(absolute_time_diffs_sec)

    new_x_vertices_metres = (
        interp_centroid_vector[0] - x_centroids_metres[nearest_time_index] +
        x_vertices_by_time_metres[nearest_time_index]
    )

    new_y_vertices_metres = (
        interp_centroid_vector[1] - y_centroids_metres[nearest_time_index] +
        y_vertices_by_time_metres[nearest_time_index]
    )

    num_vertices = len(new_x_vertices_metres)

    return pandas.DataFrame.from_dict({
        tracking_utils.SECONDARY_ID_COLUMN:
            [secondary_id_string] * num_vertices,
        STORM_VERTEX_X_COLUMN: new_x_vertices_metres,
        STORM_VERTEX_Y_COLUMN: new_y_vertices_metres
    })


def _interp_storms_in_time(storm_object_table, target_time_unix_sec,
                           max_time_before_start_sec, max_time_after_end_sec):
    """Interpolates each storm cell in time.

    :param storm_object_table: pandas DataFrame created by
        `_project_storms_latlng_to_xy`.
    :param target_time_unix_sec: Target time.  Storm cells will be interpolated
        to this time.
    :param max_time_before_start_sec: Max extrapolation time before beginning of
        storm cell.
    :param max_time_after_end_sec: Max extrapolation time after end of storm
        cell.
    :return: interp_vertex_table: pandas DataFrame with the following columns.
        Each row is one vertex of one interpolated storm object.
    interp_vertex_table.secondary_id_string: Secondary ID for storm cell.
    interp_vertex_table.vertex_x_metres: x-coordinate of vertex.
    interp_vertex_table.vertex_y_metres: y-coordinate of vertex.
    """

    max_start_time_unix_sec = target_time_unix_sec + max_time_before_start_sec
    min_end_time_unix_sec = target_time_unix_sec - max_time_after_end_sec

    sorted_storm_object_table = copy.deepcopy(storm_object_table)

    sorted_storm_object_table.sort_values(
        tracking_utils.VALID_TIME_COLUMN, axis=0, ascending=True,
        inplace=True)

    unique_secondary_id_strings = numpy.unique(numpy.array(
        sorted_storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values
    ))

    list_of_vertex_tables = []
    num_storm_cells = len(unique_secondary_id_strings)

    for j in range(num_storm_cells):
        these_main_rows = numpy.where(
            sorted_storm_object_table[tracking_utils.SECONDARY_ID_COLUMN] ==
            unique_secondary_id_strings[j]
        )[0]

        these_predecessor_rows = temporal_tracking.find_immediate_predecessors(
            storm_object_table=sorted_storm_object_table,
            target_row=these_main_rows[0]
        )

        these_successor_rows = temporal_tracking.find_immediate_successors(
            storm_object_table=sorted_storm_object_table,
            target_row=these_main_rows[-1]
        )

        these_rows = numpy.concatenate((
            these_predecessor_rows, these_main_rows, these_successor_rows
        ))

        if len(these_rows) == 1:
            continue

        this_start_time_unix_sec = sorted_storm_object_table[
            tracking_utils.VALID_TIME_COLUMN
        ].values[these_main_rows[0]]

        if this_start_time_unix_sec > max_start_time_unix_sec:
            continue

        this_end_time_unix_sec = sorted_storm_object_table[
            tracking_utils.VALID_TIME_COLUMN
        ].values[these_main_rows[-1]]

        if this_end_time_unix_sec < min_end_time_unix_sec:
            continue

        list_of_vertex_tables.append(
            _interp_one_storm_in_time(
                storm_object_table_1cell=sorted_storm_object_table.iloc[
                    these_rows],
                secondary_id_string=unique_secondary_id_strings[j],
                target_time_unix_sec=target_time_unix_sec)
        )

        if len(list_of_vertex_tables) == 1:
            continue

        list_of_vertex_tables[-1] = list_of_vertex_tables[-1].align(
            list_of_vertex_tables[0], axis=1
        )[0]

    if len(list_of_vertex_tables) == 0:
        return pandas.DataFrame(
            columns=[tracking_utils.SECONDARY_ID_COLUMN, STORM_VERTEX_X_COLUMN,
                     STORM_VERTEX_Y_COLUMN]
        )

    return pandas.concat(list_of_vertex_tables, axis=0, ignore_index=True)


def _find_nearest_storms_one_time(
        interp_vertex_table, event_x_coords_metres, event_y_coords_metres,
        max_link_distance_metres, max_polygon_attempt_distance_metres=30000.):
    """Finds nearest storm to each event.

    In this case all events are at the same time.

    N = number of events

    :param interp_vertex_table: pandas DataFrame created by
        `_interp_storms_in_time`.
    :param event_x_coords_metres: length-N numpy array with x-coordinates of
        events.
    :param event_y_coords_metres: length-N numpy array with y-coordinates of
        events.
    :param max_link_distance_metres: Max linkage distance.  If the nearest storm
        edge to event E is > `max_link_distance_metres` away, event E will not
        be linked to any storm.
    :param max_polygon_attempt_distance_metres: Max distance for attempting to
        place event inside storm.
    :return: nearest_secondary_id_strings: length-N list, where
        nearest_secondary_id_strings[i] = secondary ID of nearest storm to [i]th
        event.  If nearest_secondary_id_strings[i] = None, no storm was linked
        to [i]th event.
    :return: linkage_distances_metres: length-N numpy array of linkage
        distances.  If linkage_distances_metres[i] = NaN, [i]th event was not
        linked to any storm.
    """

    max_polygon_attempt_distance_metres = max([
        max_polygon_attempt_distance_metres, 2 * max_link_distance_metres
    ])

    unique_secondary_id_strings, orig_to_unique_indices = numpy.unique(
        interp_vertex_table[tracking_utils.SECONDARY_ID_COLUMN].values,
        return_inverse=True)
    unique_secondary_id_strings = unique_secondary_id_strings.tolist()

    num_events = len(event_x_coords_metres)
    nearest_secondary_id_strings = [None] * num_events
    linkage_distances_metres = numpy.full(num_events, numpy.nan)

    for k in range(num_events):
        these_x_diffs_metres = numpy.absolute(
            event_x_coords_metres[k] -
            interp_vertex_table[STORM_VERTEX_X_COLUMN].values
        )

        these_y_diffs_metres = numpy.absolute(
            event_y_coords_metres[k] -
            interp_vertex_table[STORM_VERTEX_Y_COLUMN].values
        )

        these_vertex_indices = numpy.where(numpy.logical_and(
            these_x_diffs_metres <= max_polygon_attempt_distance_metres,
            these_y_diffs_metres <= max_polygon_attempt_distance_metres
        ))[0]

        if len(these_vertex_indices) == 0:
            continue

        # Try placing event inside storm.
        these_secondary_id_strings = numpy.unique(
            interp_vertex_table[tracking_utils.SECONDARY_ID_COLUMN].values[
                these_vertex_indices]
        ).tolist()

        for this_secondary_id_string in these_secondary_id_strings:
            this_storm_indices = numpy.where(
                orig_to_unique_indices ==
                unique_secondary_id_strings.index(this_secondary_id_string)
            )[0]

            this_polygon_object = polygons.vertex_arrays_to_polygon_object(
                exterior_x_coords=interp_vertex_table[
                    STORM_VERTEX_X_COLUMN].values[this_storm_indices],
                exterior_y_coords=interp_vertex_table[
                    STORM_VERTEX_Y_COLUMN].values[this_storm_indices]
            )

            this_event_in_polygon = polygons.point_in_or_on_polygon(
                polygon_object=this_polygon_object,
                query_x_coordinate=event_x_coords_metres[k],
                query_y_coordinate=event_y_coords_metres[k]
            )

            if not this_event_in_polygon:
                continue

            nearest_secondary_id_strings[k] = this_secondary_id_string
            linkage_distances_metres[k] = 0.
            break

        if nearest_secondary_id_strings[k] is not None:
            continue

        # Try placing event near storm.
        these_vertex_indices = numpy.where(numpy.logical_and(
            these_x_diffs_metres <= max_link_distance_metres,
            these_y_diffs_metres <= max_link_distance_metres
        ))[0]

        if len(these_vertex_indices) == 0:
            continue

        these_distances_metres = numpy.sqrt(
            these_x_diffs_metres[these_vertex_indices] ** 2 +
            these_y_diffs_metres[these_vertex_indices] ** 2
        )

        if not numpy.any(these_distances_metres <= max_link_distance_metres):
            continue

        this_min_index = these_vertex_indices[
            numpy.argmin(these_distances_metres)
        ]
        nearest_secondary_id_strings[k] = interp_vertex_table[
            tracking_utils.SECONDARY_ID_COLUMN
        ].values[this_min_index]

        this_storm_indices = numpy.where(
            orig_to_unique_indices ==
            unique_secondary_id_strings.index(nearest_secondary_id_strings[k])
        )[0]

        this_polygon_object = polygons.vertex_arrays_to_polygon_object(
            exterior_x_coords=interp_vertex_table[STORM_VERTEX_X_COLUMN].values[
                this_storm_indices],
            exterior_y_coords=interp_vertex_table[STORM_VERTEX_Y_COLUMN].values[
                this_storm_indices]
        )

        this_event_in_polygon = polygons.point_in_or_on_polygon(
            polygon_object=this_polygon_object,
            query_x_coordinate=event_x_coords_metres[k],
            query_y_coordinate=event_y_coords_metres[k])

        if this_event_in_polygon:
            linkage_distances_metres[k] = 0.
        else:
            linkage_distances_metres[k] = numpy.min(these_distances_metres)

    return nearest_secondary_id_strings, linkage_distances_metres


def _find_nearest_storms(
        storm_object_table, event_table, max_time_before_storm_start_sec,
        max_time_after_storm_end_sec, interp_time_interval_sec,
        max_link_distance_metres):
    """Finds nearest storm to each event.

    In this case the events may be at different times.

    :param storm_object_table: pandas DataFrame created by
        `_project_storms_latlng_to_xy`.
    :param event_table: pandas DataFrame created by
        `_filter_events_by_bounding_box`.
    :param max_time_before_storm_start_sec: See doc for `_check_input_args`.
    :param max_time_after_storm_end_sec: Same.
    :param interp_time_interval_sec: Same.
    :param max_link_distance_metres: Same.

    :return: event_to_storm_table: Same as input argument `event_table`, but
        with the following additional columns.
    event_to_storm_table.nearest_secondary_id_string: Secondary ID of nearest
        storm object.  If event was not linked to a storm, this is None.
    event_to_storm_table.nearest_storm_time_unix_sec: Valid time of nearest
        storm object.  If event was not linked to a storm, this is -1.
    event_to_storm_table.linkage_distance_metres: Distance between event and
        edge of nearest storm object.  If event was not linked to a storm, this
        is NaN.
    """

    interp_times_unix_sec = number_rounding.round_to_nearest(
        event_table[EVENT_TIME_COLUMN].values, interp_time_interval_sec)
    interp_times_unix_sec = numpy.round(interp_times_unix_sec).astype(int)

    unique_interp_times_unix_sec, orig_to_unique_indices = numpy.unique(
        interp_times_unix_sec, return_inverse=True)
    unique_interp_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in unique_interp_times_unix_sec
    ]

    num_unique_interp_times = len(unique_interp_time_strings)
    num_events = len(event_table.index)

    nearest_secondary_id_strings = [None] * num_events
    nearest_times_unix_sec = numpy.full(num_events, -1, dtype=int)
    linkage_distances_metres = numpy.full(num_events, numpy.nan)

    for i in range(num_unique_interp_times):
        these_unassigned_flags = numpy.array(
            [s is None for s in nearest_secondary_id_strings], dtype=bool
        )

        these_event_rows = numpy.where(numpy.logical_and(
            orig_to_unique_indices == i, these_unassigned_flags
        ))[0]

        if len(these_event_rows) == 0:
            continue

        print('Linking events at ~{0:s} to storms...'.format(
            unique_interp_time_strings[i]
        ))

        this_interp_vertex_table = _interp_storms_in_time(
            storm_object_table=storm_object_table,
            target_time_unix_sec=unique_interp_times_unix_sec[i],
            max_time_before_start_sec=max_time_before_storm_start_sec,
            max_time_after_end_sec=max_time_after_storm_end_sec)

        these_nearest_id_strings, these_link_distances_metres = (
            _find_nearest_storms_one_time(
                interp_vertex_table=this_interp_vertex_table,
                event_x_coords_metres=event_table[EVENT_X_COLUMN].values[
                    these_event_rows],
                event_y_coords_metres=event_table[EVENT_Y_COLUMN].values[
                    these_event_rows],
                max_link_distance_metres=max_link_distance_metres)
        )

        if tornado_io.TORNADO_ID_COLUMN not in event_table:
            linkage_distances_metres[these_event_rows] = (
                these_link_distances_metres
            )

            for j in range(len(these_event_rows)):
                k = these_event_rows[j]
                nearest_secondary_id_strings[k] = these_nearest_id_strings[j]
                if nearest_secondary_id_strings[k] is None:
                    continue

                nearest_times_unix_sec[k] = unique_interp_times_unix_sec[i]

            continue

        for j in range(len(these_event_rows)):
            these_same_id_rows = numpy.where(
                event_table[tornado_io.TORNADO_ID_COLUMN].values ==
                event_table[tornado_io.TORNADO_ID_COLUMN].values[
                    these_event_rows[j]
                ]
            )[0]

            linkage_distances_metres[these_same_id_rows] = (
                these_link_distances_metres[j]
            )

            for k in these_same_id_rows:
                nearest_secondary_id_strings[k] = these_nearest_id_strings[j]
                if nearest_secondary_id_strings[k] is None:
                    continue

                nearest_times_unix_sec[k] = unique_interp_times_unix_sec[i]

    unlinked_indices = numpy.where(numpy.isnan(linkage_distances_metres))[0]

    min_storm_latitude_deg = numpy.min(
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
    )
    max_storm_latitude_deg = numpy.max(
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
    )
    min_storm_longitude_deg = numpy.min(
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values
    )
    max_storm_longitude_deg = numpy.max(
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values
    )

    for this_index in unlinked_indices:
        warning_string = (
            'Event at ({0:.2f} deg N, {1:.2f} deg E) COULD NOT BE LINKED to any'
            ' storm in box ({2:.2f}...{3:.2f} deg N, {4:.2f}...{5:.2f} deg E).'
        ).format(
            event_table[EVENT_LATITUDE_COLUMN].values[this_index],
            event_table[EVENT_LONGITUDE_COLUMN].values[this_index],
            min_storm_latitude_deg, max_storm_latitude_deg,
            min_storm_longitude_deg, max_storm_longitude_deg
        )

        warnings.warn(warning_string)

    latitude_in_box_flags = numpy.logical_and(
        event_table[EVENT_LATITUDE_COLUMN].values[unlinked_indices] >=
        min_storm_latitude_deg,
        event_table[EVENT_LATITUDE_COLUMN].values[unlinked_indices] <=
        max_storm_latitude_deg
    )

    longitude_in_box_flags = numpy.logical_and(
        event_table[EVENT_LONGITUDE_COLUMN].values[unlinked_indices] >=
        min_storm_longitude_deg,
        event_table[EVENT_LONGITUDE_COLUMN].values[unlinked_indices] <=
        max_storm_longitude_deg
    )

    in_box_flags = numpy.logical_and(
        latitude_in_box_flags, longitude_in_box_flags)

    num_unlinked_events = len(unlinked_indices)
    num_unlinked_events_in_box = numpy.sum(in_box_flags)

    print((
        'Number of events = {0:d} ... number of storm objects = {1:d} ... '
        'number of unlinked events = {2:d} ... number of these in unbuffered '
        'bounding box around storms = {3:d}'
    ).format(
        len(event_table.index), len(storm_object_table.index),
        num_unlinked_events, num_unlinked_events_in_box
    ))

    return event_table.assign(**{
        NEAREST_SECONDARY_ID_COLUMN: nearest_secondary_id_strings,
        NEAREST_TIME_COLUMN: nearest_times_unix_sec,
        LINKAGE_DISTANCE_COLUMN: linkage_distances_metres
    })


def _reverse_wind_linkages(storm_object_table, wind_to_storm_table):
    """Reverses wind linkages.

    The input `wind_to_storm_table` contains wind-to-storm linkages, where each
    wind observation is linked to 0 or 1 storms.  The output
    `storm_to_winds_table` will contain storm-to-wind linkages, where each storm
    is linked to 0 or more wind observations.

    K = number of wind observations linked to a given storm cell

    :param storm_object_table: pandas DataFrame created by
        `_project_storms_latlng_to_xy`.
    :param wind_to_storm_table: pandas DataFrame created by
        `_find_nearest_storms`.

    :return: storm_to_winds_table: Same as input `storm_object_table`, but with
        additional columns listed below.  Each row is one storm object.
    storm_to_winds_table.wind_station_ids: length-K list of string IDs for
        weather stations.
    storm_to_winds_table.event_latitudes_deg: length-K numpy array of latitudes
        (deg N).
    storm_to_winds_table.event_longitudes_deg: length-K numpy array of
        longitudes (deg E).
    storm_to_winds_table.u_winds_m_s01: length-K numpy array of u-wind
        components (metres per second).
    storm_to_winds_table.v_winds_m_s01: length-K numpy array of v-wind
        components (metres per second).
    storm_to_winds_table.linkage_distance_metres: length-K numpy array of
        linkage distances (from wind observations to nearest edge of storm
        cell).
    storm_to_winds_table.relative_event_times_unix_sec: length-K numpy array
        with relative times of wind observations (wind-ob time minus
        storm-object time).
    storm_to_winds_table.main_object_flags: length-K numpy array of Boolean
        flags.  If main_object_flags[k] = True in the [i]th row, the [i]th storm
        object is the main object to which the [k]th wind observation was
        linked.
    """

    nested_array = storm_object_table[[
        tracking_utils.SECONDARY_ID_COLUMN, tracking_utils.SECONDARY_ID_COLUMN
    ]].values.tolist()

    storm_to_winds_table = copy.deepcopy(storm_object_table)
    storm_to_winds_table = storm_to_winds_table.assign(**{
        WIND_STATION_IDS_COLUMN: nested_array,
        EVENT_LATITUDES_COLUMN: nested_array,
        EVENT_LONGITUDES_COLUMN: nested_array,
        U_WINDS_COLUMN: nested_array,
        V_WINDS_COLUMN: nested_array,
        LINKAGE_DISTANCES_COLUMN: nested_array,
        RELATIVE_EVENT_TIMES_COLUMN: nested_array,
        MAIN_OBJECT_FLAGS_COLUMN: nested_array
    })

    num_storm_objects = len(storm_to_winds_table.index)

    for i in range(num_storm_objects):
        storm_to_winds_table[WIND_STATION_IDS_COLUMN].values[i] = []
        storm_to_winds_table[EVENT_LATITUDES_COLUMN].values[i] = []
        storm_to_winds_table[EVENT_LONGITUDES_COLUMN].values[i] = []
        storm_to_winds_table[U_WINDS_COLUMN].values[i] = []
        storm_to_winds_table[V_WINDS_COLUMN].values[i] = []
        storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[i] = []
        storm_to_winds_table[RELATIVE_EVENT_TIMES_COLUMN].values[i] = []
        storm_to_winds_table[MAIN_OBJECT_FLAGS_COLUMN].values[i] = []

    num_wind_obs = len(wind_to_storm_table.index)

    for k in range(num_wind_obs):
        this_secondary_id_string = wind_to_storm_table[
            NEAREST_SECONDARY_ID_COLUMN
        ].values[k]

        if this_secondary_id_string is None:
            continue

        this_storm_cell_flags = numpy.array([
            s == this_secondary_id_string for s in
            storm_to_winds_table[tracking_utils.SECONDARY_ID_COLUMN].values
        ])
        this_storm_cell_rows = numpy.where(this_storm_cell_flags)[0]

        this_nearest_time_unix_sec = wind_to_storm_table[
            NEAREST_TIME_COLUMN].values[k]

        these_time_diffs_unix_sec = (
            this_nearest_time_unix_sec -
            storm_to_winds_table[tracking_utils.VALID_TIME_COLUMN].values[
                this_storm_cell_rows]
        )

        these_time_diffs_unix_sec[these_time_diffs_unix_sec < 0] = LARGE_INTEGER
        this_main_object_row = this_storm_cell_rows[
            numpy.argmin(these_time_diffs_unix_sec)
        ]

        these_predecessor_rows = temporal_tracking.find_predecessors(
            storm_object_table=storm_to_winds_table,
            target_row=this_main_object_row, num_seconds_back=LARGE_INTEGER,
            num_secondary_id_changes=1, return_all_on_path=True)

        for j in these_predecessor_rows:
            if (storm_to_winds_table[tracking_utils.VALID_TIME_COLUMN].values[j]
                    > this_nearest_time_unix_sec):
                continue

            storm_to_winds_table[WIND_STATION_IDS_COLUMN].values[j].append(
                wind_to_storm_table[raw_wind_io.STATION_ID_COLUMN].values[k]
            )

            storm_to_winds_table[EVENT_LATITUDES_COLUMN].values[j].append(
                wind_to_storm_table[EVENT_LATITUDE_COLUMN].values[k]
            )

            storm_to_winds_table[EVENT_LONGITUDES_COLUMN].values[j].append(
                wind_to_storm_table[EVENT_LONGITUDE_COLUMN].values[k]
            )

            storm_to_winds_table[U_WINDS_COLUMN].values[j].append(
                wind_to_storm_table[raw_wind_io.U_WIND_COLUMN].values[k]
            )

            storm_to_winds_table[V_WINDS_COLUMN].values[j].append(
                wind_to_storm_table[raw_wind_io.V_WIND_COLUMN].values[k]
            )

            storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[j].append(
                wind_to_storm_table[LINKAGE_DISTANCE_COLUMN].values[k]
            )

            this_relative_time_sec = (
                wind_to_storm_table[EVENT_TIME_COLUMN].values[k] -
                storm_to_winds_table[tracking_utils.VALID_TIME_COLUMN].values[j]
            )

            storm_to_winds_table[RELATIVE_EVENT_TIMES_COLUMN].values[j].append(
                this_relative_time_sec)

            storm_to_winds_table[MAIN_OBJECT_FLAGS_COLUMN].values[j].append(
                j == this_main_object_row
            )

    for i in range(num_storm_objects):
        storm_to_winds_table[EVENT_LATITUDES_COLUMN].values[i] = numpy.array(
            storm_to_winds_table[EVENT_LATITUDES_COLUMN].values[i]
        )

        storm_to_winds_table[EVENT_LONGITUDES_COLUMN].values[i] = numpy.array(
            storm_to_winds_table[EVENT_LONGITUDES_COLUMN].values[i]
        )

        storm_to_winds_table[U_WINDS_COLUMN].values[i] = numpy.array(
            storm_to_winds_table[U_WINDS_COLUMN].values[i]
        )

        storm_to_winds_table[V_WINDS_COLUMN].values[i] = numpy.array(
            storm_to_winds_table[V_WINDS_COLUMN].values[i]
        )

        storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[i] = numpy.array(
            storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[i]
        )

        storm_to_winds_table[RELATIVE_EVENT_TIMES_COLUMN].values[i] = (
            numpy.array(
                storm_to_winds_table[RELATIVE_EVENT_TIMES_COLUMN].values[i],
                dtype=int)
        )

        storm_to_winds_table[MAIN_OBJECT_FLAGS_COLUMN].values[i] = (
            numpy.array(
                storm_to_winds_table[MAIN_OBJECT_FLAGS_COLUMN].values[i],
                dtype=bool)
        )

    return storm_to_winds_table


def _reverse_tornado_linkages(storm_object_table, tornado_to_storm_table):
    """Reverses tornado linkages.

    The input `tornado_to_storm_table` contains tornado-to-storm linkages, where
    each tornado is linked to 0 or 1 storms.  The output
    `storm_to_tornadoes_table` will contain storm-to-tornado linkages, where
    each storm is linked to 0 or more tornadoes.

    K = number of tornadoes linked to a given storm cell

    :param storm_object_table: pandas DataFrame created by
        `_project_storms_latlng_to_xy`.
    :param tornado_to_storm_table: pandas DataFrame created by
        `_find_nearest_storms`.

    :return: storm_to_tornadoes_table: Same as input `storm_object_table`, but
        with additional columns listed below.  Each row is one storm object.
    storm_to_tornadoes_table.event_latitudes_deg: length-K numpy array of
        latitudes (deg N).
    storm_to_tornadoes_table.event_longitudes_deg: length-K numpy array of
        longitudes (deg E).
    storm_to_tornadoes_table.f_or_ef_scale_ratings: length-K list of F-scale or
        EF-scale ratings (strings).
    storm_to_tornadoes_table.linkage_distance_metres: length-K numpy array of
        linkage distances (from tornadoes to nearest edge of storm cell).
    storm_to_tornadoes_table.relative_event_times_unix_sec: length-K numpy array
        with relative times of tornadoes (tornado time minus storm-object time).
    storm_to_tornadoes_table.main_object_flags: length-K numpy array of Boolean
        flags.  If main_object_flags[k] = True in the [i]th row, the [i]th storm
        object is the main object to which the [k]th tornado was linked.
    """

    nested_array = storm_object_table[[
        tracking_utils.SECONDARY_ID_COLUMN, tracking_utils.SECONDARY_ID_COLUMN
    ]].values.tolist()

    storm_to_tornadoes_table = copy.deepcopy(storm_object_table)
    storm_to_tornadoes_table = storm_to_tornadoes_table.assign(**{
        EVENT_LATITUDES_COLUMN: nested_array,
        EVENT_LONGITUDES_COLUMN: nested_array,
        FUJITA_RATINGS_COLUMN: nested_array,
        TORNADO_IDS_COLUMN: nested_array,
        LINKAGE_DISTANCES_COLUMN: nested_array,
        RELATIVE_EVENT_TIMES_COLUMN: nested_array,
        MAIN_OBJECT_FLAGS_COLUMN: nested_array
    })

    num_storm_objects = len(storm_to_tornadoes_table.index)

    for i in range(num_storm_objects):
        storm_to_tornadoes_table[EVENT_LATITUDES_COLUMN].values[i] = []
        storm_to_tornadoes_table[EVENT_LONGITUDES_COLUMN].values[i] = []
        storm_to_tornadoes_table[FUJITA_RATINGS_COLUMN].values[i] = []
        storm_to_tornadoes_table[TORNADO_IDS_COLUMN].values[i] = []
        storm_to_tornadoes_table[LINKAGE_DISTANCES_COLUMN].values[i] = []
        storm_to_tornadoes_table[RELATIVE_EVENT_TIMES_COLUMN].values[i] = []
        storm_to_tornadoes_table[MAIN_OBJECT_FLAGS_COLUMN].values[i] = []

    num_tornadoes = len(tornado_to_storm_table.index)

    for k in range(num_tornadoes):
        this_secondary_id_string = tornado_to_storm_table[
            NEAREST_SECONDARY_ID_COLUMN
        ].values[k]

        if this_secondary_id_string is None:
            continue

        this_storm_cell_flags = numpy.array([
            s == this_secondary_id_string for s in
            storm_to_tornadoes_table[tracking_utils.SECONDARY_ID_COLUMN].values
        ])
        this_storm_cell_rows = numpy.where(this_storm_cell_flags)[0]

        this_nearest_time_unix_sec = tornado_to_storm_table[
            NEAREST_TIME_COLUMN].values[k]

        these_time_diffs_unix_sec = (
            this_nearest_time_unix_sec -
            storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values[
                this_storm_cell_rows]
        )

        these_time_diffs_unix_sec[these_time_diffs_unix_sec < 0] = LARGE_INTEGER
        this_main_object_row = this_storm_cell_rows[
            numpy.argmin(these_time_diffs_unix_sec)
        ]

        these_predecessor_rows = temporal_tracking.find_predecessors(
            storm_object_table=storm_to_tornadoes_table,
            target_row=this_main_object_row, num_seconds_back=LARGE_INTEGER,
            num_secondary_id_changes=1, return_all_on_path=True)

        for j in these_predecessor_rows:
            this_flag = (
                storm_to_tornadoes_table[
                    tracking_utils.VALID_TIME_COLUMN].values[j]
                > this_nearest_time_unix_sec
            )

            if this_flag:
                continue

            storm_to_tornadoes_table[EVENT_LATITUDES_COLUMN].values[j].append(
                tornado_to_storm_table[EVENT_LATITUDE_COLUMN].values[k]
            )

            storm_to_tornadoes_table[EVENT_LONGITUDES_COLUMN].values[j].append(
                tornado_to_storm_table[EVENT_LONGITUDE_COLUMN].values[k]
            )

            storm_to_tornadoes_table[FUJITA_RATINGS_COLUMN].values[j].append(
                tornado_to_storm_table[
                    tornado_io.FUJITA_RATING_COLUMN].values[k]
            )

            storm_to_tornadoes_table[TORNADO_IDS_COLUMN].values[j].append(
                tornado_to_storm_table[tornado_io.TORNADO_ID_COLUMN].values[k]
            )

            storm_to_tornadoes_table[LINKAGE_DISTANCES_COLUMN].values[j].append(
                tornado_to_storm_table[LINKAGE_DISTANCE_COLUMN].values[k]
            )

            this_relative_time_sec = (
                tornado_to_storm_table[EVENT_TIME_COLUMN].values[k] -
                storm_to_tornadoes_table[
                    tracking_utils.VALID_TIME_COLUMN].values[j]
            )

            storm_to_tornadoes_table[RELATIVE_EVENT_TIMES_COLUMN].values[
                j].append(this_relative_time_sec)

            storm_to_tornadoes_table[MAIN_OBJECT_FLAGS_COLUMN].values[j].append(
                j == this_main_object_row
            )

    for i in range(num_storm_objects):
        storm_to_tornadoes_table[EVENT_LATITUDES_COLUMN].values[i] = (
            numpy.array(
                storm_to_tornadoes_table[EVENT_LATITUDES_COLUMN].values[i]
            )
        )

        storm_to_tornadoes_table[EVENT_LONGITUDES_COLUMN].values[i] = (
            numpy.array(
                storm_to_tornadoes_table[EVENT_LONGITUDES_COLUMN].values[i]
            )
        )

        storm_to_tornadoes_table[LINKAGE_DISTANCES_COLUMN].values[i] = (
            numpy.array(
                storm_to_tornadoes_table[LINKAGE_DISTANCES_COLUMN].values[i]
            )
        )

        storm_to_tornadoes_table[RELATIVE_EVENT_TIMES_COLUMN].values[i] = (
            numpy.array(
                storm_to_tornadoes_table[RELATIVE_EVENT_TIMES_COLUMN].values[i],
                dtype=int)
        )

        storm_to_tornadoes_table[MAIN_OBJECT_FLAGS_COLUMN].values[i] = (
            numpy.array(
                storm_to_tornadoes_table[MAIN_OBJECT_FLAGS_COLUMN].values[i],
                dtype=bool)
        )

    return storm_to_tornadoes_table


def _remove_storms_near_start_of_period(
        storm_object_table,
        min_time_elapsed_sec=temporal_tracking.DEFAULT_MIN_VELOCITY_TIME_SEC):
    """Removes any storm object near the start of a tracking period.

    This is because velocity estimates near the start of a tracking period are
    lower-quality, which may cause erroneous linkages.

    :param storm_object_table: pandas DataFrame created by
        `_read_input_storm_tracks`.  Each row is one storm object.
    :param min_time_elapsed_sec: Minimum time into tracking period.  Any storm
        object occurring < `min_time_elapsed_sec` into a tracking period will be
        removed.
    :return: storm_object_table: Same as input but maybe with fewer rows.
    """

    times_after_start_sec = (
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values -
        storm_object_table[tracking_utils.TRACKING_START_TIME_COLUMN].values
    )

    bad_indices = numpy.where(times_after_start_sec < min_time_elapsed_sec)[0]

    print((
        '{0:d} of {1:d} storm objects occur within {2:d} seconds of beginning '
        'of tracking period.  REMOVING.'
    ).format(
        len(bad_indices), len(storm_object_table.index), min_time_elapsed_sec
    ))

    return storm_object_table.drop(
        storm_object_table.index[bad_indices], axis=0, inplace=False
    )


def _read_input_storm_tracks(tracking_file_names):
    """Reads storm tracks (input to linkage algorithm).

    :param tracking_file_names: 1-D list of paths to storm-tracking files
        (readable by `storm_tracking_io.read_file`).
    :return: storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.primary_id_string: Primary ID for storm cell.
    storm_object_table.secondary_id_string: Secondary ID for storm cell.
    storm_object_table.full_id_string: Full ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.
    storm_object_table.cell_start_time_unix_sec: First time in corresponding
        storm cell.
    storm_object_table.cell_end_time_unix_sec: Last time in corresponding storm
        cell.
    storm_object_table.tracking_start_time_unix_sec: Start of tracking period.
    storm_object_table.tracking_end_time_unix_sec: End of tracking period.
    storm_object_table.polygon_object_latlng: `shapely.geometry.Polygon`
        object with storm outline in lat-long coordinates.
    """

    list_of_storm_object_tables = []

    for this_file_name in tracking_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_storm_object_table = tracking_io.read_file(this_file_name)[
            REQUIRED_STORM_COLUMNS
        ]

        list_of_storm_object_tables.append(this_storm_object_table)
        if len(list_of_storm_object_tables) == 1:
            continue

        list_of_storm_object_tables[-1] = list_of_storm_object_tables[-1].align(
            list_of_storm_object_tables[0], axis=1
        )[0]

    storm_object_table = pandas.concat(
        list_of_storm_object_tables, axis=0, ignore_index=True)

    return _remove_storms_near_start_of_period(
        storm_object_table=storm_object_table)


def _read_input_wind_observations(
        top_directory_name, storm_times_unix_sec,
        max_time_before_storm_start_sec, max_time_after_storm_end_sec):
    """Reads wind observations (input to linkage algorithm).

    :param top_directory_name: Name of top-level directory.  Files therein will
        be found by `raw_wind_io.find_processed_hourly_files` and read by
        `raw_wind_io.read_processed_file`.
    :param storm_times_unix_sec: 1-D numpy array with valid times of storm
        objects.
    :param max_time_before_storm_start_sec: See doc for `_check_input_args`.
    :param max_time_after_storm_end_sec: Same.

    :return: wind_table: pandas DataFrame with the following columns.
    wind_table.station_id: String ID for station.
    wind_table.unix_time_sec: Valid time.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.u_wind_m_s01: u-wind (metres per second).
    wind_table.v_wind_m_s01: v-wind (metres per second).
    """

    min_wind_time_unix_sec = numpy.min(
        storm_times_unix_sec) - max_time_before_storm_start_sec
    max_wind_time_unix_sec = numpy.max(
        storm_times_unix_sec) + max_time_after_storm_end_sec

    wind_file_names, _ = raw_wind_io.find_processed_hourly_files(
        start_time_unix_sec=min_wind_time_unix_sec,
        end_time_unix_sec=max_wind_time_unix_sec,
        primary_source=raw_wind_io.MERGED_DATA_SOURCE,
        top_directory_name=top_directory_name, raise_error_if_missing=True)

    list_of_wind_tables = []

    for this_file_name in wind_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        list_of_wind_tables.append(
            raw_wind_io.read_processed_file(this_file_name)[
                REQUIRED_WIND_COLUMNS]
        )

        if len(list_of_wind_tables) == 1:
            continue

        list_of_wind_tables[-1] = list_of_wind_tables[-1].align(
            list_of_wind_tables[0], axis=1
        )[0]

    wind_table = pandas.concat(list_of_wind_tables, axis=0, ignore_index=True)

    wind_speeds_m_s01 = numpy.sqrt(
        wind_table[raw_wind_io.U_WIND_COLUMN].values ** 2 +
        wind_table[raw_wind_io.V_WIND_COLUMN].values ** 2
    )

    bad_indices = raw_wind_io.check_wind_speeds(
        wind_speeds_m_s01=wind_speeds_m_s01, one_component=False)
    wind_table.drop(wind_table.index[bad_indices], axis=0, inplace=True)

    column_dict_old_to_new = {
        raw_wind_io.TIME_COLUMN: EVENT_TIME_COLUMN,
        raw_wind_io.LATITUDE_COLUMN: EVENT_LATITUDE_COLUMN,
        raw_wind_io.LONGITUDE_COLUMN: EVENT_LONGITUDE_COLUMN
    }
    wind_table.rename(columns=column_dict_old_to_new, inplace=True)

    return wind_table


def _interp_tornadoes_along_tracks(tornado_table, interp_time_interval_sec):
    """Interpolates each tornado to many points along its track.

    :param tornado_table: pandas DataFrame returned by
        `tornado_io.read_processed_file`.
    :param interp_time_interval_sec: Will interpolate at this time interval
        between start and end points.
    :return: tornado_table: pandas DataFrame with the following columns.
    tornado_table.unix_time_sec: Valid time.
    tornado_table.latitude_deg: Latitude (deg N).
    tornado_table.longitude_deg: Longitude (deg E).
    tornado_table.tornado_id_string: Tornado ID.
    """

    num_tornadoes = len(tornado_table.index)
    tornado_times_unix_sec = numpy.array([], dtype=int)
    tornado_latitudes_deg = numpy.array([])
    tornado_longitudes_deg = numpy.array([])
    tornado_id_strings = []

    for j in range(num_tornadoes):
        this_start_time_unix_sec = tornado_table[
            tornado_io.START_TIME_COLUMN].values[j]

        this_end_time_unix_sec = tornado_table[
            tornado_io.END_TIME_COLUMN].values[j]

        this_num_query_times = 1 + int(numpy.round(
            float(this_end_time_unix_sec - this_start_time_unix_sec) /
            interp_time_interval_sec
        ))

        these_query_times_unix_sec = numpy.linspace(
            this_start_time_unix_sec, this_end_time_unix_sec,
            num=this_num_query_times, dtype=float)

        these_query_times_unix_sec = numpy.round(
            these_query_times_unix_sec
        ).astype(int)

        these_input_latitudes_deg = numpy.array([
            tornado_table[tornado_io.START_LAT_COLUMN].values[j],
            tornado_table[tornado_io.END_LAT_COLUMN].values[j]
        ])

        these_input_longitudes_deg = numpy.array([
            tornado_table[tornado_io.START_LNG_COLUMN].values[j],
            tornado_table[tornado_io.END_LNG_COLUMN].values[j]
        ])

        these_input_times_unix_sec = numpy.array([
            tornado_table[tornado_io.START_TIME_COLUMN].values[j],
            tornado_table[tornado_io.END_TIME_COLUMN].values[j]
        ], dtype=int)

        if this_num_query_times == 1:
            this_query_coord_matrix = numpy.array([
                these_input_longitudes_deg[0], these_input_latitudes_deg[0]
            ])

            this_query_coord_matrix = numpy.reshape(
                this_query_coord_matrix, (2, 1)
            )
        else:
            this_input_coord_matrix = numpy.vstack((
                these_input_longitudes_deg, these_input_latitudes_deg
            ))

            this_query_coord_matrix = interp.interp_in_time(
                input_matrix=this_input_coord_matrix,
                sorted_input_times_unix_sec=these_input_times_unix_sec,
                query_times_unix_sec=these_query_times_unix_sec,
                method_string=interp.LINEAR_METHOD_STRING, extrapolate=False)

        tornado_times_unix_sec = numpy.concatenate((
            tornado_times_unix_sec, these_query_times_unix_sec
        ))

        tornado_latitudes_deg = numpy.concatenate((
            tornado_latitudes_deg, this_query_coord_matrix[1, :]
        ))

        tornado_longitudes_deg = numpy.concatenate((
            tornado_longitudes_deg, this_query_coord_matrix[0, :]
        ))

        this_id_string = tornado_io.create_tornado_id(
            start_time_unix_sec=these_input_times_unix_sec[0],
            start_latitude_deg=these_input_latitudes_deg[0],
            start_longitude_deg=these_input_longitudes_deg[0]
        )

        tornado_id_strings += (
            [this_id_string] * len(these_query_times_unix_sec)
        )

    return pandas.DataFrame.from_dict({
        EVENT_TIME_COLUMN: tornado_times_unix_sec,
        EVENT_LATITUDE_COLUMN: tornado_latitudes_deg,
        EVENT_LONGITUDE_COLUMN: tornado_longitudes_deg,
        tornado_io.TORNADO_ID_COLUMN: tornado_id_strings
    })


def _read_input_tornado_reports(
        input_directory_name, storm_times_unix_sec,
        max_time_before_storm_start_sec, max_time_after_storm_end_sec,
        genesis_only=True, interp_time_interval_sec=None):
    """Reads tornado observations (input to linkage algorithm).

    :param input_directory_name: Name of directory with tornado observations.
        Relevant files will be found by `tornado_io.find_processed_file` and
        read by `tornado_io.read_processed_file`.
    :param storm_times_unix_sec: 1-D numpy array with valid times of storm
        objects.
    :param max_time_before_storm_start_sec: See doc for `_check_input_args`.
    :param max_time_after_storm_end_sec: Same.
    :param genesis_only: Boolean flag.  If True, will return tornadogenesis
        points only.  If False, will return all points along each tornado track.
    :param interp_time_interval_sec: [used only if `genesis_only == False`]
        Time resolution for interpolating tornado location between start and end
        points.

    :return: tornado_table: pandas DataFrame with the following columns.
    tornado_table.unix_time_sec: Valid time.
    tornado_table.latitude_deg: Latitude (deg N).
    tornado_table.longitude_deg: Longitude (deg E).
    tornado_table.tornado_id_string: Tornado ID.
    tornado_table.f_or_ef_rating: F-scale or EF-scale rating (string).
    """

    error_checking.assert_is_boolean(genesis_only)

    if not genesis_only:
        error_checking.assert_is_integer(interp_time_interval_sec)
        error_checking.assert_is_greater(interp_time_interval_sec, 0)

    min_tornado_time_unix_sec = (
        numpy.min(storm_times_unix_sec) - max_time_before_storm_start_sec
    )
    max_tornado_time_unix_sec = (
        numpy.max(storm_times_unix_sec) + max_time_after_storm_end_sec
    )

    min_tornado_year = int(time_conversion.unix_sec_to_string(
        min_tornado_time_unix_sec, YEAR_FORMAT
    ))
    max_tornado_year = int(time_conversion.unix_sec_to_string(
        max_tornado_time_unix_sec, YEAR_FORMAT
    ))
    tornado_years = numpy.linspace(
        min_tornado_year, max_tornado_year,
        num=max_tornado_year - min_tornado_year + 1, dtype=int
    )

    list_of_tornado_tables = []

    for this_year in tornado_years:
        this_file_name = tornado_io.find_processed_file(
            directory_name=input_directory_name, year=this_year)

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        list_of_tornado_tables.append(
            tornado_io.read_processed_file(this_file_name)
        )

        if len(list_of_tornado_tables) == 1:
            continue

        list_of_tornado_tables[-1] = list_of_tornado_tables[-1].align(
            list_of_tornado_tables[0], axis=1
        )[0]

    tornado_table = pandas.concat(
        list_of_tornado_tables, axis=0, ignore_index=True)

    if genesis_only:
        tornado_table = tornado_io.add_tornado_ids_to_table(tornado_table)

        column_dict_old_to_new = {
            tornado_io.START_TIME_COLUMN: EVENT_TIME_COLUMN,
            tornado_io.START_LAT_COLUMN: EVENT_LATITUDE_COLUMN,
            tornado_io.START_LNG_COLUMN: EVENT_LONGITUDE_COLUMN
        }

        tornado_table.rename(columns=column_dict_old_to_new, inplace=True)
    else:
        tornado_table = _interp_tornadoes_along_tracks(
            tornado_table=tornado_table,
            interp_time_interval_sec=interp_time_interval_sec)

    bad_time_flags = numpy.invert(numpy.logical_and(
        tornado_table[EVENT_TIME_COLUMN].values >= min_tornado_time_unix_sec,
        tornado_table[EVENT_TIME_COLUMN].values <= max_tornado_time_unix_sec
    ))
    bad_time_rows = numpy.where(bad_time_flags)[0]

    print(list(tornado_table))

    return tornado_table.drop(
        tornado_table.index[bad_time_rows], axis=0, inplace=False
    )


def _share_linkages_between_periods(
        early_storm_to_events_table, late_storm_to_events_table):
    """Shares linkages between two periods.

    The motivation behind this method is explained in the documentation for
    `share_linkages_across_spc_dates`.

    :param early_storm_to_events_table: pandas DataFrame created by
        `_reverse_wind_linkages` or `_reverse_tornado_linkages`.
    :param late_storm_to_events_table: Same but for a later period.
    :return: early_storm_to_events_table: Same as input but maybe with more
        linkages.
    :return: late_storm_to_events_table: Same as input but maybe with more
        linkages.
    """

    storm_to_events_table = pandas.concat(
        [early_storm_to_events_table, late_storm_to_events_table],
        axis=0, ignore_index=True
    )

    columns_to_change = [
        EVENT_LATITUDES_COLUMN, EVENT_LONGITUDES_COLUMN,
        LINKAGE_DISTANCES_COLUMN, RELATIVE_EVENT_TIMES_COLUMN,
        MAIN_OBJECT_FLAGS_COLUMN
    ]

    if FUJITA_RATINGS_COLUMN in early_storm_to_events_table:
        columns_to_change += [FUJITA_RATINGS_COLUMN, TORNADO_IDS_COLUMN]
    else:
        columns_to_change += [
            WIND_STATION_IDS_COLUMN, U_WINDS_COLUMN, V_WINDS_COLUMN
        ]

    num_early_storm_objects = len(early_storm_to_events_table.index)
    num_late_storm_objects = len(late_storm_to_events_table.index)
    num_storm_objects = num_early_storm_objects + num_late_storm_objects

    for i in range(num_early_storm_objects, num_storm_objects):
        these_main_object_flags = storm_to_events_table[
            MAIN_OBJECT_FLAGS_COLUMN].values[i]

        these_event_indices = numpy.where(these_main_object_flags)[0]
        if len(these_event_indices) == 0:
            continue

        these_predecessor_rows = temporal_tracking.find_predecessors(
            storm_object_table=storm_to_events_table, target_row=i,
            num_seconds_back=LARGE_INTEGER, num_secondary_id_changes=1,
            return_all_on_path=True)

        these_event_times_unix_sec = (
            storm_to_events_table[tracking_utils.VALID_TIME_COLUMN].values[i] +
            storm_to_events_table[RELATIVE_EVENT_TIMES_COLUMN].values[i][
                these_event_indices]
        )

        for j in these_predecessor_rows:
            if j == i:
                continue

            these_relative_times_sec = (
                these_event_times_unix_sec -
                storm_to_events_table[
                    tracking_utils.VALID_TIME_COLUMN].values[j]
            )

            these_main_object_flags = numpy.full(
                len(these_event_indices), False, dtype=bool
            )

            for this_column in columns_to_change:
                if this_column == RELATIVE_EVENT_TIMES_COLUMN:
                    storm_to_events_table[this_column].values[j] = (
                        numpy.concatenate((
                            storm_to_events_table[this_column].values[j],
                            these_relative_times_sec
                        ))
                    )
                elif this_column == MAIN_OBJECT_FLAGS_COLUMN:
                    storm_to_events_table[this_column].values[j] = (
                        numpy.concatenate((
                            storm_to_events_table[this_column].values[j],
                            these_main_object_flags
                        ))
                    )
                else:
                    if isinstance(storm_to_events_table[this_column].values[i],
                                  numpy.ndarray):
                        storm_to_events_table[this_column].values[j] = (
                            numpy.concatenate((
                                storm_to_events_table[this_column].values[j],
                                storm_to_events_table[this_column].values[i][
                                    these_event_indices]
                            ))
                        )
                    else:
                        this_new_list = [
                            storm_to_events_table[this_column].values[i][k]
                            for k in these_event_indices
                        ]

                        storm_to_events_table[this_column].values[j] = (
                            storm_to_events_table[this_column].values[j] +
                            this_new_list
                        )

    early_rows = numpy.linspace(
        0, num_early_storm_objects - 1, num=num_early_storm_objects, dtype=int)

    late_rows = numpy.linspace(
        num_early_storm_objects, num_storm_objects - 1,
        num=num_late_storm_objects, dtype=int)

    return (
        storm_to_events_table.iloc[early_rows],
        storm_to_events_table.iloc[late_rows]
    )


def check_event_type(event_type_string):
    """Error-checks event type.

    :param event_type_string: Event type.
    :raises: ValueError: if `event_type_string not in VALID_EVENT_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(event_type_string)

    if event_type_string not in VALID_EVENT_TYPE_STRINGS:
        error_string = (
            '\n{0:s}\nValid event types (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_EVENT_TYPE_STRINGS), event_type_string)

        raise ValueError(error_string)


def link_storms_to_winds(
        tracking_file_names, top_wind_directory_name,
        max_time_before_storm_start_sec=DEFAULT_MAX_TIME_BEFORE_STORM_SEC,
        max_time_after_storm_end_sec=DEFAULT_MAX_TIME_AFTER_STORM_SEC,
        bounding_box_padding_metres=DEFAULT_BOUNDING_BOX_PADDING_METRES,
        storm_interp_time_interval_sec=10,
        max_link_distance_metres=DEFAULT_MAX_DISTANCE_FOR_WIND_METRES):
    """Links each storm cell to zero or more wind observations.

    :param tracking_file_names: See doc for `_check_input_args`.
    :param top_wind_directory_name: See doc for `_read_input_wind_observations`.
    :param max_time_before_storm_start_sec: See doc for `_check_input_args`.
    :param max_time_after_storm_end_sec: Same.
    :param bounding_box_padding_metres: Same.
    :param storm_interp_time_interval_sec: Same.
    :param max_link_distance_metres: Same.
    :return: storm_to_winds_table: pandas DataFrame created by
        `_reverse_wind_linkages`.
    """

    _check_input_args(
        tracking_file_names=tracking_file_names,
        max_time_before_storm_start_sec=max_time_before_storm_start_sec,
        max_time_after_storm_end_sec=max_time_after_storm_end_sec,
        bounding_box_padding_metres=bounding_box_padding_metres,
        storm_interp_time_interval_sec=storm_interp_time_interval_sec,
        max_link_distance_metres=max_link_distance_metres)

    storm_object_table = _read_input_storm_tracks(tracking_file_names)
    print(SEPARATOR_STRING)

    num_storm_objects = len(storm_object_table.index)

    if num_storm_objects == 0:
        these_times_unix_sec = numpy.array(
            [tracking_io.file_name_to_time(f) for f in tracking_file_names],
            dtype=int
        )

        these_times_unix_sec = numpy.array([
            numpy.min(these_times_unix_sec), numpy.max(these_times_unix_sec)
        ], dtype=int)
    else:
        these_times_unix_sec = storm_object_table[
            tracking_utils.VALID_TIME_COLUMN].values

    wind_table = _read_input_wind_observations(
        top_directory_name=top_wind_directory_name,
        storm_times_unix_sec=these_times_unix_sec,
        max_time_before_storm_start_sec=max_time_before_storm_start_sec,
        max_time_after_storm_end_sec=max_time_after_storm_end_sec)
    print(SEPARATOR_STRING)

    if num_storm_objects == 0:
        num_wind_obs = len(wind_table.index)

        wind_to_storm_table = wind_table.assign(**{
            NEAREST_SECONDARY_ID_COLUMN: [None] * num_wind_obs,
            LINKAGE_DISTANCE_COLUMN: numpy.full(num_wind_obs, numpy.nan)
        })
    else:
        global_centroid_lat_deg, global_centroid_lng_deg = (
            geodetic_utils.get_latlng_centroid(
                latitudes_deg=storm_object_table[
                    tracking_utils.CENTROID_LATITUDE_COLUMN].values,
                longitudes_deg=storm_object_table[
                    tracking_utils.CENTROID_LONGITUDE_COLUMN].values)
        )

        projection_object = projections.init_azimuthal_equidistant_projection(
            central_latitude_deg=global_centroid_lat_deg,
            central_longitude_deg=global_centroid_lng_deg)

        storm_object_table = _project_storms_latlng_to_xy(
            storm_object_table=storm_object_table,
            projection_object=projection_object)

        wind_table = _project_events_latlng_to_xy(
            event_table=wind_table, projection_object=projection_object)

        wind_x_limits_metres, wind_y_limits_metres = (
            _get_bounding_box_for_storms(
                storm_object_table=storm_object_table,
                padding_metres=bounding_box_padding_metres)
        )

        wind_table = _filter_events_by_bounding_box(
            event_table=wind_table, x_limits_metres=wind_x_limits_metres,
            y_limits_metres=wind_y_limits_metres)

        wind_to_storm_table = _find_nearest_storms(
            storm_object_table=storm_object_table, event_table=wind_table,
            max_time_before_storm_start_sec=max_time_before_storm_start_sec,
            max_time_after_storm_end_sec=max_time_after_storm_end_sec,
            interp_time_interval_sec=storm_interp_time_interval_sec,
            max_link_distance_metres=max_link_distance_metres)
        print(SEPARATOR_STRING)

    return _reverse_wind_linkages(storm_object_table=storm_object_table,
                                  wind_to_storm_table=wind_to_storm_table)


def link_storms_to_tornadoes(
        tracking_file_names, tornado_directory_name,
        max_time_before_storm_start_sec=DEFAULT_MAX_TIME_BEFORE_STORM_SEC,
        max_time_after_storm_end_sec=DEFAULT_MAX_TIME_AFTER_STORM_SEC,
        bounding_box_padding_metres=DEFAULT_BOUNDING_BOX_PADDING_METRES,
        storm_interp_time_interval_sec=1,
        max_link_distance_metres=DEFAULT_MAX_DISTANCE_FOR_TORNADO_METRES,
        genesis_only=True, tornado_interp_time_interval_sec=60):
    """Links each storm cell to zero or more tornadoes.

    :param tracking_file_names: See doc for `_check_input_args`.
    :param tornado_directory_name: See doc for `_read_input_tornado_reports`.
    :param max_time_before_storm_start_sec: See doc for `_check_input_args`.
    :param max_time_after_storm_end_sec: Same.
    :param bounding_box_padding_metres: Same.
    :param storm_interp_time_interval_sec: Same.
    :param max_link_distance_metres: Same.
    :param genesis_only: See doc for `_read_input_tornado_reports`.
    :param tornado_interp_time_interval_sec: Same.
    :return: storm_to_tornadoes_table: pandas DataFrame created by
        `_reverse_tornado_linkages`.
    """

    _check_input_args(
        tracking_file_names=tracking_file_names,
        max_time_before_storm_start_sec=max_time_before_storm_start_sec,
        max_time_after_storm_end_sec=max_time_after_storm_end_sec,
        bounding_box_padding_metres=bounding_box_padding_metres,
        storm_interp_time_interval_sec=storm_interp_time_interval_sec,
        max_link_distance_metres=max_link_distance_metres)

    storm_object_table = _read_input_storm_tracks(tracking_file_names)
    print(SEPARATOR_STRING)

    num_storm_objects = len(storm_object_table.index)

    if num_storm_objects == 0:
        these_times_unix_sec = numpy.array(
            [tracking_io.file_name_to_time(f) for f in tracking_file_names],
            dtype=int
        )

        these_times_unix_sec = numpy.array([
            numpy.min(these_times_unix_sec), numpy.max(these_times_unix_sec)
        ], dtype=int)
    else:
        these_times_unix_sec = storm_object_table[
            tracking_utils.VALID_TIME_COLUMN].values

    tornado_table = _read_input_tornado_reports(
        input_directory_name=tornado_directory_name,
        storm_times_unix_sec=these_times_unix_sec,
        max_time_before_storm_start_sec=max_time_before_storm_start_sec,
        max_time_after_storm_end_sec=max_time_after_storm_end_sec,
        genesis_only=genesis_only,
        interp_time_interval_sec=tornado_interp_time_interval_sec)

    print(SEPARATOR_STRING)

    if num_storm_objects == 0:
        num_tornadoes = len(tornado_table.index)

        tornado_to_storm_table = tornado_table.assign(**{
            NEAREST_SECONDARY_ID_COLUMN: [None] * num_tornadoes,
            LINKAGE_DISTANCE_COLUMN: numpy.full(num_tornadoes, numpy.nan)
        })
    else:
        global_centroid_lat_deg, global_centroid_lng_deg = (
            geodetic_utils.get_latlng_centroid(
                latitudes_deg=storm_object_table[
                    tracking_utils.CENTROID_LATITUDE_COLUMN].values,
                longitudes_deg=storm_object_table[
                    tracking_utils.CENTROID_LONGITUDE_COLUMN].values)
        )

        projection_object = projections.init_azimuthal_equidistant_projection(
            central_latitude_deg=global_centroid_lat_deg,
            central_longitude_deg=global_centroid_lng_deg)

        storm_object_table = _project_storms_latlng_to_xy(
            storm_object_table=storm_object_table,
            projection_object=projection_object)

        tornado_table = _project_events_latlng_to_xy(
            event_table=tornado_table, projection_object=projection_object)

        tornado_x_limits_metres, tornado_y_limits_metres = (
            _get_bounding_box_for_storms(
                storm_object_table=storm_object_table,
                padding_metres=bounding_box_padding_metres)
        )

        tornado_table = _filter_events_by_bounding_box(
            event_table=tornado_table, x_limits_metres=tornado_x_limits_metres,
            y_limits_metres=tornado_y_limits_metres)

        tornado_to_storm_table = _find_nearest_storms(
            storm_object_table=storm_object_table, event_table=tornado_table,
            max_time_before_storm_start_sec=max_time_before_storm_start_sec,
            max_time_after_storm_end_sec=max_time_after_storm_end_sec,
            interp_time_interval_sec=storm_interp_time_interval_sec,
            max_link_distance_metres=max_link_distance_metres)
        print(SEPARATOR_STRING)

    return _reverse_tornado_linkages(
        storm_object_table=storm_object_table,
        tornado_to_storm_table=tornado_to_storm_table)


def share_linkages_across_spc_dates(
        top_input_dir_name, first_spc_date_string, last_spc_date_string,
        top_output_dir_name, event_type_string):
    """Shares linkages across SPC dates.

    This is important because linkage is usually done for one SPC date at a
    time, using either `link_storms_to_winds` or `link_storms_to_tornadoes`.
    These methods have no way of sharing info with the next or previous SPC
    dates.  Thus, if event E is linked to storm cell S on day D, the files for
    days D - 1 and D + 1 will not contain this linkage, even if storm cell S
    exists on day D - 1 or D + 1.

    :param top_input_dir_name: Name of top-level input directory.  Files therein
        will be found by `find_linkage_file` and read by `read_linkage_file`.
    :param first_spc_date_string: First SPC date (format "yyyymmdd").  Linkages
        will be shared for all dates in the period `first_spc_date_string`...
        `last_spc_date_string`.
    :param last_spc_date_string: See above.
    :param top_output_dir_name: Name of top-level output directory.  Files will
        be written by `write_linkage_file` to locations therein, determined by
        `find_linkage_file`.
    :param event_type_string: Event type (must be accepted by
        `check_event_type`).
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    num_spc_dates = len(spc_date_strings)
    orig_linkage_file_names = [''] * num_spc_dates
    new_linkage_file_names = [''] * num_spc_dates

    for i in range(num_spc_dates):
        orig_linkage_file_names[i] = find_linkage_file(
            top_directory_name=top_input_dir_name,
            event_type_string=event_type_string, raise_error_if_missing=True,
            spc_date_string=spc_date_strings[i])

        new_linkage_file_names[i] = find_linkage_file(
            top_directory_name=top_output_dir_name,
            event_type_string=event_type_string, raise_error_if_missing=False,
            spc_date_string=spc_date_strings[i])

    if num_spc_dates == 1:
        warning_string = (
            'There is only one SPC date ("{0:s}"), so the method '
            '`share_linkages_across_spc_dates` has nothing to do.'
        ).format(spc_date_strings[0])

        warnings.warn(warning_string)

        if orig_linkage_file_names[0] == new_linkage_file_names[0]:
            return

        print('Copying file from "{0:s}" to "{1:s}"...'.format(
            orig_linkage_file_names[0], new_linkage_file_names[0]
        ))

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=new_linkage_file_names[0]
        )

        shutil.copyfile(orig_linkage_file_names[0], new_linkage_file_names[0])
        return

    storm_to_events_table_by_date = [pandas.DataFrame()] * num_spc_dates

    for i in range(num_spc_dates + 1):
        if i == num_spc_dates:
            for j in [num_spc_dates - 2, num_spc_dates - 1]:
                print('Writing new linkages to: "{0:s}"...'.format(
                    new_linkage_file_names[j]
                ))

                write_linkage_file(
                    storm_to_events_table=storm_to_events_table_by_date[j],
                    pickle_file_name=new_linkage_file_names[j]
                )

            break

        if i >= 2:
            print('Writing new linkages to: "{0:s}"...'.format(
                new_linkage_file_names[i - 2]
            ))

            write_linkage_file(
                storm_to_events_table=storm_to_events_table_by_date[i - 2],
                pickle_file_name=new_linkage_file_names[i - 2]
            )

            storm_to_events_table_by_date[i - 2] = pandas.DataFrame()

        for j in [i - 1, i, i + 1]:
            if j < 0 or j >= num_spc_dates:
                continue

            if not storm_to_events_table_by_date[j].empty:
                continue

            print('Reading original linkages from: "{0:s}"...'.format(
                orig_linkage_file_names[j]
            ))

            storm_to_events_table_by_date[j] = read_linkage_file(
                orig_linkage_file_names[j]
            )

        if i != num_spc_dates - 1:
            (storm_to_events_table_by_date[i],
             storm_to_events_table_by_date[i + 1]
            ) = _share_linkages_between_periods(
                early_storm_to_events_table=storm_to_events_table_by_date[i],
                late_storm_to_events_table=storm_to_events_table_by_date[i + 1]
            )

            print(SEPARATOR_STRING)


def find_linkage_file(
        top_directory_name, event_type_string, spc_date_string,
        raise_error_if_missing=True, unix_time_sec=None):
    """Finds linkage file for either one time or one SPC date.

    :param top_directory_name: Name of top-level directory with linkage files.
    :param event_type_string: Event type (must be accepted by
        `check_event_type`).
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :param unix_time_sec: Valid time.
    :return: linkage_file_name: Path to linkage file.  If file is missing and
        `raise_error_if_missing = False`, this will be the *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    check_event_type(event_type_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if event_type_string == WIND_EVENT_STRING:
        pathless_file_name_prefix = 'storm_to_winds'
    elif event_type_string == TORNADOGENESIS_EVENT_STRING:
        pathless_file_name_prefix = 'storm_to_tornadogenesis'
    else:
        pathless_file_name_prefix = 'storm_to_tornadoes'

    if unix_time_sec is None:
        time_conversion.spc_date_string_to_unix_sec(spc_date_string)

        linkage_file_name = '{0:s}/{1:s}/{2:s}_{3:s}.p'.format(
            top_directory_name, spc_date_string[:4], pathless_file_name_prefix,
            spc_date_string
        )
    else:
        spc_date_string = time_conversion.time_to_spc_date_string(unix_time_sec)
        valid_time_string = time_conversion.unix_sec_to_string(
            unix_time_sec, TIME_FORMAT)

        linkage_file_name = '{0:s}/{1:s}/{2:s}/{3:s}_{4:s}.p'.format(
            top_directory_name, spc_date_string[:4], spc_date_string,
            pathless_file_name_prefix, valid_time_string
        )

    if raise_error_if_missing and not os.path.isfile(linkage_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            linkage_file_name)
        raise ValueError(error_string)

    return linkage_file_name


def write_linkage_file(storm_to_events_table, pickle_file_name):
    """Writes linkages to Pickle file.

    :param storm_to_events_table: pandas DataFrame created by
        `_reverse_wind_linkages` or `_reverse_tornado_linkages`.
    :param pickle_file_name: Path to output file.
    """

    try:
        error_checking.assert_columns_in_dataframe(
            storm_to_events_table, REQUIRED_WIND_LINKAGE_COLUMNS)
    except:
        error_checking.assert_columns_in_dataframe(
            storm_to_events_table, REQUIRED_TORNADO_LINKAGE_COLUMNS)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_to_events_table, pickle_file_handle)
    pickle_file_handle.close()


def read_linkage_file(pickle_file_name):
    """Reads linkages from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_to_events_table: pandas DataFrame created by
        `_reverse_wind_linkages` or `_reverse_tornado_linkages`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_to_events_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    try:
        error_checking.assert_columns_in_dataframe(
            storm_to_events_table, REQUIRED_WIND_LINKAGE_COLUMNS)
    except:
        error_checking.assert_columns_in_dataframe(
            storm_to_events_table, REQUIRED_TORNADO_LINKAGE_COLUMNS)

    return storm_to_events_table
