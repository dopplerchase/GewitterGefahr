"""Downloads zero-hour RUC (Rapid Update Cycle) analyses to local machine."""

import time
import argparse
import numpy
from gewittergefahr.gg_io import nwp_model_io
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods

INPUT_TIME_FORMAT = '%Y%m%d%H'
DEFAULT_TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600
PAUSE_BETWEEN_FILES_IN_SECONDS = 5

FIRST_TIME_INPUT_ARG = 'first_time_string'
LAST_TIME_INPUT_ARG = 'last_time_string'
TARGET_DIR_INPUT_ARG = 'target_directory_name'

TIME_HELP_STRING = (
    'RUC-initialization time in format "yyyymmddHH".  This script downloads '
    'zero-hour forecasts for all hours from `{0:s}`...`{1:s}`.').format(
        FIRST_TIME_INPUT_ARG, LAST_TIME_INPUT_ARG)
TARGET_DIR_HELP_STRING = (
    'Name of top-level target directory (on local machine).  RUC files will be '
    'downloaded here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_INPUT_ARG, type=str, required=True, help=TIME_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_INPUT_ARG, type=str, required=True, help=TIME_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_INPUT_ARG, type=str, required=True,
    help=TARGET_DIR_HELP_STRING)


def _download_ruc_analyses(
        first_time_string, last_time_string, top_target_dir_name):
    """Downloads zero-hour RUC (Rapid Update Cycle) analyses to local machine.
    
    :param first_time_string: RUC-initialization time in format "yyyymmddHH".
        This script downloads zero-hour forecasts for all hours from
        `first_time_string`...`last_time_string`.
    :param last_time_string: See above.
    :param top_target_dir_name: Name of top-level target directory (on local
        machine).  RUC files will be downloaded here.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)
    time_interval_sec = HOURS_TO_SECONDS * nwp_model_utils.get_time_steps(
        nwp_model_utils.RUC_MODEL_NAME)[1]

    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=time_interval_sec)

    num_times = len(init_times_unix_sec)
    local_file_names = [None] * num_times

    for i in range(num_times):
        local_file_names[i] = nwp_model_io.find_ruc_grib_file(
            init_times_unix_sec[i], lead_time_hours=0,
            top_directory_name=top_target_dir_name,
            raise_error_if_missing=False)
        if local_file_names[i] is not None:
            continue

        local_file_names[i] = nwp_model_io.download_ruc_grib_file(
            init_times_unix_sec[i], lead_time_hours=0,
            top_local_directory_name=top_target_dir_name,
            raise_error_if_fails=False)

        if local_file_names[i] is None:
            this_init_time_string = time_conversion.unix_sec_to_string(
                init_times_unix_sec[i], DEFAULT_TIME_FORMAT)

            print('\nPROBLEM.  Download failed for {0:s}.\n\n'.format(
                this_init_time_string))
        else:
            print('\nSUCCESS.  File was downloaded to "{0:s}".\n\n'.format(
                local_file_names[i]))

        time.sleep(PAUSE_BETWEEN_FILES_IN_SECONDS)

    num_downloaded = numpy.sum(numpy.array(
        [f is not None for f in local_file_names]))
    print('{0:d} of {1:d} files were downloaded successfully!'.format(
        num_downloaded, num_times))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    FIRST_TIME_STRING = getattr(INPUT_ARG_OBJECT, FIRST_TIME_INPUT_ARG)
    LAST_TIME_STRING = getattr(INPUT_ARG_OBJECT, LAST_TIME_INPUT_ARG)
    TOP_TARGET_DIR_NAME = getattr(INPUT_ARG_OBJECT, TARGET_DIR_INPUT_ARG)

    _download_ruc_analyses(
        first_time_string=FIRST_TIME_STRING, last_time_string=LAST_TIME_STRING,
        top_target_dir_name=TOP_TARGET_DIR_NAME)
