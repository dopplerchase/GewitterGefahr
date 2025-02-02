"""Makes figure with sanity checks for saliency maps."""

import os
import pickle
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from PIL import Image
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.plotting import significance_plotting
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.scripts import plot_input_examples as plot_examples

# TODO(thunderhoser): A lot of this code is hacky and redundant with
# make_saliency_figure.py.

NONE_STRINGS = ['None', 'none']
POSSIBLE_MAX_COLOUR_VALUES = numpy.array([
    0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2
])

RADAR_HEIGHTS_M_AGL = numpy.array([2000, 6000, 10000], dtype=int)
# RADAR_FIELD_NAMES = [
#     radar_utils.REFL_NAME, radar_utils.VORTICITY_NAME,
#     radar_utils.SPECTRUM_WIDTH_NAME
# ]
RADAR_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.VORTICITY_NAME,
    radar_utils.DIVERGENCE_NAME
]

MIN_COLOUR_VALUE_LOG10 = -3.

COLOUR_BAR_LENGTH = 0.25
PANEL_NAME_FONT_SIZE = 30
COLOUR_BAR_FONT_SIZE = 25

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 150
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

SALIENCY_FILES_ARG_NAME = 'input_saliency_file_names'
MC_FILES_ARG_NAME = 'input_monte_carlo_file_names'
COMPOSITE_NAMES_ARG_NAME = 'composite_names'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_VALUES_ARG_NAME = 'max_colour_values'
HALF_NUM_CONTOURS_ARG_NAME = 'half_num_contours'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_grid_cells'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SALIENCY_FILES_HELP_STRING = (
    'List of saliency files (each will be read by `saliency.read_file`).'
)
MC_FILES_HELP_STRING = (
    'List of files with Monte Carlo significance (one per saliency file).  Each'
    ' will be read by `_read_monte_carlo_test`.  If you do not want to plot '
    'significance for the [i]th composite, make the [i]th list element "None".'
)
COMPOSITE_NAMES_HELP_STRING = (
    'List of composite names (one for each saliency file).  This list must be '
    'space-separated, but after reading the list, underscores within each item '
    'will be replaced by spaces.'
)
COLOUR_MAP_HELP_STRING = (
    'Colour scheme for saliency.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
MAX_VALUES_HELP_STRING = (
    'Max absolute saliency in each colour scheme (one per file).  Use negative '
    'values to let these be determined automatically.'
)
HALF_NUM_CONTOURS_HELP_STRING = (
    'Number of saliency contours on either side of zero (positive and '
    'negative).'
)
SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num grid cells).  If you do not '
    'want to smooth saliency maps, make this negative.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=SALIENCY_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MC_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=MC_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPOSITE_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=COMPOSITE_NAMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='binary',
    help=COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VALUES_ARG_NAME, type=float, nargs='+', required=True,
    help=MAX_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HALF_NUM_CONTOURS_ARG_NAME, type=int, required=False,
    default=10, help=HALF_NUM_CONTOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=1., help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_one_composite(saliency_file_name, smoothing_radius_grid_cells,
                        monte_carlo_file_name):
    """Reads saliency map for one composite.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    H = number of heights in grid
    F = number of radar fields

    :param saliency_file_name: Path to input file (will be read by
        `saliency.read_file`).
    :param smoothing_radius_grid_cells: Radius for Gaussian smoother, used only
        for saliency map.
    :param monte_carlo_file_name: Path to Monte Carlo file (will be read by
        `_read_monte_carlo_file`).
    :return: mean_radar_matrix: E-by-M-by-N-by-H-by-F numpy array with mean
        radar fields.
    :return: mean_saliency_matrix: E-by-M-by-N-by-H-by-F numpy array with mean
        saliency fields.
    :return: significance_matrix: E-by-M-by-N-by-H-by-F numpy array of Boolean
        flags.
    :return: model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    """

    print('Reading saliency maps from: "{0:s}"...'.format(saliency_file_name))
    saliency_dict = saliency_maps.read_file(saliency_file_name)[0]

    mean_radar_matrix = numpy.expand_dims(
        saliency_dict[saliency_maps.MEAN_PREDICTOR_MATRICES_KEY][0], axis=0
    )
    mean_saliency_matrix = numpy.expand_dims(
        saliency_dict[saliency_maps.MEAN_SALIENCY_MATRICES_KEY][0], axis=0
    )

    if smoothing_radius_grid_cells is not None:
        print((
            'Smoothing saliency maps with Gaussian filter (e-folding radius of '
            '{0:.1f} grid cells)...'
        ).format(
            smoothing_radius_grid_cells
        ))

        num_fields = mean_radar_matrix.shape[-1]

        for k in range(num_fields):
            mean_saliency_matrix[0, ..., k] = (
                general_utils.apply_gaussian_filter(
                    input_matrix=mean_saliency_matrix[0, ..., k],
                    e_folding_radius_grid_cells=smoothing_radius_grid_cells
                )
            )

    model_file_name = saliency_dict[saliency_maps.MODEL_FILE_KEY]
    model_metafile_name = cnn.find_metafile(model_file_name)

    if monte_carlo_file_name is None:
        significance_matrix = numpy.full(
            mean_radar_matrix.shape, False, dtype=bool
        )
    else:
        print('Reading Monte Carlo test from: "{0:s}"...'.format(
            monte_carlo_file_name
        ))

        this_file_handle = open(monte_carlo_file_name, 'rb')
        monte_carlo_dict = pickle.load(this_file_handle)
        this_file_handle.close()

        significance_matrix = numpy.logical_or(
            monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][0] <
            monte_carlo_dict[monte_carlo.MIN_MATRICES_KEY][0],
            monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][0] >
            monte_carlo_dict[monte_carlo.MAX_MATRICES_KEY][0]
        )
        significance_matrix = numpy.expand_dims(significance_matrix, axis=0)

    print('Fraction of significant differences: {0:.4f}'.format(
        numpy.mean(significance_matrix.astype(float))
    ))

    print('Reading CNN metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    good_indices = numpy.array([
        numpy.where(
            training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] == h
        )[0][0]
        for h in RADAR_HEIGHTS_M_AGL
    ], dtype=int)

    mean_radar_matrix = mean_radar_matrix[..., good_indices, :]
    mean_saliency_matrix = mean_saliency_matrix[..., good_indices, :]
    significance_matrix = significance_matrix[..., good_indices, :]

    good_indices = numpy.array([
        training_option_dict[trainval_io.RADAR_FIELDS_KEY].index(f)
        for f in RADAR_FIELD_NAMES
    ], dtype=int)

    mean_radar_matrix = mean_radar_matrix[..., good_indices]
    mean_saliency_matrix = mean_saliency_matrix[..., good_indices]
    significance_matrix = significance_matrix[..., good_indices]

    training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] = RADAR_HEIGHTS_M_AGL
    training_option_dict[trainval_io.RADAR_FIELDS_KEY] = RADAR_FIELD_NAMES
    training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] = None
    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY] = training_option_dict

    return (
        mean_radar_matrix, mean_saliency_matrix, significance_matrix,
        model_metadata_dict
    )


def _overlay_text(
        image_file_name, x_offset_from_center_px, y_offset_from_top_px,
        text_string):
    """Overlays text on image.

    :param image_file_name: Path to image file.
    :param x_offset_from_center_px: Center-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -gravity north -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_NAME,
        x_offset_from_center_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _plot_one_composite(
        saliency_file_name, monte_carlo_file_name, composite_name_abbrev,
        composite_name_verbose, colour_map_object, max_colour_value,
        half_num_contours, smoothing_radius_grid_cells, output_dir_name):
    """Plots saliency map for one composite.

    :param saliency_file_name: Path to saliency file (will be read by
        `saliency.read_file`).
    :param monte_carlo_file_name: Path to Monte Carlo file (will be read by
        `_read_monte_carlo_file`).
    :param composite_name_abbrev: Abbrev composite name (will be used in file
        names).
    :param composite_name_verbose: Verbose composite name (will be used in
        figure title).
    :param colour_map_object: See documentation at top of file.
    :param max_colour_value: Max value in colour bar (may be NaN).
    :param half_num_contours: See documentation at top of file.
    :param smoothing_radius_grid_cells: Same.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :return: main_figure_file_name: Path to main image file created by this
        method.
    :return: max_colour_value: Same as input but cannot be None.
    """

    (mean_radar_matrix, mean_saliency_matrix, significance_matrix,
     model_metadata_dict
    ) = _read_one_composite(
        saliency_file_name=saliency_file_name,
        smoothing_radius_grid_cells=smoothing_radius_grid_cells,
        monte_carlo_file_name=monte_carlo_file_name)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

    num_fields = mean_radar_matrix.shape[-1]
    num_heights = mean_radar_matrix.shape[-2]

    if numpy.isnan(max_colour_value):
        max_colour_value = numpy.percentile(
            numpy.absolute(mean_saliency_matrix[0, ...]), 99.
        )
        these_indices = numpy.where(
            max_colour_value <= POSSIBLE_MAX_COLOUR_VALUES
        )[0]

        if len(these_indices) == 0:
            this_index = -1
        else:
            this_index = these_indices[0]

        this_index -= 1
        max_colour_value = POSSIBLE_MAX_COLOUR_VALUES[this_index]

    handle_dict = plot_examples.plot_one_example(
        list_of_predictor_matrices=[mean_radar_matrix],
        model_metadata_dict=model_metadata_dict, pmm_flag=True,
        allow_whitespace=True, plot_panel_names=True,
        panel_name_font_size=PANEL_NAME_FONT_SIZE,
        add_titles=False, label_colour_bars=True,
        colour_bar_length=COLOUR_BAR_LENGTH,
        colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
        num_panel_rows=num_heights)

    figure_objects = handle_dict[plot_examples.RADAR_FIGURES_KEY]
    axes_object_matrices = handle_dict[plot_examples.RADAR_AXES_KEY]

    for k in range(num_fields):
        this_saliency_matrix = mean_saliency_matrix[0, ..., k]

        saliency_plotting.plot_many_2d_grids_with_contours(
            saliency_matrix_3d=numpy.flip(this_saliency_matrix, axis=0),
            axes_object_matrix=axes_object_matrices[k],
            colour_map_object=colour_map_object,
            max_absolute_contour_level=max_colour_value,
            contour_interval=max_colour_value / half_num_contours
        )

        this_sig_matrix = significance_matrix[0, ..., k]

        significance_plotting.plot_many_2d_grids_without_coords(
            significance_matrix=numpy.flip(this_sig_matrix, axis=0),
            axes_object_matrix=axes_object_matrices[k]
        )

    panel_file_names = [None] * num_fields

    for k in range(num_fields):
        panel_file_names[k] = '{0:s}/{1:s}_{2:s}.jpg'.format(
            output_dir_name, composite_name_abbrev,
            field_names[k].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(panel_file_names[k]))

        figure_objects[k].savefig(
            panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_objects[k])

    main_figure_file_name = '{0:s}/{1:s}_saliency.jpg'.format(
        output_dir_name, composite_name_abbrev)

    print('Concatenating panels to: "{0:s}"...'.format(main_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=main_figure_file_name,
        num_panel_rows=1, num_panel_columns=num_fields, border_width_pixels=50)

    imagemagick_utils.resize_image(
        input_file_name=main_figure_file_name,
        output_file_name=main_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)

    imagemagick_utils.trim_whitespace(
        input_file_name=main_figure_file_name,
        output_file_name=main_figure_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 25)

    _overlay_text(
        image_file_name=main_figure_file_name,
        x_offset_from_center_px=0, y_offset_from_top_px=0,
        text_string=composite_name_verbose)

    imagemagick_utils.trim_whitespace(
        input_file_name=main_figure_file_name,
        output_file_name=main_figure_file_name,
        border_width_pixels=10)

    return main_figure_file_name, max_colour_value


def _add_colour_bar(figure_file_name, colour_map_object, max_colour_value,
                    temporary_dir_name):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param max_colour_value: Max value in colour scheme.
    :param temporary_dir_name: Name of temporary output directory.
    """

    this_image_matrix = Image.open(figure_file_name)
    figure_width_px, figure_height_px = this_image_matrix.size
    figure_width_inches = float(figure_width_px) / FIGURE_RESOLUTION_DPI
    figure_height_inches = float(figure_height_px) / FIGURE_RESOLUTION_DPI

    extra_figure_object, extra_axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )
    extra_axes_object.axis('off')

    dummy_values = numpy.array([0., max_colour_value])

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=extra_axes_object, data_matrix=dummy_values,
        colour_map_object=colour_map_object,
        min_value=0., max_value=max_colour_value,
        orientation_string='vertical', fraction_of_axis_length=1.25,
        extend_min=False, extend_max=True, font_size=COLOUR_BAR_FONT_SIZE
    )

    colour_bar_object.set_label(
        'Absolute saliency', fontsize=COLOUR_BAR_FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()

    if max_colour_value <= 0.005:
        tick_strings = ['{0:.4f}'.format(v) for v in tick_values]
    elif max_colour_value <= 0.05:
        tick_strings = ['{0:.3f}'.format(v) for v in tick_values]
    else:
        tick_strings = ['{0:.2f}'.format(v) for v in tick_values]

    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    extra_file_name = '{0:s}/saliency_colour-bar.jpg'.format(temporary_dir_name)
    print('Saving colour bar to: "{0:s}"...'.format(extra_file_name))

    extra_figure_object.savefig(
        extra_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(extra_figure_object)

    print('Concatenating colour bar to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[figure_file_name, extra_file_name],
        output_file_name=figure_file_name,
        num_panel_rows=1, num_panel_columns=2,
        extra_args_string='-gravity Center'
    )

    os.remove(extra_file_name)
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name
    )


def _run(saliency_file_names, monte_carlo_file_names, composite_names,
         colour_map_name, max_colour_values, half_num_contours,
         smoothing_radius_grid_cells, output_dir_name):
    """Makes figure with sanity checks for saliency maps.

    This is effectively the main method.

    :param saliency_file_names: See documentation at top of file.
    :param monte_carlo_file_names: Same.
    :param composite_names: Same.
    :param colour_map_name: Same.
    :param max_colour_values: Same.
    :param half_num_contours: Same.
    :param smoothing_radius_grid_cells: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if smoothing_radius_grid_cells <= 0:
        smoothing_radius_grid_cells = None

    colour_map_object = pyplot.cm.get_cmap(colour_map_name)
    error_checking.assert_is_geq(half_num_contours, 5)

    num_composites = len(saliency_file_names)
    expected_dim = numpy.array([num_composites], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(composite_names), exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array(
        numpy.array(monte_carlo_file_names), exact_dimensions=expected_dim
    )

    monte_carlo_file_names = [
        None if f in NONE_STRINGS else f for f in monte_carlo_file_names
    ]

    max_colour_values[max_colour_values <= 0] = numpy.nan
    error_checking.assert_is_numpy_array(
        max_colour_values, exact_dimensions=expected_dim
    )

    composite_names_abbrev = [
        n.replace('_', '-').lower() for n in composite_names
    ]
    composite_names_verbose = [
        '({0:s}) {1:s}'.format(
            chr(ord('a') + i), composite_names[i].replace('_', ' ')
        )
        for i in range(num_composites)
    ]

    panel_file_names = [None] * num_composites

    for i in range(num_composites):
        panel_file_names[i], max_colour_values[i] = _plot_one_composite(
            saliency_file_name=saliency_file_names[i],
            monte_carlo_file_name=monte_carlo_file_names[i],
            composite_name_abbrev=composite_names_abbrev[i],
            composite_name_verbose=composite_names_verbose[i],
            colour_map_object=colour_map_object,
            max_colour_value=max_colour_values[i],
            half_num_contours=half_num_contours,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells,
            output_dir_name=output_dir_name
        )

        _add_colour_bar(
            figure_file_name=panel_file_names[i],
            colour_map_object=colour_map_object,
            max_colour_value=max_colour_values[i],
            temporary_dir_name=output_dir_name
        )

        print('\n')

    figure_file_name = '{0:s}/saliency_concat.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(figure_file_name))

    num_panel_rows = int(numpy.ceil(
        numpy.sqrt(num_composites)
    ))
    num_panel_columns = int(numpy.floor(
        float(num_composites) / num_panel_rows
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=figure_file_name, border_width_pixels=100,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name,
        border_width_pixels=10
    )
    imagemagick_utils.resize_image(
        input_file_name=figure_file_name, output_file_name=figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        saliency_file_names=getattr(INPUT_ARG_OBJECT, SALIENCY_FILES_ARG_NAME),
        monte_carlo_file_names=getattr(INPUT_ARG_OBJECT, MC_FILES_ARG_NAME),
        composite_names=getattr(INPUT_ARG_OBJECT, COMPOSITE_NAMES_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_VALUES_ARG_NAME), dtype=float
        ),
        half_num_contours=getattr(INPUT_ARG_OBJECT, HALF_NUM_CONTOURS_ARG_NAME),
        smoothing_radius_grid_cells=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
