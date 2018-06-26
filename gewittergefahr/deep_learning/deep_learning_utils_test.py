"""Unit tests for deep_learning_utils.py"""

import copy
import unittest
import numpy
import keras
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings_only
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

TOLERANCE = 1e-6
TOLERANCE_FOR_CLASS_WEIGHT = 1e-3

# The following constants are used to test class_fractions_to_num_examples.
SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT = {0: 0.1, 1: 0.9}
SAMPLING_FRACTION_BY_WIND_3CLASS_DICT = {-2: 0.1, 0: 0.2, 1: 0.7}
WIND_TARGET_NAME_3CLASSES = (
    'wind-speed_percentile=100.0_lead-time=1800-3600sec_distance=00000-10000m'
    '_cutoffs=50kt')

NUM_EXAMPLES_LARGE = 17
NUM_EXAMPLES_LARGE_BY_TORNADO_CLASS_DICT = {0: 2, 1: 15}
NUM_EXAMPLES_LARGE_BY_WIND_3CLASS_DICT = {-2: 2, 0: 3, 1: 12}

NUM_EXAMPLES_MEDIUM = 8
NUM_EXAMPLES_MEDIUM_BY_TORNADO_CLASS_DICT = {0: 1, 1: 7}
NUM_EXAMPLES_MEDIUM_BY_WIND_3CLASS_DICT = {-2: 1, 0: 2, 1: 5}

NUM_EXAMPLES_SMALL = 4
NUM_EXAMPLES_SMALL_BY_TORNADO_CLASS_DICT = {0: 1, 1: 3}
NUM_EXAMPLES_SMALL_BY_WIND_3CLASS_DICT = {-2: 1, 0: 1, 1: 2}

NUM_EXAMPLES_XSMALL = 3
NUM_EXAMPLES_XSMALL_BY_TORNADO_CLASS_DICT = {0: 1, 1: 2}
NUM_EXAMPLES_XSMALL_BY_WIND_3CLASS_DICT = {-2: 1, 0: 1, 1: 1}

# The following constants are used to test class_fractions_to_weights.
LF_WEIGHT_BY_TORNADO_CLASS_DICT = {0: 0.9, 1: 0.1}
LF_WEIGHT_BY_WIND_3CLASS_DICT = {0: 0.7, 1: 0.3}

SAMPLING_FRACTION_BY_WIND_7CLASS_DICT = {
    -2: 0.1, 0: 0.4, 1: 0.2, 2: 0.1, 3: 0.05, 4: 0.05, 5: 0.1}
WIND_TARGET_NAME_7CLASSES = (
    'wind-speed_percentile=100.0_lead-time=1800-3600sec_distance=00000-10000m'
    '_cutoffs=10-20-30-40-50kt')

LF_WEIGHT_BY_WIND_7CLASS_DICT = {
    0: 2. / 67, 1: 5. / 67, 2: 10. / 67, 3: 20. / 67, 4: 20. / 67, 5: 10. / 67}
LF_WEIGHT_BY_WIND_7CLASS_DICT_BINARIZED = {0: 0.1, 1: 0.9}

# The following constants are used to test check_radar_images,
# stack_radar_fields, and stack_radar_heights.
RADAR_IMAGE_MATRIX_1D = numpy.array([1, 2, 3, 4], dtype=numpy.float32)
RADAR_IMAGE_MATRIX_2D = numpy.array([[1, 2, 3, 4],
                                     [5, 6, 7, 8],
                                     [9, 10, 11, 12]], dtype=numpy.float32)

RADAR_IMAGE_MATRIX_3D = numpy.stack(
    (RADAR_IMAGE_MATRIX_2D, RADAR_IMAGE_MATRIX_2D), axis=0)

TUPLE_OF_3D_RADAR_MATRICES = (
    RADAR_IMAGE_MATRIX_3D, RADAR_IMAGE_MATRIX_3D, RADAR_IMAGE_MATRIX_3D,
    RADAR_IMAGE_MATRIX_3D, RADAR_IMAGE_MATRIX_3D, RADAR_IMAGE_MATRIX_3D)
RADAR_IMAGE_MATRIX_4D = numpy.stack(TUPLE_OF_3D_RADAR_MATRICES, axis=-1)

TUPLE_OF_4D_RADAR_MATRICES = (
    RADAR_IMAGE_MATRIX_4D, RADAR_IMAGE_MATRIX_4D, RADAR_IMAGE_MATRIX_4D,
    RADAR_IMAGE_MATRIX_4D, RADAR_IMAGE_MATRIX_4D)
RADAR_IMAGE_MATRIX_5D = numpy.stack(TUPLE_OF_4D_RADAR_MATRICES, axis=-2)

# The following constants are used to test check_target_values.
TORNADO_CLASSES_1D = numpy.array(
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], dtype=int)
WIND_CLASSES_1D = numpy.array(
    [1, 2, 3, 4, 0, 2, 0, 1, 1, 5, 2], dtype=int)

TORNADO_CLASS_MATRIX = keras.utils.to_categorical(TORNADO_CLASSES_1D, 2)
WIND_CLASS_MATRIX = keras.utils.to_categorical(
    WIND_CLASSES_1D, numpy.max(WIND_CLASSES_1D) + 1)

# The following constants are used to test normalize_radar_images.
PERCENTILE_OFFSET_FOR_NORMALIZATION = 0.
RADAR_FIELD_NAMES = [radar_utils.REFL_NAME, radar_utils.DIFFERENTIAL_REFL_NAME]
RADAR_NORMALIZATION_DICT = {
    radar_utils.DIFFERENTIAL_REFL_NAME: numpy.array([-8., 8.]),
    radar_utils.REFL_NAME: numpy.array([1., 10.])
}

REFL_MATRIX_EXAMPLE1_HEIGHT1 = numpy.array(
    [[0, 1, 2, 3],
     [4, 5, 6, 7]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE1_HEIGHT2 = numpy.array(
    [[2, 4, 6, 8],
     [10, 12, 14, 16]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE1_HEIGHT3 = numpy.array(
    [[-5, 2, 0, 5],
     [3, -3, 11, 10]], dtype=numpy.float32)

REFL_MATRIX_EXAMPLE2_HEIGHT1 = numpy.array(
    [[3, 2, 1, 2],
     [6, 10, 16, -6]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE2_HEIGHT2 = numpy.array(
    [[0, 0, 0, 0],
     [0, 1, 1, 2]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE2_HEIGHT3 = numpy.array(
    [[17, 7, 0, 3],
     [6, 7, 8, 4]], dtype=numpy.float32)

DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1 = -1 * REFL_MATRIX_EXAMPLE1_HEIGHT1
DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2 = -1 * REFL_MATRIX_EXAMPLE1_HEIGHT2
DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3 = -1 * REFL_MATRIX_EXAMPLE1_HEIGHT3
DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1 = -1 * REFL_MATRIX_EXAMPLE2_HEIGHT1
DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2 = -1 * REFL_MATRIX_EXAMPLE2_HEIGHT2
DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3 = -1 * REFL_MATRIX_EXAMPLE2_HEIGHT3

RADAR_MATRIX_EXAMPLE1_HEIGHT1_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE1_HEIGHT1, DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1), axis=-1)
RADAR_MATRIX_EXAMPLE2_HEIGHT1_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE2_HEIGHT1, DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1), axis=-1)
RADAR_MATRIX_4D_UNNORMALIZED = numpy.stack(
    (RADAR_MATRIX_EXAMPLE1_HEIGHT1_UNNORMALIZED,
     RADAR_MATRIX_EXAMPLE2_HEIGHT1_UNNORMALIZED), axis=0)

RADAR_MATRIX_EXAMPLE1_HEIGHT1_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT1 + 6) / 22,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1 + 16) / 22), axis=-1)
RADAR_MATRIX_EXAMPLE2_HEIGHT1_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT1 + 6) / 22,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1 + 16) / 22), axis=-1)
RADAR_MATRIX_4D_NORM_BY_BATCH = numpy.stack(
    (RADAR_MATRIX_EXAMPLE1_HEIGHT1_NORM_BY_BATCH,
     RADAR_MATRIX_EXAMPLE2_HEIGHT1_NORM_BY_BATCH), axis=0)

RADAR_MATRIX_EXAMPLE1_HEIGHT1_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT1 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1 + 8) / 16), axis=-1)
RADAR_MATRIX_EXAMPLE2_HEIGHT1_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT1 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1 + 8) / 16), axis=-1)
RADAR_MATRIX_4D_NORM_BY_CLIMO = numpy.stack(
    (RADAR_MATRIX_EXAMPLE1_HEIGHT1_NORM_BY_CLIMO,
     RADAR_MATRIX_EXAMPLE2_HEIGHT1_NORM_BY_CLIMO), axis=0)

RADAR_MATRIX_EXAMPLE1_HEIGHT2_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE1_HEIGHT2, DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2), axis=-1)
RADAR_MATRIX_EXAMPLE2_HEIGHT2_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE2_HEIGHT2, DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2), axis=-1)
RADAR_MATRIX_HEIGHT2_UNNORMALIZED = numpy.stack(
    (RADAR_MATRIX_EXAMPLE1_HEIGHT2_UNNORMALIZED,
     RADAR_MATRIX_EXAMPLE2_HEIGHT2_UNNORMALIZED), axis=0)

RADAR_MATRIX_EXAMPLE1_HEIGHT3_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE1_HEIGHT3, DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3), axis=-1)
RADAR_MATRIX_EXAMPLE2_HEIGHT3_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE2_HEIGHT3, DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3), axis=-1)
RADAR_MATRIX_HEIGHT3_UNNORMALIZED = numpy.stack(
    (RADAR_MATRIX_EXAMPLE1_HEIGHT3_UNNORMALIZED,
     RADAR_MATRIX_EXAMPLE2_HEIGHT3_UNNORMALIZED), axis=0)

RADAR_MATRIX_5D_UNNORMALIZED = numpy.stack(
    (RADAR_MATRIX_4D_UNNORMALIZED, RADAR_MATRIX_HEIGHT2_UNNORMALIZED,
     RADAR_MATRIX_HEIGHT3_UNNORMALIZED), axis=-2)

RADAR_MATRIX_EXAMPLE1_HEIGHT1_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT1 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1 + 17) / 23), axis=-1)
RADAR_MATRIX_EXAMPLE2_HEIGHT1_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT1 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1 + 17) / 23), axis=-1)
RADAR_MATRIX_HEIGHT1_NORM_BY_BATCH = numpy.stack(
    (RADAR_MATRIX_EXAMPLE1_HEIGHT1_NORM_BY_BATCH,
     RADAR_MATRIX_EXAMPLE2_HEIGHT1_NORM_BY_BATCH), axis=0)

RADAR_MATRIX_EXAMPLE1_HEIGHT2_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT2 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2 + 17) / 23), axis=-1)
RADAR_MATRIX_EXAMPLE2_HEIGHT2_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT2 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2 + 17) / 23), axis=-1)
RADAR_MATRIX_HEIGHT2_NORM_BY_BATCH = numpy.stack(
    (RADAR_MATRIX_EXAMPLE1_HEIGHT2_NORM_BY_BATCH,
     RADAR_MATRIX_EXAMPLE2_HEIGHT2_NORM_BY_BATCH), axis=0)

RADAR_MATRIX_EXAMPLE1_HEIGHT3_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT3 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3 + 17) / 23), axis=-1)
RADAR_MATRIX_EXAMPLE2_HEIGHT3_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT3 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3 + 17) / 23), axis=-1)
RADAR_MATRIX_HEIGHT3_NORM_BY_BATCH = numpy.stack(
    (RADAR_MATRIX_EXAMPLE1_HEIGHT3_NORM_BY_BATCH,
     RADAR_MATRIX_EXAMPLE2_HEIGHT3_NORM_BY_BATCH), axis=0)

RADAR_MATRIX_5D_NORM_BY_BATCH = numpy.stack(
    (RADAR_MATRIX_HEIGHT1_NORM_BY_BATCH, RADAR_MATRIX_HEIGHT2_NORM_BY_BATCH,
     RADAR_MATRIX_HEIGHT3_NORM_BY_BATCH), axis=-2)

RADAR_MATRIX_EXAMPLE1_HEIGHT2_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT2 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2 + 8) / 16), axis=-1)
RADAR_MATRIX_EXAMPLE2_HEIGHT2_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT2 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2 + 8) / 16), axis=-1)
RADAR_MATRIX_HEIGHT2_NORM_BY_CLIMO = numpy.stack(
    (RADAR_MATRIX_EXAMPLE1_HEIGHT2_NORM_BY_CLIMO,
     RADAR_MATRIX_EXAMPLE2_HEIGHT2_NORM_BY_CLIMO), axis=0)

RADAR_MATRIX_EXAMPLE1_HEIGHT3_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT3 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3 + 8) / 16), axis=-1)
RADAR_MATRIX_EXAMPLE2_HEIGHT3_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT3 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3 + 8) / 16), axis=-1)
RADAR_MATRIX_HEIGHT3_NORM_BY_CLIMO = numpy.stack(
    (RADAR_MATRIX_EXAMPLE1_HEIGHT3_NORM_BY_CLIMO,
     RADAR_MATRIX_EXAMPLE2_HEIGHT3_NORM_BY_CLIMO), axis=0)

RADAR_MATRIX_5D_NORM_BY_CLIMO = numpy.stack(
    (RADAR_MATRIX_4D_NORM_BY_CLIMO, RADAR_MATRIX_HEIGHT2_NORM_BY_CLIMO,
     RADAR_MATRIX_HEIGHT3_NORM_BY_CLIMO), axis=-2)

# The following constants are used to test normalize_soundings.
SOUNDING_FIELD_NAMES = [
    soundings_only.RELATIVE_HUMIDITY_NAME, soundings_only.TEMPERATURE_NAME,
    soundings_only.U_WIND_NAME, soundings_only.V_WIND_NAME,
    soundings_only.SPECIFIC_HUMIDITY_NAME,
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME
]

THIS_FIRST_MATRIX = numpy.array([[0.9, 300., -10., 5., 0.02, 310.],
                                 [0.7, 285., 0., 15., 0.015, 310.],
                                 [0.95, 270., 15., 20., 0.01, 325.],
                                 [0.93, 260., 30., 30., 0.007, 341.]])
THIS_SECOND_MATRIX = numpy.array([[0.7, 305., 0., 0., 0.015, 312.5],
                                  [0.5, 290., 15., 12.5, 0.015, 310.],
                                  [0.8, 273., 25., 15., 0.012, 333.],
                                  [0.9, 262., 40., 20., 0.008, 345.]])
SOUNDING_MATRIX_UNNORMALIZED = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)

SOUNDING_NORMALIZATION_DICT = {
    soundings_only.RELATIVE_HUMIDITY_NAME: numpy.array([0., 1.]),
    soundings_only.TEMPERATURE_NAME: numpy.array([250., 300.]),
    soundings_only.WIND_SPEED_KEY: numpy.array([0., 50.]),
    soundings_only.SPECIFIC_HUMIDITY_NAME: numpy.array([0., 0.02]),
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME:
        numpy.array([300., 350.])
}

THIS_FIRST_MATRIX = numpy.array([[0.9, 1., -0.2, 0.1, 1., 0.2],
                                 [0.7, 0.7, 0., 0.3, 0.75, 0.2],
                                 [0.95, 0.4, 0.3, 0.4, 0.5, 0.5],
                                 [0.93, 0.2, 0.6, 0.6, 0.35, 0.82]])

THIS_SECOND_MATRIX = numpy.array([[0.7, 1.1, 0., 0., 0.75, 0.25],
                                  [0.5, 0.8, 0.3, 0.25, 0.75, 0.2],
                                  [0.8, 0.46, 0.5, 0.3, 0.6, 0.66],
                                  [0.9, 0.24, 0.8, 0.4, 0.4, 0.9]])

SOUNDING_MATRIX_NORMALIZED = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)

# The following constants are used to test sample_by_class.
TORNADO_LABELS_TO_SAMPLE = numpy.array(
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    dtype=int)
TORNADO_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

NUM_EXAMPLES_TOTAL = 30
BALANCED_FRACTION_BY_TORNADO_CLASS_DICT = {0: 0.5, 1: 0.5}
TORNADO_INDICES_TO_KEEP = numpy.array(
    [0, 1, 2, 4, 7, 8, 9, 10, 11, 13, 3, 5, 6, 12, 15, 16, 19, 25, 45, 48],
    dtype=int)

WIND_LABELS_TO_SAMPLE = numpy.array(
    [2, 1, 1, 2, 0, -2, -2, 1, -2, 2, 2, 1, 2, 1, 0, 0, 2, 1, 1, 0, 0, -2, 0, 1,
     2, 2, 2, 0, -2, 0, 0, 2, -2, 2, -2, 0, -2, 0, 1, -2, 2, -2, 2, 1, 1, 1, 0,
     0, 0, 1], dtype=int)
WIND_TARGET_NAME_4CLASSES = (
    'wind-speed_percentile=100.0_lead-time=1800-3600sec_distance=00000-10000m'
    '_cutoffs=30-50kt')

SAMPLING_FRACTION_BY_WIND_4CLASS_DICT = {-2: 0.2, 0: 0.1, 1: 0.4, 2: 0.3}
WIND_INDICES_TO_KEEP = numpy.array(
    [4, 14, 15, 1, 2, 7, 11, 13, 17, 18, 23, 38, 43, 44, 45, 0, 3, 9, 10, 12,
     16, 24, 25, 26, 5, 6, 8, 21, 28, 32], dtype=int)


class DeepLearningUtilsTests(unittest.TestCase):
    """Each method is a unit test for deep_learning_utils.py."""

    def test_class_fractions_to_num_examples_tornado_large(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = tornado occurrence and number of
        examples to draw is large.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME,
            num_examples_total=NUM_EXAMPLES_LARGE)

        self.assertTrue(this_dict == NUM_EXAMPLES_LARGE_BY_TORNADO_CLASS_DICT)

    def test_class_fractions_to_num_examples_wind_large(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = wind-speed class and number of examples
        to draw is large.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES,
            num_examples_total=NUM_EXAMPLES_LARGE)

        self.assertTrue(this_dict == NUM_EXAMPLES_LARGE_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_num_examples_tornado_medium(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = tornado occurrence and number of
        examples to draw is medium.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME,
            num_examples_total=NUM_EXAMPLES_MEDIUM)

        self.assertTrue(this_dict == NUM_EXAMPLES_MEDIUM_BY_TORNADO_CLASS_DICT)

    def test_class_fractions_to_num_examples_wind_medium(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = wind-speed class and number of examples
        to draw is medium.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES,
            num_examples_total=NUM_EXAMPLES_MEDIUM)

        self.assertTrue(this_dict == NUM_EXAMPLES_MEDIUM_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_num_examples_tornado_small(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = tornado occurrence and number of
        examples to draw is small.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME,
            num_examples_total=NUM_EXAMPLES_SMALL)

        self.assertTrue(this_dict == NUM_EXAMPLES_SMALL_BY_TORNADO_CLASS_DICT)

    def test_class_fractions_to_num_examples_wind_small(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = wind-speed class and number of examples
        to draw is small.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES,
            num_examples_total=NUM_EXAMPLES_SMALL)

        self.assertTrue(this_dict == NUM_EXAMPLES_SMALL_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_num_examples_tornado_xsmall(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = tornado occurrence and number of
        examples to draw is extra small.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME,
            num_examples_total=NUM_EXAMPLES_XSMALL)

        self.assertTrue(this_dict == NUM_EXAMPLES_XSMALL_BY_TORNADO_CLASS_DICT)

    def test_class_fractions_to_num_examples_wind_xsmall(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = wind-speed class and number of examples
        to draw is extra small.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES,
            num_examples_total=NUM_EXAMPLES_XSMALL)

        self.assertTrue(this_dict == NUM_EXAMPLES_XSMALL_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_weights_tornado_nonbinarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = tornado occurrence and binarize_target =
        False.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME, binarize_target=False)

        self.assertTrue(this_dict == LF_WEIGHT_BY_TORNADO_CLASS_DICT)

    def test_class_fractions_to_weights_tornado_binarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = tornado occurrence and binarize_target =
        True.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME, binarize_target=True)

        expected_keys = LF_WEIGHT_BY_TORNADO_CLASS_DICT.keys()
        actual_keys = this_dict.keys()
        self.assertTrue(set(expected_keys) == set(actual_keys))

        for this_key in expected_keys:
            self.assertTrue(numpy.isclose(
                LF_WEIGHT_BY_TORNADO_CLASS_DICT[this_key], this_dict[this_key],
                atol=TOLERANCE_FOR_CLASS_WEIGHT))

    def test_class_fractions_to_weights_wind_3class_nonbinarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = wind-speed class; there are 3 classes;
        and binarize_target = False.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES, binarize_target=False)

        self.assertTrue(this_dict == LF_WEIGHT_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_weights_wind_3class_binarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = wind-speed class; there are 3 classes;
        and binarize_target = True.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES, binarize_target=True)

        self.assertTrue(this_dict == LF_WEIGHT_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_weights_wind_7class_nonbinarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = wind-speed class; there are 7 classes;
        and binarize_target = False.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_7CLASS_DICT,
            target_name=WIND_TARGET_NAME_7CLASSES, binarize_target=False)

        self.assertTrue(this_dict == LF_WEIGHT_BY_WIND_7CLASS_DICT)

    def test_class_fractions_to_weights_wind_7class_binarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = wind-speed class; there are 7 classes;
        and binarize_target = True.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_7CLASS_DICT,
            target_name=WIND_TARGET_NAME_7CLASSES, binarize_target=True)

        self.assertTrue(this_dict == LF_WEIGHT_BY_WIND_7CLASS_DICT_BINARIZED)

    def test_check_radar_images_1d(self):
        """Ensures correct output from check_radar_images.

        In this case the input matrix is 1-D (bad).
        """

        with self.assertRaises(ValueError):
            dl_utils.check_radar_images(
                radar_image_matrix=RADAR_IMAGE_MATRIX_1D)

    def test_check_radar_images_2d(self):
        """Ensures correct output from check_radar_images.

        In this case the input matrix is 2-D (bad).
        """

        with self.assertRaises(ValueError):
            dl_utils.check_radar_images(
                radar_image_matrix=RADAR_IMAGE_MATRIX_2D)

    def test_check_radar_images_3d_good(self):
        """Ensures correct output from check_radar_images.

        In this case the input matrix is 3-D (good).
        """

        dl_utils.check_radar_images(radar_image_matrix=RADAR_IMAGE_MATRIX_3D)

    def test_check_radar_images_3d_bad(self):
        """Ensures correct output from check_radar_images.

        In this case, the input matrix is 3-D but a 4-D or 5-D matrix is
        expected.
        """

        with self.assertRaises(ValueError):
            dl_utils.check_radar_images(
                radar_image_matrix=RADAR_IMAGE_MATRIX_3D, min_num_dimensions=4)

    def test_check_radar_images_4d(self):
        """Ensures correct output from check_radar_images.

        In this case the input matrix is 4-D (good).
        """

        dl_utils.check_radar_images(radar_image_matrix=RADAR_IMAGE_MATRIX_4D)

    def test_check_radar_images_5d_good(self):
        """Ensures correct output from check_radar_images.

        In this case the input matrix is 5-D (good).
        """

        dl_utils.check_radar_images(radar_image_matrix=RADAR_IMAGE_MATRIX_5D)

    def test_check_radar_images_5d_bad(self):
        """Ensures correct output from check_radar_images.

        In this case, the input matrix is 5-D but a 3-D or 4-D matrix is
        expected.
        """

        with self.assertRaises(ValueError):
            dl_utils.check_radar_images(
                radar_image_matrix=RADAR_IMAGE_MATRIX_5D, max_num_dimensions=4)

    def test_check_target_values_1d_binary_good(self):
        """Ensures correct output from check_target_values.

        In this case the input array is 1-D and contains 2 classes, as expected.
        """

        dl_utils.check_target_values(
            target_values=TORNADO_CLASSES_1D, num_dimensions=1, num_classes=2)

    def test_check_target_values_1d_binary_bad_dim(self):
        """Ensures correct output from check_target_values.

        In this case, the input array is 1-D but a 2-D array is expected.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_values(
                target_values=TORNADO_CLASSES_1D, num_dimensions=2,
                num_classes=2)

    def test_check_target_values_1d_binary_bad_class_num(self):
        """Ensures correct output from check_target_values.

        In this case, 6 classes are expected and the input array contains only 2
        classes.  However, there is no way to ascertain that the 2-class array
        is wrong (maybe higher classes just did not occur in the sample).
        """

        dl_utils.check_target_values(
            target_values=TORNADO_CLASSES_1D, num_dimensions=1,
            num_classes=WIND_CLASS_MATRIX.shape[1])

    def test_check_target_values_1d_multiclass_good(self):
        """Ensures correct output from check_target_values.

        In this case the input array is 1-D and multiclass, as expected.
        """

        dl_utils.check_target_values(
            target_values=WIND_CLASSES_1D, num_dimensions=1,
            num_classes=WIND_CLASS_MATRIX.shape[1])

    def test_check_target_values_1d_multiclass_bad_class_num(self):
        """Ensures correct output from check_target_values.

        In this case, the input array contains 6 classes but only 2 classes are
        expected.
        """

        with self.assertRaises(ValueError):
            dl_utils.check_target_values(
                target_values=WIND_CLASSES_1D, num_dimensions=1, num_classes=2)

    def test_check_target_values_2d_binary_good(self):
        """Ensures correct output from check_target_values.

        In this case the input array is 2-D and contains 2 classes, as expected.
        """

        dl_utils.check_target_values(
            target_values=TORNADO_CLASS_MATRIX, num_dimensions=2, num_classes=2)

    def test_check_target_values_2d_bad_dim(self):
        """Ensures correct output from check_target_values.

        In this case, the input array is 2-D but a 1-D array is expected.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_values(
                target_values=TORNADO_CLASS_MATRIX, num_dimensions=1,
                num_classes=2)

    def test_check_target_values_2d_binary_bad_class_num(self):
        """Ensures correct output from check_target_values.

        In this case, the input array contains 2 classes but 6 classes are
        expected.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_values(
                target_values=TORNADO_CLASS_MATRIX, num_dimensions=2,
                num_classes=WIND_CLASS_MATRIX.shape[1])

    def test_check_target_values_2d_multiclass_good(self):
        """Ensures correct output from check_target_values.

        In this case the input array is 2-D and multiclass, as expected.
        """

        dl_utils.check_target_values(
            target_values=WIND_CLASS_MATRIX, num_dimensions=2,
            num_classes=WIND_CLASS_MATRIX.shape[1])

    def test_check_target_values_2d_multiclass_bad_class_num(self):
        """Ensures correct output from check_target_values.

        In this case, the input array contains 6 classes but 2 classes are
        expected.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_values(
                target_values=WIND_CLASS_MATRIX, num_dimensions=2,
                num_classes=2)

    def test_stack_radar_fields(self):
        """Ensures correct output from stack_radar_fields."""

        this_matrix = dl_utils.stack_radar_fields(TUPLE_OF_3D_RADAR_MATRICES)
        self.assertTrue(numpy.allclose(
            this_matrix, RADAR_IMAGE_MATRIX_4D, atol=TOLERANCE, equal_nan=True))

    def test_stack_radar_heights(self):
        """Ensures correct output from stack_radar_heights."""

        this_matrix = dl_utils.stack_radar_heights(TUPLE_OF_4D_RADAR_MATRICES)
        self.assertTrue(numpy.allclose(
            this_matrix, RADAR_IMAGE_MATRIX_5D, atol=TOLERANCE, equal_nan=True))

    def test_normalize_radar_images_4d_by_batch(self):
        """Ensures correct output from normalize_radar_images.

        In this case a 4-D matrix is normalized by batch.
        """

        this_radar_matrix = dl_utils.normalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_4D_UNNORMALIZED),
            normalize_by_batch=True,
            percentile_offset=PERCENTILE_OFFSET_FOR_NORMALIZATION)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_NORM_BY_BATCH,
            atol=TOLERANCE, equal_nan=True))

    def test_normalize_radar_images_4d_by_climo(self):
        """Ensures correct output from normalize_radar_images.

        In this case a 4-D matrix is normalized by climatology.
        """

        this_radar_matrix = dl_utils.normalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_4D_UNNORMALIZED),
            normalize_by_batch=False, field_names=RADAR_FIELD_NAMES,
            normalization_dict=RADAR_NORMALIZATION_DICT)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_NORM_BY_CLIMO,
            atol=TOLERANCE, equal_nan=True))

    def test_normalize_radar_images_5d_by_batch(self):
        """Ensures correct output from normalize_radar_images.

        In this case a 5-D matrix is normalized by batch.
        """

        this_radar_matrix = dl_utils.normalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_5D_UNNORMALIZED),
            normalize_by_batch=True,
            percentile_offset=PERCENTILE_OFFSET_FOR_NORMALIZATION)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_NORM_BY_BATCH,
            atol=TOLERANCE, equal_nan=True))

    def test_normalize_radar_images_5d_by_climo(self):
        """Ensures correct output from normalize_radar_images.

        In this case a 5-D matrix is normalized by climatology.
        """

        this_radar_matrix = dl_utils.normalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_5D_UNNORMALIZED),
            normalize_by_batch=False, field_names=RADAR_FIELD_NAMES,
            normalization_dict=RADAR_NORMALIZATION_DICT)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_NORM_BY_CLIMO,
            atol=TOLERANCE, equal_nan=True))

    def test_normalize_soundings(self):
        """Ensures correct output from normalize_soundings."""

        this_input_matrix = copy.deepcopy(SOUNDING_MATRIX_UNNORMALIZED)
        this_normalized_matrix = dl_utils.normalize_soundings(
            sounding_matrix=this_input_matrix,
            pressureless_field_names=SOUNDING_FIELD_NAMES,
            normalization_dict=SOUNDING_NORMALIZATION_DICT)

        self.assertTrue(numpy.allclose(
            this_normalized_matrix, SOUNDING_MATRIX_NORMALIZED, atol=TOLERANCE))

    def test_sample_by_class_tornado(self):
        """Ensures correct output from sample_by_class.

        In this case, target variable = tornado occurrence.
        """

        these_indices = dl_utils.sample_by_class(
            sampling_fraction_by_class_dict=
            BALANCED_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME,
            target_values=TORNADO_LABELS_TO_SAMPLE,
            num_examples_total=NUM_EXAMPLES_TOTAL, test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices, TORNADO_INDICES_TO_KEEP))

    def test_sample_by_class_wind(self):
        """Ensures correct output from sample_by_class.

        In this case, target variable = wind-speed category.
        """

        these_indices = dl_utils.sample_by_class(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_4CLASS_DICT,
            target_name=WIND_TARGET_NAME_4CLASSES,
            target_values=WIND_LABELS_TO_SAMPLE,
            num_examples_total=NUM_EXAMPLES_TOTAL, test_mode=True)

        self.assertTrue(numpy.array_equal(these_indices, WIND_INDICES_TO_KEEP))


if __name__ == '__main__':
    unittest.main()
