"""Methods for running the permutation test.

The permutation test (or "permutation-based variable importance") is explained
in Lakshmanan et al. (2015).

--- REFERENCES ---

Lakshmanan, V., C. Karstens, J. Krause, K. Elmore, A. Ryzhkov, and S. Berkseth,
    2015: "Which polarimetric variables are important for weather/no-weather
    discrimination?" Journal of Atmospheric and Oceanic Technology, 32 (6),
    1209-1223.
"""

import copy
import numpy
import keras.utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn


def prediction_function_2d_cnn(model_object, list_of_input_matrices):
    """Prediction function for 2-D GewitterGefahr CNN.

    E = number of examples
    K = number of target classes

    :param model_object: See doc for `run_permutation_test`.
    :param list_of_input_matrices: Same.
    :return: class_probability_matrix: E-by-K numpy array, where
        class_probability_matrix[i, k] is the probability that the [i]th example
        belongs to the [k]th class.
    """

    if len(list_of_input_matrices) == 2:
        sounding_matrix = list_of_input_matrices[1]
    else:
        sounding_matrix = None

    return cnn.apply_2d_cnn(
        model_object=model_object, radar_image_matrix=list_of_input_matrices[0],
        sounding_matrix=sounding_matrix)


def prediction_function_3d_cnn(model_object, list_of_input_matrices):
    """Prediction function for 3-D GewitterGefahr CNN.

    E = number of examples
    K = number of target classes

    :param model_object: See doc for `run_permutation_test`.
    :param list_of_input_matrices: Same.
    :return: class_probability_matrix: E-by-K numpy array, where
        class_probability_matrix[i, k] is the probability that the [i]th example
        belongs to the [k]th class.
    """

    if len(list_of_input_matrices) == 2:
        sounding_matrix = list_of_input_matrices[1]
    else:
        sounding_matrix = None

    return cnn.apply_3d_cnn(
        model_object=model_object, radar_image_matrix=list_of_input_matrices[0],
        sounding_matrix=sounding_matrix)


def prediction_function_2d3d_cnn(model_object, list_of_input_matrices):
    """Prediction function for hybrid 2D/3D GewitterGefahr CNN.

    E = number of examples
    K = number of target classes

    :param model_object: See doc for `run_permutation_test`.
    :param list_of_input_matrices: Same.
    :return: class_probability_matrix: E-by-K numpy array, where
        class_probability_matrix[i, k] is the probability that the [i]th example
        belongs to the [k]th class.
    """

    if len(list_of_input_matrices) == 3:
        sounding_matrix = list_of_input_matrices[2]
    else:
        sounding_matrix = None

    return cnn.apply_2d3d_cnn(
        model_object=model_object,
        reflectivity_image_matrix_dbz=list_of_input_matrices[0],
        az_shear_image_matrix_s01=list_of_input_matrices[1],
        sounding_matrix=sounding_matrix)


def cross_entropy_function(target_values, class_probability_matrix):
    """Cross-entropy cost function.

    This function works for binary or multi-class classification.

    :param target_values: See doc for `run_permutation_test`.
    :param class_probability_matrix: Same.
    :return: cross_entropy: Scalar.
    """

    num_examples = class_probability_matrix.shape[0]
    num_classes = class_probability_matrix.shape[1]

    target_matrix = keras.utils.to_categorical(
        target_values, num_classes
    ).astype(int)

    return -1 * numpy.sum(
        target_matrix * numpy.log2(class_probability_matrix)
    ) / num_examples


def run_permutation_test(
        model_object, list_of_input_matrices, predictor_names_by_matrix,
        target_values, prediction_function, cost_function):
    """Runs the permutation test.

    N = number of input matrices
    E = number of examples
    C_q = number of channels (predictors) in the [q]th matrix
    K = number of target classes

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param list_of_input_matrices: length-T list of matrices (numpy arrays), in
        the order that they were fed to the model for training.  In other words,
        if the order of training matrices was [radar images, soundings], the
        order of these matrices must be [radar images, soundings].  The first
        axis of each matrix should have length E, and the last axis of the [q]th
        matrix should have length C_q.
    :param predictor_names_by_matrix: length-T list of lists.  The [q]th list
        should be a list of predictor variables in the [q]th matrix, with length
        C_q.
    :param target_values: length-E numpy array of target values (integer class
        labels).
    :param prediction_function: Function used to generate predictions from the
        model.  Should have the following inputs and outputs.
    Input: model_object: Same as input to this method.
    Input: list_of_input_matrices: Same as input to this method, except maybe
        with permuted values.
    Output: class_probability_matrix: E-by-K numpy array, where
        class_probability_matrix[i, k] is the probability that the [i]th example
        belongs to the [k]th class.

    :param cost_function: Function used to evaluate predictions from the model.
        Should have the following inputs and outputs.  This method will assume
        that lower values are better.  In other words, the cost function must be
        negatively oriented.
    Input: target_values: Same as input to this method.
    Input: class_probability_matrix: Output from `prediction_function`.
    Output: cost: Scalar value.

    :return: Not sure yet.
    :raises: ValueError: if length of `list_of_input_matrices` != length of
        `predictor_names_by_matrix`.
    :raises: ValueError: if any input matrix has < 3 dimensions.
    """

    error_checking.assert_is_integer_numpy_array(target_values)
    error_checking.assert_is_geq_numpy_array(target_values, 0)

    if len(list_of_input_matrices) != len(predictor_names_by_matrix):
        error_string = (
            'Number of input matrices ({0:d}) should equal number of predictor-'
            'name lists ({1:d}).'
        ).format(len(list_of_input_matrices), len(predictor_names_by_matrix))

        raise ValueError(error_string)

    num_input_matrices = len(list_of_input_matrices)
    num_examples = len(target_values)

    for q in range(num_input_matrices):
        error_checking.assert_is_numpy_array_without_nan(
            list_of_input_matrices[q])

        this_num_dimensions = len(list_of_input_matrices[q].shape)
        if this_num_dimensions < 3:
            error_string = (
                '{0:d}th input matrix has {1:d} dimensions.  Should have at '
                'least 3.'
            ).format(q + 1, this_num_dimensions)

            raise ValueError(error_string)

        error_checking.assert_is_string_list(predictor_names_by_matrix[q])
        this_num_predictors = len(predictor_names_by_matrix[q])

        these_expected_dimensions = (
            (num_examples,) + list_of_input_matrices[q].shape[1:-1] +
            (this_num_predictors,)
        )
        these_expected_dimensions = numpy.array(
            these_expected_dimensions, dtype=int)

        error_checking.assert_is_numpy_array(
            list_of_input_matrices[q],
            exact_dimensions=these_expected_dimensions)

    # Get original cost (with no permutation).
    class_probability_matrix = prediction_function(
        model_object, list_of_input_matrices)
    original_cost = cost_function(target_values, class_probability_matrix)
    print 'Original cost (no permutation): {0:.4e}'.format(original_cost)

    remaining_predictor_names_by_matrix = copy.deepcopy(
        predictor_names_by_matrix)
    step_num = 0

    while True:
        print '\n'
        step_num += 1

        highest_cost = -numpy.inf
        best_matrix_index = None
        best_predictor_name = None
        best_predictor_permuted_values = None

        stopping_criterion = True

        for q in range(num_input_matrices):
            if len(remaining_predictor_names_by_matrix[q]) == 0:
                continue

            for this_predictor_name in remaining_predictor_names_by_matrix[q]:
                stopping_criterion = False

                print (
                    'Trying predictor "{0:s}" at step {1:d} of permutation '
                    'test...'
                ).format(this_predictor_name, step_num)

                these_input_matrices = copy.deepcopy(list_of_input_matrices)
                this_predictor_index = predictor_names_by_matrix[q].index(
                    this_predictor_name)

                for i in range(num_examples):
                    these_input_matrices[q][
                        i, ..., this_predictor_index
                    ] = numpy.random.permutation(
                        these_input_matrices[q][i, ..., this_predictor_index])

                this_probability_matrix = prediction_function(
                    model_object, these_input_matrices)
                this_cost = cost_function(
                    target_values, this_probability_matrix)

                if this_cost < highest_cost:
                    continue

                highest_cost = this_cost + 0.
                best_matrix_index = q
                best_predictor_name = this_predictor_name + ''
                best_predictor_permuted_values = these_input_matrices[q][
                    ..., this_predictor_index]

        if stopping_criterion:  # No more predictors to permute.
            break

        # Remove best predictor from list.
        remaining_predictor_names_by_matrix[best_matrix_index].remove(
            best_predictor_name)

        # Leave values of best predictor permuted.
        this_best_predictor_index = predictor_names_by_matrix[
            best_matrix_index].index(best_predictor_name)
        list_of_input_matrices[best_matrix_index][
            ..., this_best_predictor_index
        ] = best_predictor_permuted_values

        print 'Best predictor = "{0:s}" ... new cost = {1:.4e}'.format(
            best_predictor_name, highest_cost)
