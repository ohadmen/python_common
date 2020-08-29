import multiprocessing
from functools import partial
import numpy as np
from typing import Callable

class InlierData:
    def __init__(self, err, threshold, model_points):
        self._err = np.sqrt(np.mean(err ** 2))
        self._inliers = err < threshold
        self._inliers[model_points] = True
        self._inliers_count = np.sum(self._inliers)

    def __le__(self, other):
        if self.get_inliers_count()>other.get_inliers_count():
            return True
        elif self.get_inliers_count()==other.get_inliers_count():
            if self.get_error() < other.get_error():
                return True
        return False

    def get_inliers(self):
        return self._inliers

    def get_inliers_count(self):
        return self._inliers_count

    def get_error(self):
        return self._err

def ransac(data: np.ndarray,
    gen_model_func: Callable[[np.ndarray], object],
    gen_error_func: Callable[[np.ndarray, object], np.ndarray],
    n_model_points: int, threshold: float, n_itr: int = 1000,
    random_seed: int = -1,
    selected_set_check: Callable[[np.ndarray], bool] = lambda x: True
    ) -> [object, np.ndarray]:
    """
    use a ransac algorithm to find a model that describes the given data
    :param selected_set_check: (optional) check the selected set validity
    :param data: fit model to these data
    :param gen_model_func: model to describe the data
    :param gen_error_func: function to calculate the consensus set
    :param n_model_points: dimension of the model
    :param threshold: threshold for counting values as inliers
    :param n_itr: number of ransac iteration
    :param random_seed: random seed initialisation
    :return: model to describe the process with corresponding inliers
    """
    if random_seed > 0:
        np.random.seed(random_seed)

    n = data.shape[0]
    assert (n_model_points < n)
    best_inliers = InlierData(np.ones(n) * np.inf, threshold, [])

    for i in range(n_itr):
        indx = np.random.choice(n, n_model_points, replace=False)
        model_points = data[indx]
        if not selected_set_check(model_points):
            continue
        maybe_model = gen_model_func(model_points)

        maybe_inliers = InlierData(gen_error_func(data, maybe_model), threshold, model_points)

        if best_inliers <= maybe_inliers:
            best_inliers = maybe_inliers

        if best_inliers.get_inliers_count() < n_model_points:
            best_model = gen_model_func(data)
            best_inliers_out = best_inliers
        else:
            best_model = gen_model_func(data[best_inliers.get_inliers()])
            best_inliers_out = gen_error_func(data, best_model)
        np.random.seed(random_seed + i)
    n = data.shape[0]
    indx = np.random.choice(n, n_model_points, replace=False)
    model_points = data[indx]
    if not selected_set_check(model_points):
        return InlierData(np.ones(n) * np.inf, threshold, [])
    maybe_model = gen_model_func(model_points)

    return InlierData(gen_error_func(data, maybe_model), threshold, model_points)

# def ransac_p(data: np.ndarray,
#     gen_model_func: Callable[[np.ndarray], object],
#     gen_error_func: Callable[[np.ndarray, object], np.ndarray],
#     n_model_points: int, threshold: float, n_itr: int = 1000,
#     random_seed: int = -1,
#     selected_set_check: Callable[[np.ndarray], bool] = lambda x: True
#     ) -> [object, np.ndarray]:
#     """
#     use a ransac algorithm to find a model that describes the given data
#     :param selected_set_check:
#     :param data: fit model to these data
#     :param gen_model_func: model to describe the data
#     :param gen_error_func: function to calculate the consensus set
#     :param n_model_points: dimension of the model
#     :param threshold: threshold for counting values as inliers
#     :param n_itr: number of ransac iteration
#     :param random_seed: random seed initialisation
#     :return: model to describe the process with corresponding inliers
#     """
#     func = partial(ransac_p_thread_func, data, gen_model_func, gen_error_func, n_model_points, threshold, random_seed,
#     selected_set_check)
#     processing_pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
#     inliers = processing_pool.map(func, range(n_itr))
#
#     best_indx = np.argmax([x.get_inliers_count() for x in inliers])
#     best_inliers = inliers[best_indx]
#     if best_inliers.get_inliers_count() < n_model_points:
#             best_model = gen_model_func(data)
#             best_inliers_out = best_inliers
#         else:
#             best_model = gen_model_func(data[best_inliers.get_inliers()])
#             best_inliers_out = gen_error_func(data, best_model) < threshold
#         return best_model, best_inliers_out
