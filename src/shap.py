import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

class KernelExplainer:
    """
    An implementation of the Kernel SHAP algorithm for explaining model predictions.
    This method approximates SHAP values using a weighted linear regression approach.
    """

    def __init__(self, model_fn, background_data, max_samples=100, random_state=None):
        """
        Initialize the explainer.

        Args:
            model_fn (callable): A function that takes input data and returns model predictions.
            background_data (numpy.ndarray or pandas.DataFrame): Data representing the feature distribution.
            max_samples (int): Maximum number of samples to use for approximation.
            random_state (int or None): Seed for reproducibility.
        """
        self.model_fn = model_fn
        self.background_data = background_data.values if isinstance(background_data, pd.DataFrame) else background_data
        self.max_samples = max_samples
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.N, self.M = self.background_data.shape
        background_preds = model_fn(self.background_data)

        if len(background_preds.shape) == 1:
            self.expected_value = np.mean(background_preds)
            self.multi_output = False
        else:
            self.expected_value = np.mean(background_preds, axis=0)
            self.multi_output = True

    def _shapley_kernel(self, M, s):
        """
        Compute the Shapley kernel weight for a coalition of size `s`.

        Args:
            M (int): Total number of features.
            s (int): Size of the coalition.

        Returns:
            float: The kernel weight.
        """
        if s == 0 or s == M:
            return 1000
        return (M - 1) / (s * (M - s))

    def _generate_coalition_weights(self, num_players):
        """
        Generate weights for all possible feature coalitions.

        Args:
            num_players (int): Number of features.

        Returns:
            numpy.ndarray: Array of coalition weights.
        """
        weights = np.zeros(2**num_players)
        for i in range(2**num_players):
            coalition = bin(i)[2:].zfill(num_players)
            s = sum(int(bit) for bit in coalition)
            weights[i] = self._shapley_kernel(num_players, s)
        return weights

    def _generate_feature_coalitions(self, num_features, num_samples):
        """
        Generate binary feature coalitions for sampling.

        Args:
            num_features (int): Number of features.
            num_samples (int): Number of coalitions to generate.

        Returns:
            numpy.ndarray: Binary matrix representing coalitions.
        """
        if num_samples >= 2**num_features - 2:
            coalitions = np.zeros((2**num_features, num_features), dtype=np.bool_)
            for i in range(2**num_features):
                binary = bin(i)[2:].zfill(num_features)
                coalitions[i] = np.array([int(bit) for bit in binary], dtype=np.bool_)
            return coalitions[1:-1]
        else:
            coalitions = np.zeros((num_samples, num_features), dtype=np.bool_)
            coalitions[0] = np.zeros(num_features, dtype=np.bool_)
            coalitions[1] = np.ones(num_features, dtype=np.bool_)
            for i in range(2, num_samples):
                coalitions[i] = self.rng.binomial(1, 0.5, num_features).astype(np.bool_)
            return coalitions

    def _create_masked_samples(self, instance, coalitions, background_indices):
        """
        Create masked samples by combining instance and background data.

        Args:
            instance (numpy.ndarray): Instance to explain.
            coalitions (numpy.ndarray): Binary matrix of feature coalitions.
            background_indices (numpy.ndarray): Indices of background samples.

        Returns:
            numpy.ndarray: Matrix of masked instances.
        """
        num_coalitions = coalitions.shape[0]
        num_background = len(background_indices)
        masked_samples = np.zeros((num_coalitions * num_background, self.M))

        for i, coalition in enumerate(coalitions):
            for j, bg_idx in enumerate(background_indices):
                sample_idx = i * num_background + j
                masked_samples[sample_idx] = self.background_data[bg_idx].copy()
                masked_samples[sample_idx, coalition] = instance[coalition]
        return masked_samples

    def shap_values(self, X, nsamples=None):
        """
        Compute SHAP values for the given instances.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Instances to explain.
            nsamples (int or None): Number of samples to use (overrides `max_samples` if provided).

        Returns:
            numpy.ndarray: SHAP values for each instance and feature.
        """
        X_numpy = X.values if isinstance(X, pd.DataFrame) else X
        num_instances = X_numpy.shape[0]
        samples = nsamples if nsamples is not None else self.max_samples
        samples = min(samples, 2**self.M - 2)

        bg_size = min(self.N, 10)
        bg_indices = self.rng.choice(self.N, bg_size, replace=False)
        coalitions = self._generate_feature_coalitions(self.M, samples)
        num_coalitions = coalitions.shape[0]

        coalition_weights = np.array([self._shapley_kernel(self.M, coalition.sum()) for coalition in coalitions])

        if self.multi_output:
            num_outputs = len(self.expected_value)
            all_shap_values = np.zeros((num_outputs, num_instances, self.M))
        else:
            all_shap_values = np.zeros((num_instances, self.M))

        for i in tqdm(range(num_instances), desc="Calculating SHAP values", unit="instance"):
            instance = X_numpy[i]
            masked_samples = self._create_masked_samples(instance, coalitions, bg_indices)
            masked_preds = self.model_fn(masked_samples)

            if self.multi_output:
                masked_preds_reshaped = masked_preds.reshape(num_coalitions, bg_size, -1)
                masked_preds_avg = masked_preds_reshaped.mean(axis=1)
                for j in range(num_outputs):
                    y = masked_preds_avg[:, j]
                    model = LinearRegression()
                    model.fit(coalitions, y, sample_weight=coalition_weights)
                    all_shap_values[j, i, :] = model.coef_
            else:
                masked_preds_reshaped = masked_preds.reshape(num_coalitions, bg_size)
                masked_preds_avg = masked_preds_reshaped.mean(axis=1)
                model = LinearRegression()
                model.fit(coalitions, masked_preds_avg, sample_weight=coalition_weights)
                all_shap_values[i, :] = model.coef_

        return all_shap_values if self.multi_output else all_shap_values


class SamplingExplainer:
    """
    An approximate SHAP explainer (coded specially for CatBoostExplainer) that uses sampling and perturbation to estimate SHAP values.
    This method is inspired by SHAP's additive feature attribution framework but does not rely
    on tree structures. It is suitable for models where exact SHAP computation is infeasible.
    """

    def __init__(self, predict, background_data):
        """
        Initialize the explainer.

        Args:
            predict (callable): A function that takes input data and returns model predictions.
            background_data (numpy.ndarray or pandas.DataFrame): Background data representing
                the feature distribution.
        """
        self.predict = predict
        self.background_data = background_data.values if isinstance(background_data, pd.DataFrame) else background_data

        # Transform background predictions into log-odds space
        bg_probas = predict(self.background_data)[:, 1]
        self.bg_logodds = np.log(bg_probas / (1 - bg_probas + 1e-12))
        self.expected_value = np.mean(self.bg_logodds)

    def shap_values(self, X, max_samples=100):
        """
        Compute SHAP values for the given instances using a sampling-based approach.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Instances to explain.
            max_samples (int): Maximum number of background samples to use for approximation.

        Returns:
            numpy.ndarray: SHAP values for each instance and feature.
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        n_samples, n_features = X_values.shape

        # Sample background data for efficiency
        bg_mask = np.random.choice(self.background_data.shape[0], 
                                   min(max_samples, self.background_data.shape[0]), 
                                   replace=False)
        bg_samples = self.background_data[bg_mask]
        bg_logodds = self.bg_logodds[bg_mask]

        # Initialize SHAP matrix
        shap_values = np.zeros((n_samples, n_features))

        for i in tqdm(range(n_samples), desc="Calculating SHAP values"):
            instance_proba = self.predict(X_values[i].reshape(1, -1))[0, 1]
            instance_logodds = np.log(instance_proba / (1 - instance_proba + 1e-12))

            for j in range(n_features):
                # Create perturbed dataset by replacing feature j with the instance value
                perturbed = bg_samples.copy()
                perturbed[:, j] = X_values[i, j]

                # Compute perturbed predictions in log-odds space
                perturbed_probas = self.predict(perturbed)[:, 1]
                perturbed_logodds = np.log(perturbed_probas / (1 - perturbed_probas + 1e-12))

                # Calculate marginal contribution of feature j
                shap_values[i, j] = (perturbed_logodds - bg_logodds).mean()

            # Ensure SHAP values sum to the difference between prediction and expected value
            sum_diff = instance_logodds - self.expected_value
            current_sum = shap_values[i].sum()

            if abs(current_sum - sum_diff) > 1e-8:
                # Scale SHAP values to preserve additivity
                ratio = sum_diff / current_sum
                shap_values[i] *= ratio

        return shap_values