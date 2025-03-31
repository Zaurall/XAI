import numpy as np
import matplotlib.pyplot as plt


class ShapPlotter:
    """
    A class for visualizing SHAP values using various plot types.
    """

    def __init__(self, shap_red='#ff0051', shap_blue='#008bfb'):
        """
        Initialize the SHAP plotter.

        Args:
            shap_red (str): Color for positive SHAP values. Defaults to '#ff0051'.
            shap_blue (str): Color for negative SHAP values. Defaults to '#008bfb'.
        """
        self.shap_red = shap_red
        self.shap_blue = shap_blue

    def beeswarm(self, shap_values, feature_names, number_features=9, figsize=(12, 6)):
        """
        Plots a beeswarm plot of SHAP values for feature importance visualization.

        Args:
            shap_values (np.ndarray): SHAP values of shape (n_samples, n_features).
            feature_names (list of str): List of feature names corresponding to shap_values.
            number_features (int, optional): Number of top features to display. Defaults to 9.
            figsize (tuple, optional): Figure size. Defaults to (12, 6).
        """
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_shap_values = shap_values[:, sorted_indices]

        if len(feature_names) == number_features + 1:
            top_features = sorted_features
            top_shap_values = sorted_shap_values
        else:
            top_features = sorted_features[:number_features]
            top_shap_values = sorted_shap_values[:, :number_features]
            if sorted_shap_values.shape[1] > number_features:
                other_shap_values = np.sum(sorted_shap_values[:, number_features:], axis=1)
                top_features.append(f"Other {len(feature_names) - number_features} Features (Sum)")
                top_shap_values = np.column_stack((top_shap_values, other_shap_values))

        plt.figure(figsize=figsize)

        for i, feature in enumerate(top_features):
            y = np.random.normal(len(top_features) - i - 1, 0.1, size=len(top_shap_values))
            plt.scatter(
                top_shap_values[:, i], y,
                c=np.where(top_shap_values[:, i] > 0, self.shap_red, self.shap_blue),
                alpha=0.8,
                s=20
            )

        plt.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.scatter([], [], c=self.shap_red, label='high feature value', alpha=0.8, s=20)
        plt.scatter([], [], c=self.shap_blue, label='low feature value', alpha=0.8, s=20)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        plt.yticks(range(len(top_features)), top_features[::-1])
        plt.xlabel('SHAP Value (impact on model output)', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def dependence(self, shap_values, feature_values, feature_name, figsize=(9, 6)):
        """
        Plots a SHAP dependence plot showing the relationship between feature values and SHAP values.

        Args:
            shap_values (np.ndarray): SHAP values for a single feature.
            feature_values (np.ndarray): Corresponding feature values.
            feature_name (str): Name of the feature being plotted.
            figsize (tuple, optional): Figure size. Defaults to (9, 6).
        """
        plt.figure(figsize=figsize)
        plt.scatter(feature_values, shap_values, alpha=0.8, color=self.shap_red)
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel('SHAP Value', fontsize=12)
        plt.title(f'Dependence Plot for {feature_name}', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def bar(self, shap_values, feature_names, number_features=9, figsize=(9, 6)):
        """
        Plots a bar chart of SHAP values, showing feature importance.

        Args:
            shap_values (np.ndarray): SHAP values for features, either for a single instance or averaged globally.
            feature_names (list of str): List of feature names corresponding to shap_values.
            number_features (int, optional): Number of top features to display. Defaults to 9.
            figsize (tuple, optional): Figure size. Defaults to (9, 6).
        """
        if len(shap_values.shape) == 1:
            sorted_indices = np.argsort(np.abs(shap_values))[::-1]
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_shap_values = shap_values[sorted_indices]

            if len(feature_names) == number_features + 1:
                top_features = sorted_features
                top_shap_values = sorted_shap_values
            else:
                if number_features is not None and number_features < len(sorted_features):
                    top_features = sorted_features[:number_features]
                    top_shap_values = sorted_shap_values[:number_features]
                    other_shap_value = np.sum(sorted_shap_values[number_features:])
                    top_features.append(f"Other {len(feature_names) - number_features} Features (Sum)")
                    top_shap_values = np.append(top_shap_values, other_shap_value)
                else:
                    top_features = sorted_features
                    top_shap_values = sorted_shap_values

            plt.figure(figsize=figsize)
            colors = [self.shap_red if val > 0 else self.shap_blue for val in top_shap_values]
            bars = plt.barh(top_features, top_shap_values, color=colors)

            for bar, value in zip(bars, top_shap_values):
                width = bar.get_width()
                sign = '+' if value > 0 else '-'
                plt.text(width + 0.01 * np.sign(width), bar.get_y() + bar.get_height() / 2,
                         f'{sign}{abs(value):.2f}',
                         va='center', ha='left' if value > 0 else 'right', fontsize=10)

            plt.axvline(0, color='black', linestyle='--', alpha=0.5)
            x_min = min(top_shap_values) - 0.2 * abs(min(top_shap_values))
            x_max = max(top_shap_values) + 0.2 * abs(max(top_shap_values))
            plt.xlim(x_min, x_max)
            plt.xlabel('SHAP Value (Local Explanation)', fontsize=12)
            plt.gca().invert_yaxis()
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        else:
            shap_values = np.mean(np.abs(shap_values), axis=0)
            sorted_indices = np.argsort(shap_values)[::-1]
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_shap_values = shap_values[sorted_indices]

            if len(feature_names) == number_features + 1:
                top_features = sorted_features
                top_shap_values = sorted_shap_values
            else:
                if number_features is not None and number_features < len(sorted_features):
                    top_features = sorted_features[:number_features]
                    top_shap_values = sorted_shap_values[:number_features]
                    other_shap_value = np.sum(sorted_shap_values[number_features:])
                    top_features.append(f"Other {len(feature_names) - number_features} Features (Sum)")
                    top_shap_values = np.append(top_shap_values, other_shap_value)
                else:
                    top_features = sorted_features
                    top_shap_values = sorted_shap_values

            plt.figure(figsize=figsize)
            bars = plt.barh(top_features, top_shap_values, color=self.shap_red)

            for bar, value in zip(bars, top_shap_values):
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'+{value:.2f}',
                         va='center', ha='left', fontsize=10)

            plt.axvline(0, color='black', linestyle='--', alpha=0.5)
            x_min = min(top_shap_values) - 1 * abs(min(top_shap_values))
            x_max = max(top_shap_values) + 0.1 * abs(max(top_shap_values))
            plt.xlim(x_min, x_max)
            plt.xlabel('Mean |SHAP Value| (Global Explanation)', fontsize=12)
            plt.title('Global Feature Importance (SHAP)', fontsize=14)
            plt.ylabel('Features', fontsize=12)
            plt.gca().invert_yaxis()
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    def waterfall(self, shap_values, feature_names, expected_value, number_features=9, figsize=(9, 6)):
        """
        Plots a SHAP waterfall chart for visualizing the impact of each feature on a prediction.

        Args:
            shap_values (np.ndarray): SHAP values for a single prediction.
            feature_names (list of str): List of feature names corresponding to shap_values.
            expected_value (float): Baseline model output (typically the mean prediction).
            number_features (int, optional): Number of top features to display. Defaults to 9.
            figsize (tuple, optional): Figure size. Defaults to (9, 6).
        """
        shap_values = np.array(shap_values)
        feature_names = np.array(feature_names)

        sort_idx = np.argsort(-np.abs(shap_values))
        shap_values = shap_values[sort_idx]
        feature_names = feature_names[sort_idx]

        if len(feature_names) != number_features + 1:
            if len(shap_values) > number_features:
                top_shap = shap_values[:number_features]
                top_features = feature_names[:number_features]
                others_sum = np.sum(shap_values[number_features:])
                shap_values = np.concatenate([top_shap, [others_sum]])
                feature_names = np.concatenate([top_features, [f"Others {len(feature_names) - number_features} features"]])

        cum_values = [expected_value]
        for val in shap_values:
            cum_values.append(cum_values[-1] + val)
        cum_values = np.array(cum_values)

        n_items = len(shap_values)

        fig, ax = plt.subplots(figsize=figsize)

        for i in range(n_items):
            start = cum_values[i]
            width = shap_values[i]
            color = self.shap_red if width >= 0 else self.shap_blue

            ax.barh(i, width, left=start, color=color, height=0.5)

            text_x = start + width
            ha = 'left' if width >= 0 else 'right'
            string = f" {width:+.2f}" if width >= 0 else f"{width:+.2f} "
            ax.text(text_x, i, string, va='center', ha=ha, fontsize=9)

        ax.axvline(expected_value, color='gray', linestyle='--', linewidth=1)

        yticks = list(range(n_items))
        ylabels = list(feature_names)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        ax.invert_yaxis()

        ax.set_xlabel("SHAP Waterfall Plot")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()