"""Multiple Correspondence Analysis (MCA)"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError

from prince import CA
from prince import plot

class MCA(CA):

    def __init__(self, n_components=2, n_iter=10, copy=True, check_input=True, benzecri=False,
                 random_state=None, engine='auto'):
        super().__init__(
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            benzecri=benzecri,
            random_state=random_state,
            engine=engine
        )
        self._ohe = OneHotEncoder(handle_unknown="ignore")

    def _call_ohe(self, X):
        try:
            ohe = self._ohe.transform(X)
        except NotFittedError:
            ohe = self._ohe.fit_transform(X)

        return ohe.toarray()

    def fit(self, X, y=None, **kwargs):

        if self.check_input:
            utils.check_array(X, dtype=[str])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_initial_columns = X.shape[1]

        one_hot = self._call_ohe(X)
        # Apply CA to the indicator matrix
        super().fit(one_hot)

        # Compute the total inertia
        n_new_columns = one_hot.shape[1]
        self.total_inertia_ = (n_new_columns - n_initial_columns) / n_initial_columns

        return self

    def row_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return super().row_coordinates(self._call_ohe(X))

    def column_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return super().column_coordinates(self._call_ohe(X))

    def transform(self, X):
        """Computes the row principal coordinates of a dataset."""
        utils.validation.check_is_fitted(self)
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])
        return self.row_coordinates(X)

    def plot_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                         show_row_points=True, row_points_size=10,
                         row_points_alpha=0.6, show_row_labels=False,
                         show_column_points=True, column_points_size=30, show_column_labels=False,
                         legend_n_cols=1):
        """Plot row and column principal coordinates.

        Parameters:
            ax (matplotlib.Axis): A fresh one will be created and returned if not provided.
            figsize ((float, float)): The desired figure size if `ax` is not provided.
            x_component (int): Number of the component used for the x-axis.
            y_component (int): Number of the component used for the y-axis.
            show_row_points (bool): Whether to show row principal components or not.
            row_points_size (float): Row principal components point size.
            row_points_alpha (float): Alpha for the row principal component.
            show_row_labels (bool): Whether to show row labels or not.
            show_column_points (bool): Whether to show column principal components or not.
            column_points_size (float): Column principal components point size.
            show_column_labels (bool): Whether to show column labels or not.
            legend_n_cols (int): Number of columns used for the legend.

        Returns:
            matplotlib.Axis
        """

        utils.validation.check_is_fitted(self)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add style
        ax = plot.stylize_axis(ax)

        # Plot row principal coordinates
        if show_row_points or show_row_labels:

            row_coords = self.row_coordinates(X)

            if show_row_points:
                ax.scatter(
                    row_coords.iloc[:, x_component],
                    row_coords.iloc[:, y_component],
                    s=row_points_size,
                    label=None,
                    color=plot.GRAY['dark'],
                    alpha=row_points_alpha
                )

            if show_row_labels:
                for _, row in row_coords.iterrows():
                    ax.annotate(row.name, (row[x_component], row[y_component]))

        # Plot column principal coordinates
        if show_column_points or show_column_labels:

            col_coords = self.column_coordinates(X)
            x = col_coords[x_component]
            y = col_coords[y_component]

            prefixes = col_coords.index.str.split('_').map(lambda x: x[0])

            for prefix in prefixes.unique():
                mask = prefixes == prefix

                if show_column_points:
                    ax.scatter(x[mask], y[mask], s=column_points_size, label=prefix)

                if show_column_labels:
                    for i, label in enumerate(col_coords[mask].index):
                        ax.annotate(label, (x[mask][i], y[mask][i]))

            ax.legend(ncol=legend_n_cols)

        # Text
        ax.set_title('Row and column principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

        return ax

if __name__=="__main__":


    X = pd.DataFrame(
        data=[
            ['A', 'A', 'A', 2, 5, 7, 6, 3, 6, 7],
            ['A', 'A', 'A', 4, 4, 4, 2, 4, 4, 3],
            ['B', 'A', 'B', 5, 2, 1, 1, 7, 1, 1],
            ['B', 'A', 'B', 7, 2, 1, 2, 2, 2, 2],
            ['B', 'B', 'B', 3, 5, 6, 5, 2, 6, 6],
            ['B', 'B', 'A', 3, 5, 4, 5, 1, 7, 5]
        ],
        columns=['E1 fruity', 'E1 woody', 'E1 coffee',
                 'E2 red fruit', 'E2 roasted', 'E2 vanillin', 'E2 woody',
                 'E3 fruity', 'E3 butter', 'E3 woody'],
        index=['Wine {}'.format(i + 1) for i in range(6)]
    )

    Z = pd.DataFrame(
        data=[
            ['A', 'B', 'A', 1, 5, 4, 6, 3, 6, 4],
            ['C', 'A', 'F', 4, 4, 3, 2, 4, 4, 5],
            ['A', 'B', 'B', 2, 2, 4, 1, 7, 1, 3],
            ['D', 'B', 'B', 7, 2, 1, 2, 2, 2, 2],
            ['A', 'A', 'C', 3, 5, 3, 5, 2, 6, 6],
            ['B', 'C', 'A', 4, 5, 5, 5, 1, 7, 5]
        ],
        columns=['E1 fruity', 'E1 woody', 'E1 coffee',
                 'E2 red fruit', 'E2 roasted', 'E2 vanillin', 'E2 woody',
                 'E3 fruity', 'E3 butter', 'E3 woody'],
        index=['Wine {}'.format(i + 1) for i in range(6, 12)]
    )

    X['Oak type'] = [1, 2, 2, 2, 1, 1]
    Z['Oak type'] = [1, 1, 2, 1, 2, 1]
    Y = pd.concat([X, Z])

    X = X.astype(str)
    Z = Z.astype(str)
    Y = Y.astype(str)

    mca = MCA(
        n_components=5,
        n_iter=2,
        copy=True,
        check_input=True,
        engine='auto',
        random_state=42
    )


    mca = mca.fit(X)
    print(X)
    print(mca.row_coordinates(X))
    print(Z)
    print(mca.row_coordinates(Z))