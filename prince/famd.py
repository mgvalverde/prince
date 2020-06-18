"""Factor Analysis of Mixed Data (FAMD)"""
import numpy as np
import pandas as pd

from prince import mfa


class FAMD(mfa.MFA):

    def __init__(self, n_components=2, n_iter=3, copy=True, check_input=True, random_state=None,
                 engine='auto'):
        super().__init__(
            groups=None,
            normalize=True,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine
        )

    def fit(self, X, y=None):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Separate numerical columns from categorical columns
        num_cols = X.select_dtypes(np.number).columns.tolist()
        cat_cols = list(set(X.columns) - set(num_cols))

        # Make one group per variable type
        self.groups = {}
        if num_cols:
            self.groups['Numerical'] = num_cols
        else:
            raise ValueError('FAMD works with categorical and numerical data but ' +
                             'you only have categorical data; you should consider using MCA')
        if cat_cols:
            self.groups['Categorical'] = cat_cols
        else:
            raise ValueError('FAMD works with categorical and numerical data but ' +
                             'you only have numerical data; you should consider using PCA')

        return super().fit(X)

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

    # X['Oak type'] = [1, 2, 2, 2, 1, 1]
    # Z['Oak type'] = [1, 1, 2, 1, 2, 1]
    Y = pd.concat([X, Z])

    # X = X.astype(str)
    # Z = Z.astype(str)
    # Y = Y.astype(str)

    famd = FAMD(
        n_components=5,
        n_iter=2,
        copy=True,
        check_input=True,
        engine='auto',
        random_state=42
    )

    famd = famd.fit(X)
    print(X)
    print(famd.row_coordinates(X))
    print(Z)
    print(famd.row_coordinates(Z))