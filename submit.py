import numpy as np
import sklearn
from scipy.linalg import khatri_rao
import warnings
warnings.filterwarnings("ignore", message=".*alpha=0.*")
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your models using training CRPs
	# X_train has 8 columns containing the challenge bits
	# y_train contains the values for responses


    # Step 1: Feature map (φ)
    def phi(X):
        n = X.shape[0]
        feat = np.zeros((n, 64))
        for i in range(64):
            indices = np.arange(i, 8)
            partial = 1 - 2 * X[:, indices]
            feat[:, i] = np.prod(partial, axis=1)
        return feat

    # Apply φ to raw challenges
    X_feat = phi(X_train)

    # Step 2: Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # Step 3: Train Logistic Regression
    model = LogisticRegression(
        max_iter=100,
        C=0.01,
        solver='liblinear',
        random_state=50
    )
    model.fit(X_scaled, y_train)

    # Step 4: Return model weights and bias
    w = model.coef_[0]
    b = model.intercept_[0]
    return w, b



################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

    # Step 1: φ map
    n = X.shape[0]
    feat = np.zeros((n, 64))
    for i in range(64):
        indices = np.arange(i, 8)
        partial = 1 - 2 * X[:, indices]
        feat[:, i] = np.prod(partial, axis=1)

    # Step 2: Standardize
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat)

    return feat_scaled


################################
# Non Editable Region Starting #
################################
def my_decode(w):
################################
#  Non Editable Region Ending  #
################################
    w1 = np.array(w[:-1])  # 64-dim weight vector
    b = w[-1]              # scalar bias
    w = w1

    A = np.zeros((65, 256))
    y = np.zeros(65)

    # Row 0
    A[0, 0] = 0.5
    A[0, 1] = -0.5
    A[0, 2] = 0.5
    A[0, 3] = -0.5
    y[0] = w[0]

    for i in range(1, 64):
        A[i, 4*i+0] = 0.5
        A[i, 4*i+1] = -0.5
        A[i, 4*i+2] = 0.5
        A[i, 4*i+3] = -0.5
        A[i, 4*(i-1)+0] += 0.5
        A[i, 4*(i-1)+1] += -0.5
        A[i, 4*(i-1)+2] += -0.5
        A[i, 4*(i-1)+3] += 0.5
        y[i] = w[i]

    # Row 64 (bias)
    A[64, 4*63+0] = 0.5
    A[64, 4*63+1] = -0.5
    A[64, 4*63+2] = -0.5
    A[64, 4*63+3] = 0.5
    y[64] = b

    model = Lasso(alpha=1e-18, positive=True, fit_intercept=False, max_iter=1000)
    model.fit(A, y)
    x = model.coef_

    # Split delays
    p = x[0::4]
    q = x[1::4]
    r = x[2::4]
    s = x[3::4]

    return p, q, r, s

