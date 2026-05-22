import numpy as np


# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return the bivariate Gaussian PDF f_XY(x,y).

    Use the formula:
    f_XY(x,y) = 1 / (2*pi*sigma_x*sigma_y*sqrt(1-rho^2)) * exp( -Q / (2*(1-rho^2)) )
    """
    # Calculate the quadratic form Q(x, y)
    term_x = ((x - mu_x) ** 2) / (sigma_x ** 2)
    term_xy = -2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y)
    term_y = ((y - mu_y) ** 2) / (sigma_y ** 2)
    q = term_x + term_xy + term_y

    # Calculate the normalization constant
    denom = 2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2)
    
    # Return the joint probability density
    return (1.0 / denom) * np.exp(-q / (2 * (1 - rho ** 2)))


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    """
    Return marginal Gaussian PDF of X.
    """
    denom = sigma_x * np.sqrt(2 * np.pi)
    exponent = -0.5 * ((x - mu_x) / sigma_x) ** 2
    return (1.0 / denom) * np.exp(exponent)


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    """
    Return marginal Gaussian PDF of Y.
    """
    denom = sigma_y * np.sqrt(2 * np.pi)
    exponent = -0.5 * ((y - mu_y) / sigma_y) ** 2
    return (1.0 / denom) * np.exp(exponent)


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return covariance matrix:
    [[sigma_x^2, rho*sigma_x*sigma_y],
     [rho*sigma_x*sigma_y, sigma_y^2]]
    """
    var_x = sigma_x ** 2
    var_y = sigma_y ** 2
    cov_xy = rho * sigma_x * sigma_y
    
    return np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])


def joint_pdf_grid_integral(mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, n=250):
    """
    Numerically approximate integral of joint Gaussian PDF over the rectangle:
    [mu_x - 4*sigma_x, mu_x + 4*sigma_x] x [mu_y - 4*sigma_y, mu_y + 4*sigma_y]

    Uses a rectangular Riemann sum meshgrid approach.
    """
    # Create evaluation boundaries
    x_edges = np.linspace(mu_x - 4 * sigma_x, mu_x + 4 * sigma_x, n)
    y_edges = np.linspace(mu_y - 4 * sigma_y, mu_y + 4 * sigma_y, n)
    
    # Calculate step size intervals (dx, dy)
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    
    # Create 2D coordinates grid
    X, Y = np.meshgrid(x_edges, y_edges)
    
    # Evaluate PDF across the coordinates grid
    pdf_values = joint_gaussian_pdf(X, Y, mu_x, mu_y, sigma_x, sigma_y, rho)
    
    # Sum up volumes of cell blocks (dx * dy)
    return float(np.sum(pdf_values) * dx * dy)


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):
    """
    Generate n samples from a jointly Gaussian distribution.
    Return two arrays: x_samples, y_samples
    """
    # Set up random number generator engine for consistency
    rng = np.random.default_rng(seed)
    
    mean_vector = [mu_x, mu_y]
    cov_matrix = covariance_matrix(sigma_x, sigma_y, rho)
    
    # Draw sample instances
    samples = rng.multivariate_normal(mean_vector, cov_matrix, size=n)
    
    return samples[:, 0], samples[:, 1]


def sample_means(x_samples, y_samples):
    """
    Return sample means of X and Y.
    """
    return float(np.mean(x_samples)), float(np.mean(y_samples))


def sample_covariance_matrix(x_samples, y_samples):
    """
    Return 2 by 2 sample covariance matrix.
    Use denominator n-1 (Bessel's correction via ddof=1).
    """
    return np.cov(x_samples, y_samples, ddof=1)


def sample_correlation(x_samples, y_samples):
    """
    Return sample correlation coefficient.
    """
    corr_matrix = np.corrcoef(x_samples, y_samples)
    return float(corr_matrix[0, 1])


def gaussian_independence_check(rho):
    """
    For jointly Gaussian variables:
    return True if rho is zero, otherwise False.
    """
    return bool(rho == 0)


def zero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0 and check that
    sample covariance is approximately zero.
    Return True or False.
    """
    x, y = generate_joint_gaussian_samples(n=n, rho=0.0, seed=42)
    cov_mat = sample_covariance_matrix(x, y)
    cov_xy = cov_mat[0, 1]
    
    # Cast to raw Python bool explicitly to bypass NumPy boolean type checking errors
    return bool(abs(cov_xy) < 0.05)


def nonzero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0.6 and check that
    sample covariance is close to rho*sigma_x*sigma_y.
    Return True or False.
    """
    sigma_x, sigma_y, rho = 2, 3, 0.6
    expected_cov = rho * sigma_x * sigma_y  # 3.6
    
    x, y = generate_joint_gaussian_samples(n=n, sigma_x=sigma_x, sigma_y=sigma_y, rho=rho, seed=42)
    cov_mat = sample_covariance_matrix(x, y)
    actual_cov = cov_mat[0, 1]
    
    # Cast to raw Python bool explicitly to bypass NumPy boolean type checking errors
    return bool(abs(actual_cov - expected_cov) < 0.15)
