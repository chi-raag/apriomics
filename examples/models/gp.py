import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
from typing import Dict, Any, Optional, Tuple

# Data container for GP model state
class GPState:
    def __init__(self, 
                 kernel_scale: float = 1.0, 
                 noise: float = 0.1,
                 mcmc = None, 
                 samples = None):
        self.kernel_scale = kernel_scale
        self.noise = noise
        self.mcmc = mcmc
        self.samples = samples

def create_gp_model(kernel_scale: float = 1.0, noise: float = 0.1) -> GPState:
    """
    Create a new GP model state
    
    Parameters:
    -----------
    kernel_scale : float
        Scaling factor for the similarity kernel
    noise : float
        Observation noise standard deviation
        
    Returns:
    --------
    GPState
        Initial GP model state
    """
    return GPState(kernel_scale=kernel_scale, noise=noise)

def gp_model(X, y, similarity_matrix, kernel_scale: float, noise: float):
    """
    Gaussian Process model using similarity matrix as prior covariance.
    
    Parameters:
    -----------
    X : array-like
        Input features (n_samples, n_features) 
    y : array-like
        Target values (n_samples,)
    similarity_matrix : array-like
        Pre-computed similarity matrix between metabolites
    kernel_scale : float
        Scaling factor for the similarity kernel
    noise : float
        Observation noise standard deviation
    """
    # Scale similarity matrix to create kernel
    kernel = kernel_scale * similarity_matrix
    
    # Add small diagonal term for numerical stability
    kernel = kernel + 1e-6 * jnp.eye(kernel.shape[0])
    
    # Sample from GP prior
    f = numpyro.sample(
        "f",
        dist.MultivariateNormal(
            loc=jnp.zeros(X.shape[0]),
            covariance_matrix=kernel
        )
    )
    
    # Sample observations with noise
    numpyro.sample(
        "y",
        dist.Normal(f, noise),
        obs=y
    )

def fit_gp(state: GPState, 
           X, 
           y, 
           similarity_matrix, 
           num_warmup: int = 500, 
           num_samples: int = 1000, 
           random_seed: int = 0) -> GPState:
    """
    Fit the GP model using MCMC.
    
    Parameters:
    -----------
    state : GPState
        GP model state
    X : array-like
        Input features (n_samples, n_features)
    y : array-like
        Target values (n_samples,)
    similarity_matrix : array-like
        Pre-computed similarity matrix between metabolites
    num_warmup : int
        Number of warmup steps for MCMC
    num_samples : int
        Number of samples to draw
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    GPState
        Updated GP model state with MCMC samples
    """
    # Set random seed for reproducibility
    rng_key = random.PRNGKey(random_seed)
    
    # Create model function with fixed parameters
    model_fn = lambda X, y, similarity_matrix: gp_model(
        X, y, similarity_matrix, state.kernel_scale, state.noise
    )
    
    # Initialize MCMC sampler
    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    
    # Run MCMC
    mcmc.run(
        rng_key, 
        X=jnp.array(X), 
        y=jnp.array(y), 
        similarity_matrix=jnp.array(similarity_matrix)
    )
    
    # Get samples
    samples = mcmc.get_samples()
    
    # Return updated state
    return GPState(
        kernel_scale=state.kernel_scale,
        noise=state.noise,
        mcmc=mcmc,
        samples=samples
    )

def predict_gp(state: GPState, 
               similarity_matrix_new: Optional[Any] = None) -> Dict[str, Any]:
    """
    Make predictions using the fitted GP model.
    
    Currently supports predictions on the training set only.
    Future versions will support predictions on new data points
    using the similarity matrix.
    
    Parameters:
    -----------
    state : GPState
        Fitted GP model state
    similarity_matrix_new : array-like, optional
        Similarity matrix for new data points with respect to training points.
        Not yet implemented.
        
    Returns:
    --------
    dict
        Dictionary containing prediction statistics (mean, std, etc.)
    """
    if state.samples is None:
        raise ValueError("Model not fitted. Run fit_gp() first.")
        
    if similarity_matrix_new is not None:
        raise NotImplementedError(
            "Prediction on new data points not yet implemented. "
            "Currently only supports predictions on training set."
        )
    
    # Get posterior samples of f
    f_samples = state.samples["f"]
    
    # Calculate prediction statistics
    pred_mean = jnp.mean(f_samples, axis=0)
    pred_std = jnp.std(f_samples, axis=0)
    pred_interval = jnp.percentile(f_samples, jnp.array([2.5, 97.5]), axis=0)
    
    return {
        "mean": pred_mean,
        "std": pred_std,
        "lower_ci": pred_interval[0],
        "upper_ci": pred_interval[1],
        "samples": f_samples
    }

# For backward compatibility
class ChemPriorGP:
    """
    Gaussian Process model using chemical similarity priors.
    
    This class provides a wrapper around the functional GP implementation
    for backward compatibility.
    """
    
    def __init__(self, kernel_scale=1.0, noise=0.1):
        """
        Initialize the Gaussian Process model.
        
        Parameters:
        -----------
        kernel_scale : float
            Scaling factor for the similarity kernel
        noise : float
            Observation noise standard deviation
        """
        self.state = create_gp_model(kernel_scale, noise)
    
    def model(self, X, y, similarity_matrix):
        """
        Gaussian Process model using similarity matrix as prior covariance.
        
        Parameters:
        -----------
        X : array-like
            Input features (n_samples, n_features) 
        y : array-like
            Target values (n_samples,)
        similarity_matrix : array-like
            Pre-computed similarity matrix between metabolites
        """
        gp_model(X, y, similarity_matrix, self.state.kernel_scale, self.state.noise)
    
    def fit(self, X, y, similarity_matrix, num_warmup=500, num_samples=1000, random_seed=0):
        """
        Fit the GP model using MCMC.
        
        Parameters:
        -----------
        X : array-like
            Input features (n_samples, n_features)
        y : array-like
            Target values (n_samples,)
        similarity_matrix : array-like
            Pre-computed similarity matrix between metabolites
        num_warmup : int
            Number of warmup steps for MCMC
        num_samples : int
            Number of samples to draw
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        self
            For method chaining
        """
        self.state = fit_gp(
            self.state, X, y, similarity_matrix, 
            num_warmup, num_samples, random_seed
        )
        return self
    
    def predict(self, similarity_matrix_new=None):
        """
        Make predictions using the fitted GP model.
        
        Parameters:
        -----------
        similarity_matrix_new : array-like, optional
            Similarity matrix for new data points with respect to training points.
            Not yet implemented.
            
        Returns:
        --------
        dict
            Dictionary containing prediction statistics (mean, std, etc.)
        """
        return predict_gp(self.state, similarity_matrix_new) 