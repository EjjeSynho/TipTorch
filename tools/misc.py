from scipy.optimize import curve_fit
import torch


class LineModel:
    """
    A PyTorch-compatible line model for non-linear optimization.

    The model function is defined as:
        f(x; k, y_intercept) = k * x + y_intercept
    """

    def __init__(self, x: torch.Tensor):
        """
        Initialize the LineModel.

        Parameters:
            x (torch.Tensor): Tensor of x-values.
        """
        self.x = x

    @staticmethod
    def line_function(x, k, y_intercept):
        """
        Compute the line function.

        Parameters:
            x (Tensor or numpy.ndarray): Input x-values.
            k (float): Slope.
            y_intercept (float): Y-intercept.

        Returns:
            Tensor or numpy.ndarray: Computed y-values.
        """
        return k * x + y_intercept

    def fit(self, y: torch.Tensor):
        """
        Fit the line model to target y-values using non-linear least squares.

        This method converts the input tensors to numpy arrays and then
        uses SciPy's curve_fit to estimate the parameters [k, y_intercept].

        Parameters:
            y (torch.Tensor): Tensor of target y-values.

        Returns:
            array: Optimal parameters [k, y_intercept].
        """
        # Convert tensors to 1D numpy arrays for curve_fit.
        x_np = self.x.flatten().cpu().numpy()
        y_np = y.flatten().cpu().numpy()

        # Define the fitting function.
        def fit_func(x, k, y_intercept):
            return self.line_function(x, k, y_intercept)

        popt, _ = curve_fit(fit_func, x_np, y_np, p0=[1e-6, 1e-6])
        return popt

    def __call__(self, params):
        """
        Evaluate the model using the provided parameters.

        Parameters:
            params (tuple): Parameters (k, y_intercept).

        Returns:
            Tensor: Computed y-values.
        """
        return self.line_function(self.x, *params)


class QuadraticModel:
    """
    A PyTorch-compatible quadratic model for non-linear optimization.

    The model function is defined as:
        f(x; a, b, c) = a * x^2 + b * x + c
    """

    def __init__(self, x: torch.Tensor):
        """
        Initialize the QuadraticModel.

        Parameters:
            x (torch.Tensor): Tensor of x-values.
        """
        self.x = x

    @staticmethod
    def quadratic_function(x, a, b, c):
        """
        Compute the quadratic function.

        Parameters:
            x (Tensor or numpy.ndarray): Input x-values.
            a (float): Quadratic coefficient.
            b (float): Linear coefficient.
            c (float): Constant term.

        Returns:
            Tensor or numpy.ndarray: Computed y-values.
        """
        return a * x ** 2 + b * x + c

    def fit(self, y: torch.Tensor, p0=[1e-6, 1e-6, 1e-6]):
        """
        Fit the quadratic model to target y-values using non-linear least squares.

        This method converts the input tensors to numpy arrays and then
        uses SciPy's curve_fit to estimate the parameters [a, b, c].

        Parameters:
            y (torch.Tensor): Tensor of target y-values.

        Returns:
            array: Optimal parameters [a, b, c].
        """
        # Convert tensors to 1D numpy arrays for curve_fit.
        x_np = self.x.flatten().cpu().numpy()
        y_np = y.flatten().cpu().numpy()

        # Define the fitting function.
        def fit_func(x, a, b, c):
            return self.quadratic_function(x, a, b, c)

        popt, _ = curve_fit(fit_func, x_np, y_np, p0=p0)
        return popt

    def __call__(self, params):
        """
        Evaluate the model using the provided parameters.

        Parameters:
            params (tuple): Parameters (a, b, c).

        Returns:
            Tensor: Computed y-values.
        """
        return self.quadratic_function(self.x, *params)