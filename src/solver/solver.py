<<<<<<< HEAD
=======
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 14:48:12 2025

@author: INDRAJIT
"""

>>>>>>> master
class solver:
    """
    Base class for ODE solvers.
    This class defines the interface for ODE solvers.
    Subclasses must implement the solve() method.
    """
<<<<<<< HEAD
    def __init__(self, func, ini_cond, t_final, num_points, *args, **kwargs):
=======
    def __init__(self, func, *args, **kwargs):
>>>>>>> master
        """
        Initialize the ODE solver with the function, initial conditions,
        final time, number of points, and any additional arguments.
        
        Parameters:
            func       : callable f(t, y, *args, **kwargs) returning dy/dt
<<<<<<< HEAD
            ini_cond   : list or array of initial values [y1(0), y2(0), ..., yN(0)]
            t_final    : final time up to which you want the solution
            num_points : number of time points between 0 and t_final
=======
>>>>>>> master
            args       : any extra positional arguments needed by func
            kwargs     : any extra keyword arguments needed by func
        """
        self.func = func
<<<<<<< HEAD
        self.ini_cond = ini_cond
        self.t_final = t_final
        self.num_points = num_points
        self.args = args
        self.kwargs = kwargs

    def solve(self):
        """
        Solve the ODE from t=0 to t=t_final with initial conditions ini_cond.
        
=======
        self.args = args
        self.kwargs = kwargs

    def solve(self, ini_cond, t_final, num_points):
        """
        Solve the ODE from t=0 to t=t_final with initial conditions ini_cond.
        
        Parameters:
            ini_cond   : list or array of initial values [y1(0), y2(0), ..., yN(0)]
            t_final    : final time up to which you want the solution
            num_points : number of time points between 0 and t_final
        
>>>>>>> master
        Returns:
            t : np.ndarray of time points
            Y : np.ndarray of shape (num_points, len(ini_cond)) with the solution
        """
<<<<<<< HEAD
        raise NotImplementedError("Subclasses must implement this method.")
=======
        raise NotImplementedError("Subclasses must implement this method.")
>>>>>>> master
