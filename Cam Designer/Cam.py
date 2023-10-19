'''

Description:
-----------
    This library is intended as a Cam Design Toolkit to quickly Design Cams.
    This librray provides two class objects designed to achieve this purpose.
    These are:

    1. Cam:
       ----
        A class to make cam designs given that a profile is provided in the form:

            s = f(theta)

    2. CamPair:
       -------
       This class can be initialized to quickly compare two cam designs for differences in their s-theta, v-theta, a-theta and j-theta plots.


    In addition to these constructs, a variety of other constructs are provided, for instance, computing an entire cam profile using just the "rise function", ie. the s-theta displacemment function during the rise portion of the motion is tedious task. To help with this, the function consistentProfileFunction is provided which produces a "consistent" profile, one that has similar features according to a criterion the user can select or provide themeselves. The default implementation makes the accelaration function same and continuous on both sides of the dwell.


    Another couple of constructs that are provided by this library are:

    1) rise_form_velocity:
        This function produces a rise function by integrating a provided velocity Profile.

    2) rise_form_accelaration:
            This function produces a rise function by integrating a provided accelaration profile.

'''

# Import the necessary modules
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import scienceplots
import sys


#The Symbol "Theta" is used explicitly throughout the code
theta = sym.symbols(r'theta')


class Cam:
    
    """
    
    This class creates a Cam with a specifed Displacement Function and a specified
    angular velocity for the cam . The displacement function is then symbolically 
    differentiated to produce the SVAJ plots for the Cam. 


    Methods:
    -------

    1) vectorizeSVAJ:
        ------------
            This function numpy arrays of the SVAJ functions 
    
    2) plotSVAJ:
        --------
            Produces a plot of the S,V,A and J profiles of the cam.
            calls vectorizeSVAJ to get numpy arrays of these functions.

    3) plotProfile:
        -----------
            Plots a profile of this function.
    """
    
    def __init__(self, s, omega=1, name=""):
        self.s = s
        self.v = sym.Derivative(self.s, theta, evaluate=True) * omega
        self.a = sym.Derivative(self.v, theta, evaluate=True) * omega
        self.j = sym.Derivative(self.a, theta, evaluate=True) * omega
        
        # name of the cam
        self.name = name
        

    # parts specifies the region of the displacement cruve to be plotted
    # chose parts=1 for plotting the entire range
    def plotSVAJ(self, theta_range, savefile=False,
                filename="SVAJ-", fileformat="png",
                display=True):

        # If using the default filename then append the cam
        # name to it
        if filename=="SVAJ-":
            filename += self.name
        
        # Specifies the part of the *theta* domain of intrest
        # for instance, it is 0 to pi/2 by default
        THETA = np.linspace(theta_range[0], theta_range[1] , 200)

        # Convert each of the sympy functions into regular python functions
        # And evaluate them over each THETA
        S, V, A, J = self.vectorizeSVAJ(theta_range)


        # Setup Figure for plotting
        fig, ax = plt.subplots(4, 1, figsize=(10,20))

        # Plot Each of the Function, S, V, A, J and attach appropriate labels
        labels = ['S', 'V', 'A', 'J']
        for i, j in enumerate(labels):
            labels[i] = '$'+labels[i]+'_{'+ self.name + '}$'
        
        plotting_variables = [S, V, A, J]

        for i in range(4):
            ax[i].plot(THETA, plotting_variables[i], label=labels[i])


        # Set labels for X & Y axes
        Ylabels = [r'$S\left(m\right)$',
                       r'$V \left(\frac{m}{s}\right)$', 
                       r'$A \left(\frac{m}{s^2}\right)}$',
                       r'$J \left(\frac{m}{s^3}\right)}$']
            
        for j,i in enumerate(ax):
            i.set_ylabel(Ylabels[j])
            i.set_xlabel(r'$\theta(rad)$')
            i.legend(loc='upper right')

        # Attach a Superttile
        fig.suptitle("Plot of Kinematic Properites of " + self.name + " Displacement Funciton", fontsize=20)
        
        # Save Figure To Files If you have been asked to do so.
        if savefile:
            plt.savefig(filename + '.' + fileformat) # Save the result as an image
        
        if display:
            plt.show()

    
    # R_internal is the radius of the base circle for the cam
    def plotProfile(self, Rinternal=5, savefile=False, filename="CAM-",
                    fileformat="png", display=True):
        
        """
        
        Description:
        ===========

        Plots the Cam Profile. Rinternal specifies the radius of the base circle.
        
        Arguments:
        ==========

            filename:
                     is the filename of the CAM. If his is left untouched, the filename
        is automatically changed to filename-Camname. This option is only relevant when saving
        alongwith fileformat variable.
        
            display: 
                    Stores whether to display the Cam Function in a plot, possible in a separate window
        or not.
        
        """
        
        if filename=="CAM-":
            filename += self.name
        
              # give s a shorter name
        s = self.s

            # Make a figure and set the projection to polar
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(5,5))
        fig.suptitle('Cam Profile for ' + self.name)

        # remove x & y ticks & grid
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

              # remove the axis spines to clear the figure
        ax.spines['polar'].set_visible(False)

              # Make a polar plot of the function
        THETA = np.linspace(0, 2 * np.pi, 100)
        R     = np.vectorize(lambda x: s.subs(theta,x))(THETA) + Rinternal

        ax.plot(THETA, R)
        
        if savefile:
            plt.savefig(filename+'.'+fileformat)

        if display:
            plt.show()
                        
    def vectorizeSVAJ(self, theta_range):
        """
        
        Returns numpy arrays containing vectorized values of the SVAJ functions over theta_range.
        
        Arguments:
        
        theta_range:
            The Values of theta over which the S, V, A & J functions are evalulated.
            
        """
        THETA = np.linspace(theta_range[0], theta_range[1] , 200)

        S     =     np.vectorize(sym.lambdify(theta, self.s, ["numpy", "scipy"]))(THETA)
        V     =     np.vectorize(sym.lambdify(theta, self.v, ["numpy", "scipy"]))(THETA)
        A     =     np.vectorize(sym.lambdify(theta, self.a, ["numpy", "scipy"]))(THETA)
        J     =     np.vectorize(sym.lambdify(theta, self.j, ["numpy", "scipy"]))(THETA)
        
        return [S, V, A, J]

    '''
        S     = sym.lambdify(theta, self.s, ["numpy", "scipy"])
        V     = sym.lambdify(theta, self.v, ["numpy", "scipy"])
        A     = sym.lambdify(theta, self.a, ["numpy", "scipy"])
        J     = sym.lambdify(theta, self.j, ["numpy", "scipy"])
        
        #print(inspect.getsource(S))

        S     = S(THETA)
        V     = V(THETA)
        A     = A(THETA)
        J     = J(THETA)

        return [S, V, A, J]
    '''
class CamPair:
    
    """
    
    Stores a pair of Cams for comparing them etc. It doensn't make much sense
    to compare more than 2 Cams at a time so I haven't coded a camarray class.
    
    """
    
    def __init__(self, Cam1, Cam2):
        self.cam1 = Cam1
        self.cam2 = Cam2
        
        
    def compareSVAJ(self, angular_range, savefile=False, filename="COMPARE-SVAJ", fileformat="png", display=True):
        
        """
        
        compareSVAJ compares the plot of S, V, A, J. The result is a plot of S, V, A and J against
        theta.
        
        Arguments:
        
        1. angular_range:
            A tuple (theta_minimum, theta_maximum)
            
            Specifies the theta range for plotting.
            
        2. savefile:
                Whether to Save the result as a file or not.
                
        3. filename:
                The name of the resulting file.
                
                Note:
                
                    If this argument is left to default then the resulting filename is automatically changed to COMPARE-SVAJ-<CAM1.NAME>-<CAM2.NAME>
                
        4. fileformat:
        
                    The fileformat for this file to be saved.
                    
        5. display:
        
                    Specifies whether the result is to be displayed or not.
        
        
        """
        
        if filename=="COMPARE-SVAJ":
            filename="COMPARE-SVAJ-"+self.cam1.name + "-" + self.cam2.name
        
        # Specifies the part of the *theta* domain of intrest
        # for instance, it is 0 to pi/2 by default
        THETA = np.linspace(angular_range[0], angular_range[1] , 200)

        # Convert each of the sympy functions into regular python functions
        # And evaluate them over each THETA
        S, V, A, J = self.cam1.vectorizeSVAJ([THETA[0], THETA[-1]])
        S2, V2, A2, J2 = self.cam2.vectorizeSVAJ([THETA[0], THETA[-1]])

        
        # Setup Figure for plotting
        fig, ax = plt.subplots(4, 1, figsize=(10,20))

        # Plot Each of the Function, S, V, A, J and attach appropriate labels
        labels = [('$S_{'+self.cam1.name+'}$', '$S_{'+self.cam2.name+'}$'),
                  ('$V_{'+self.cam1.name+'}$', '$V_{'+self.cam2.name+'}$'),
                  ('$A_{'+self.cam1.name+'}$', '$A_{'+self.cam2.name+'}$'), 
                  ('$J_{'+self.cam1.name+'}$', '$J_{'+self.cam2.name+'}$')]
        
        
        plotting_variables = [(S, S2), (V, V2), (A, A2), (J, J2)]

        for i in range(4):
            for k, j in enumerate(plotting_variables[i]):
                ax[i].plot(THETA, j, label=labels[i][k])


        # Set labels for X & Y axes
        for j,i in enumerate(ax):
            Ylabels = [r'$S\left(m\right)$', r'$V \left(\frac{m}{s}\right)$', r'$A \left(\frac{m}{s^2}\right)$', r'$J \left(\frac{m}{s^3}\right)$']
            
            i.set_ylabel(Ylabels[j])
            i.set_xlabel(r'$\theta(rad)$')
            i.legend(loc='upper right')

        # Attach a Superttile
        fig.suptitle("Comparison of Kinematic Properites of " + self.cam1.name + " and " +
                     self.cam2.name + " Displacment Funcitons", fontsize=20)
        
        if savefile:
            plt.savefig(filename + '.' + fileformat) # Save the result as an image
        
        if display:
            plt.show()

        
    def compareProfiles(self, thetamin=0, thetamax=180, Rinternal=5, h=1, savefile=False,
                        filename="CompareProfiles", fileformat='png',
                       display=True):
        ''' Rinternal is the radius of the base circle 
            h provides a scaling factor for the cam excepting the base
            cricle to allow clearer distinction between two different cam types'''
        
        # Setup the displacement functions
        s = self.cam1.s 
        s2 = self.cam2.s
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(projection='polar'))

        # Remove axes grid & everything else
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        #ax.set_thetamin(thetamin)
        #ax.set_thetamax(thetamax)

        ax.set_rmax(5 + h)
        ax.set_rmin(5 - h)

        ax.spines['polar'].set_visible(False)

        #THETA  = np.linspace((beta[0]+beta[1] -dtheta/2) * np.pi/180, (beta[0]+beta[1]+dtheta/2) * np.pi/180 + np.pi/2 , 100)
        THETA  = np.linspace(0,2 * np.pi,100)
        R      = np.vectorize(lambda x: s2.subs(theta,x))(THETA) + Rinternal
        R2     = np.vectorize(lambda x: s.subs(theta,x))(THETA) + Rinternal

        ax.plot(THETA, R, label=self.cam1.name)
        ax.plot(THETA, R2, '--', label=self.cam2.name)
        ax.legend(loc='lower left')  

        fig.suptitle("Comparison of Plots For " + self.cam1.name + " And " + 
                     self.cam2.name + " Displacement Functions")
        
        if savefile:
            plt.savefig(filename+'.'+fileformat)

        if display:
            plt.show()
            

def consistentProfileFunction(

        rise, 
    
        theta_rise, 
        theta_dwell, 
        theta_fall, 
        theta_dwell2,

        method= "dwell_and_rise_adjusted"
    
    ):

    """

    Description:
    ===========

    This function produces a cam function that has all the rise, dwell and fall segments integrated into it.
    So it would be practical to pass this as an argument to Cam class' constructor.

    Arguments:
    =========

    rise: The rise function. This has to be provided by the user. Rise should be a function of theta.
    theta_rise: The theta_rise is the rise angle rising through which the cam reaches the maximum height.
    theta_dwell: The angle, after having traversed the theta_rise, the function stays in the dwell period for  
    theta_fall: The angle through which the cam falls to the base level
    theta_dwell2: The angle for the second dwell, in a similar fashion to the above.
    
    Note:
    =====

    It is important to note that:

    theta_rise + theta_dwell + theta_fall + theta_dwell2 = 2 * Pi

    """    

    def fall(method="dwell_and_rise_adjusted"):
        """ Calculate the Fall Function.
        
        How this works:
        ==============

        Assume that the rise function is at least 2nd order differentiable.
        Let r be the rise function, r' is the derivative. r'' is the second derivative.
        Assume that the r'' function is the same for the rise and fall. 

        (Rise and fall should have same accelaration profiles)

        Let's call the fall function f. So we have, the following ODE to solve.

        f'' = r''                                   (1)

        Boundary Conditions:
        ==================

        f(theta_dwell+ theta_rise) = r(theta_rise)   (A)
        f(theta_dwell+theta_rise+theta_fall) = 0     (B)
        
        Use sympy to solve this second order ODE. This is implemented below.


        Methods:
        ========

        1) Standard:
            -------

            The standard method simply solves the ODE using a claculated accelaration function.
            This is highly NOT Recommended!

        2) Dwell and Rise Adjusted (dwell_and_rise_adjusted):
            -------------------------------

            This method adjusts the accelaration profile to match on either side of the first dwell.
            This yeilds a solution that is much more pleasing to the eye.

            This has been implemented as the default method.

            The method is as follows:
            ------------------------

            The following ODE is solved in this case:

                r'' = f''(theta - (theta_rise + theta_dwell) * (theta_rise/theta_dwell))        (1)

            The last multipliction encountered in f'' adjusts the domain so that smoothness of the solution
            is not affected by the irregularities in domain size.


        3) To be Implemented.... 
        """
        
        # ** ODE specified as a equation evaluating to zero**

        f   = sym.Function("f")

        if method=="Standard":
            ode = sym.Derivative(f(theta), theta, theta) - sym.Derivative(rise, theta, theta)
        elif method=="dwell_and_rise_adjusted":
            ode = sym.Derivative(f(theta), theta, theta) - sym.Derivative(rise, theta, theta).subs(
                    theta,
                    (theta_rise/theta_fall) * (theta-theta_dwell-theta_rise)
                )    
        else:
            raise RuntimeError("Invalid Choice of Method in fall() called by symmetricCamProfile. Please Use a Valid Method")


        # ** ICs and BCs specified as f(x) - k = 0 */
        # Just need to fix this all is fine otherwise
        # calling rise here will lead to an error - will fix this now
        
        IC = rise.subs(theta, theta_rise)
        BC = 0

        sol =  sym.dsolve(ode, f(theta), ics={
            f(theta).subs(theta, theta_rise+theta_dwell): IC,
            f(theta).subs(theta, theta_rise+theta_dwell+theta_fall): BC
        }).rhs
        
        return sol

    
    fall_function = fall(method)

    # Return a piecewise sympy function that contains the entire profile
    return sym.Piecewise(
            (rise, theta <= theta_rise),
            (rise.subs(theta, theta_rise), theta <= theta_rise + theta_dwell),
            (fall_function, theta <= theta_rise + theta_dwell + theta_fall),
            (0,  theta <= 2 * np.pi)
        )

def rise_from_accelaration(accelaration, maximum_rise, theta_rise):
    
    '''
    This Function Helps in Setting Up a Rise Function That has a Specific Accelaration Function.


    Notice that the rise_form_accelaration function produces a rise proifle and enforces two boundary conditions:

    1) Rise(Theta=0) = 0
    2) Rise(Theta=1) = maximum_rise

    '''

    f = sym.symbols(r'f', cls=sym.Function)
    ode = sym.Derivative(f(theta), theta, theta) - accelaration

    sol = sym.dsolve(
        ode, f(theta), ics={

            f(theta).subs(theta, 0) : 0,
            f(theta).subs(theta, theta_rise) : maximum_rise

        }
    ).rhs

    return sol

def rise_from_velocity(velocity, maximum_rise=None, theta_rise=np.deg2rad(90)):

    '''
    This funcciton sets up a  Rise Function with respect to a provided velocity Function. Notice that this function only enforces the initial condition,

    Rise(Theta=0) = 0

    The second condition ie.
        Rise(Theta= Theta_Rise) = Height (Or Whatever you would like to call it)

    Can be enforced using the maximum_rise parameter. Specifying a maximum_rise will force the downscaling of the cam profile so that it respects the maximum_rise condition.

    '''

    f = sym.symbols(r'f', cls=sym.Function)
    ode = sym.Derivative(f(theta), theta) - velocity

    sol = sym.dsolve(
        ode, f(theta), ics={
            f(theta).subs(theta, 0) : 0
        }
    ).rhs
    

    if not(maximum_rise is None):
        ''' Rescale '''
        sol *= (maximum_rise)/(sol.subs(theta, theta_rise))


    return sol

'''
    <-- End -->
'''