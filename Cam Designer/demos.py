'''Some Demos on How the Cam.py Libaray can be used.

Author: Daniyal Ahsen
'''


#The Only Necessary One
from Cam import consistentProfileFunction, Cam, theta, rise_from_velocity, rise_from_accelaration, CamPair

# The library thinks in rad, so it makes sense to have a deg2rad function at hand
from numpy import deg2rad 

# For Styling the Graphs
from matplotlib.pyplot import style

# Setup styling ; not necessary
style.use(['bmh', 'grid'])

# Specify rise function
rise = theta * 1/deg2rad(90)

# Use this rise function to generate 
# the entire profile
Giza_profile = consistentProfileFunction(
        rise,

        deg2rad(180),
        deg2rad(0),
        deg2rad(180),
        deg2rad(0)
    )

# Setup the Cam!
Giza = Cam(
    Giza_profile,
    name = "Giza Cam"
)

# Plot cam Profile!
Giza.plotSVAJ((deg2rad(0), deg2rad(360)))

# Done for Giza Cam!

# Now let's try something more difficult. What about a parabolic profile.

rise = theta**2 / (deg2rad(90)**2)       # The second part is just there to normalize
quad_profile = consistentProfileFunction(
        rise,
        deg2rad(180),
        deg2rad(0),
        deg2rad(180),
        deg2rad(0)
)
cam = Cam(quad_profile, name="quad")
cam.plotSVAJ(deg2rad((0, 360)))

# Let's add a dwell!

rise = theta**2 
quad_profile = consistentProfileFunction(
        rise,
        deg2rad(150),
        deg2rad(50),
        deg2rad(150),
        deg2rad(0)
)
cam = Cam(quad_profile, name="quad")
cam.plotSVAJ(deg2rad((0, 360)))

# Why have one when you can have two?

rise = theta**2
quad_profile = consistentProfileFunction(
        rise,
        deg2rad(120),
        deg2rad(50),
        deg2rad(120),
        deg2rad(50)
)
cam = Cam(quad_profile, name="quad")
cam.plotSVAJ(deg2rad((0, 360)))

# The third degree's the lucky charm!

rise = (theta**3 + 2 * theta**2 + 1)
quad_profile = consistentProfileFunction(
        rise,
        deg2rad(120),
        deg2rad(50),
        deg2rad(120),
        deg2rad(50)
)

cam = Cam(quad_profile, name="cubic")
cam.plotSVAJ(deg2rad((0, 360)))

# For those who Sin(e)
from sympy import sin


rise = (theta**3 * sin(theta)**2+ 2 * theta**2 + 1)
quad_profile = consistentProfileFunction(
        rise,
        deg2rad(120),
        deg2rad(50),
        deg2rad(120),
        deg2rad(50)
)

cam = Cam(quad_profile, name="Sin")
cam.plotSVAJ(deg2rad((0, 360)))


# Let's throw in the velocity!

velocity_profile = sin(theta)
rise_function    = rise_from_velocity(velocity_profile, 1)
CamOfHellProfile = consistentProfileFunction(rise_function, deg2rad(90), deg2rad(90), deg2rad(90), deg2rad(90))
TheCamOfHell     = Cam(CamOfHellProfile, name="COH")
TheCamOfHell.plotSVAJ(deg2rad((0, 360)))

# Do you feel the accelaration?

from sympy import cos

accelaration_profile = cos(theta)
rise_function        = rise_from_accelaration(accelaration_profile, 1, 90)
CamOfHellProfile    = consistentProfileFunction(rise_function, deg2rad(90), deg2rad(90), deg2rad(90), deg2rad(90))
TheCamOfHell2         = Cam(CamOfHellProfile, name="COH_2")
TheCamOfHell2.plotSVAJ(deg2rad((0, 360)))

# But Why Won't I see the Cam?

TheCamOfHell2.plotProfile()

# Comparison is the root of all Unhapiness.
# Definitely belongs in Hell.

HellCamComparison = CamPair(TheCamOfHell, TheCamOfHell2)
HellCamComparison.compareProfiles()
HellCamComparison.compareSVAJ(deg2rad((0, 360)))
