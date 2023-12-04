"""
:module ebm3: an energy-based 3-layer radiative transfer model based on a
    transformation matrix relating layerwise temperature to layerwise
    absorption of solar insolation. The matrix parameters are solved for
    exactly using constants derived from the radiative transfer equation.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Layer:
    """
    The Layer class is an abstraction on a 2d plane-parallel atmospheric
    layer with the provided absorption, transmission, and reflection
    coefficients. Validation enforces the property that all components
    must be a real number in in [0,1], and

    :@param a: Absorption coefficient (float in [0,1])
    :@param t: Transmission coefficient (float in [0,1])
    :@param r: Reflection coefficient (float in [0,1])
    """
    a:float
    t:float
    r:float

    def validate(self, tolerance=1e-08):
        """
        All the Layer's components must sum to ~1 and be a real number in [0,1]

        :@param tolerance: Optional change in threshold of how close the sum of
            component magnitudes needs to be to 1.
        :@return: True if valid (raises AssertionError otherwise).
        """
        components = tuple(map(float, (self.a, self.t, self.r)))
        assert all(0<=c<=1 for c in components)
        assert abs(sum(components)-1) <= tolerance
        return True

def sw_abs_l1(sw1:Layer, sw2:Layer, sw3:Layer, Q=341.0):
    """
    Given 3 layers of (absorption, transmittance, reflectance) coefficients,
    returns the shortwave absorption in the first Layer as a scalar in W/m^2

    :@param sw1: Layer representing the highest altitude properties
    :@param sw2: Layer representing the middle altitude properties
    :@param sw3: Layer representing the surface properties
    :@param Q: Average surface insolation (ie S_0/4 in W/m^2)
    """
    [layer.validate() for layer in (sw1, sw2, sw3)]
    c1 = sw1.t * (sw2.r*(1-sw2.r*sw3.r) + sw2.t**2*sw3.r)
    c2 = (1-sw1.r*sw2.r)*(1-sw2.r*sw3.r)-sw2.t**2*sw3.r*sw1.r
    return Q * sw1.a * (1 + c1/c2)

def lw_emit_l1(lw1:Layer, lw2:Layer, lw3:Layer):
    """
    Returns an vector of coefficients (v1, v2, v3) corresponding to layers 1-3
    (counted top-down) of ebm3. The inner product of this vector with the
    layerwise 4th-power temperature vector (T1^4, T2^4, T3^4) returns the
    longwave net loss for all 4 layers.

    This vector forms the first row of the transform matrix for solving
    for the temperature layers

    S_1 = V1*T1^4 + V2*T2^4 + V3*T3^4

    :@param lw1: Layer representing the highest altitude longwave properties
    :@param lw2: Layer representing the middle altitude longwave properties
    :@param lw3: Layer representing the surface longwave properties
    """
    [layer.validate() for layer in (lw1, lw2, lw3)]
    sigma = 5.670373e-8
    c1 = 2
    c2 = lw1.a
    c3 = lw2.r*(1-lw2.r*lw3.r) / \
            ((1-lw1.r*lw2.r)*(1-lw2.r*lw3.r) - lw2.t**2*lw3.r*lw1.r)
    c4 = lw1.r
    c5 = lw2.t*lw1.r/(1-lw2.r*lw3.r)
    c6 = lw2.t*lw1.r*lw3.r/(1-lw2.r*lw3.r)
    c7 = lw2.t*(1-lw1.r*lw2.r) / \
            ((1-lw1.r*lw2.r)*(1-lw2.r*lw3.r)-lw2.t**2*lw3.r*lw1.r)
    c8 = lw2.t*lw3.r / (1-lw1.r*lw2.r)
    c9 = lw2.t*lw3.r*lw1.r / (1-lw1.r*lw2.r)
    c10 = lw3.r

    V1 = lw1.a * sigma * (c1-c2*c3-c2*c7*c8)
    V2 = lw2.a * sigma * (-c2-c2*c3*c4-c2*c3*c6-c2*c7*c9-c2*c7*c10)
    V3 = lw3.a * sigma * (-c2*c3*c5-c2*c7)
    return np.array((V1, V2, V3))

def sw_abs_l2(sw1:Layer, sw2:Layer, sw3:Layer, Q=341.0):
    """
    Given 3 layers of (absorption, transmittance, reflectance) coefficients,
    returns the shortwave absorption in the second Layer as a scalar in W/m^2

    :@param sw1: Layer representing the highest altitude properties
    :@param sw2: Layer representing the middle altitude properties
    :@param sw3: Layer representing the surface properties
    :@param Q: Average surface insolation (ie S_0/4 in W/m^2)
    """
    [layer.validate() for layer in (sw1, sw2, sw3)]
    return Q * sw1.t*sw2.a*(1-sw2.r*sw3.r+sw2.t*sw3.r) / \
            ((1-sw1.r*sw2.r)*(1-sw2.r*sw3.r) - sw2.t**2*sw3.r*sw1.r)


def lw_emit_l2(lw1:Layer, lw2:Layer, lw3:Layer):
    """
    Returns an vector of coefficients (v1, v2, v3) corresponding to layers 1-3
    (counted top-down) of ebm3. The inner product of this vector with the
    layerwise 4th-power temperature vector (T1^4, T2^4, T3^4) returns the
    longwave net loss for all 4 layers.

    S_2 = V1*T1^4 + V2*T2^4 + V3*T3^4

    This vector forms the second row of the transform matrix for solving
    for the temperature layers

    :@param lw1: Layer representing the highest altitude longwave properties
    :@param lw2: Layer representing the middle altitude longwave properties
    :@param lw3: Layer representing the surface longwave properties
    """
    [layer.validate() for layer in (lw1,lw2,lw3)]
    sigma = 5.670373e-8
    c1 = 2
    c2 = -lw2.a / ((1-lw1.r*lw2.r)*(1-lw2.r*lw3.r) - lw2.t**2*lw3.r*lw1.r)
    c3 = 1 - lw2.r*lw3.r + lw2.t*lw3.r
    c4 = lw1.r
    c5 = 1 - lw1.r*lw2.r + lw2.t*lw1.r
    c6 = lw3.r

    V1 = lw1.a * sigma * (c2*c3)
    V2 = lw2.a * sigma * (c1+c2*c3*c4+c2*c5*c6)
    V3 = lw3.a * sigma * (c2*c5)
    return np.array((V1, V2, V3))

def sw_abs_l3(sw1:Layer, sw2:Layer, sw3:Layer, Q=341.0):
    """
    Given 3 layers of (absorption, transmittance, reflectance) coefficients,
    returns the shortwave absorption in the ground layer as a scalar in W/m^2

    :@param sw1: Layer representing the shortwave highest altitude properties
    :@param sw2: Layer representing the shortwave middle altitude properties
    :@param sw3: Layer representing the shortwave surface properties
    :@param Q: Average surface insolation (ie S_0/4 in W/m^2)
    """
    [layer.validate() for layer in (sw1, sw2, sw3)]
    c1 = (1-sw1.r*sw2.r)*(1-sw2.r*sw3.r)-sw2.t*sw3.r*sw2.t*sw1.r
    c2 = sw1.t * sw3.a * (sw2.t*(1-sw2.r*sw3.r) + sw2.r*sw2.t*sw3.r)
    return Q*c2/c1

def lw_emit_l3(lw1:Layer, lw2:Layer, lw3:Layer):
    """
    Returns an vector of coefficients (v1, v2, v3) corresponding to layers 1-3
    (counted top-down) of ebm3. The inner product of this vector with the
    layerwise 4th-power temperature vector (T1^4, T2^4, T3^4) returns the
    longwave net loss for all 4 layers.

    This vector forms the third row of the transform matrix for solving
    for the temperature layers

    S_3 = V1*T1^4 + V2*T2^4 + V3*T3^4

    :@param lw1: Layer representing the highest altitude longwave properties
    :@param lw2: Layer representing the middle altitude longwave properties
    :@param lw3: Layer representing the surface longwave properties
    """
    [layer.validate() for layer in (lw1,lw2,lw3)]
    sigma = 5.670373e-8
    ca = ((1-lw1.r*lw2.r)*(1-lw2.r*lw3.r)-lw2.t*lw3.r*lw2.t*lw1.r)

    c1 = 1
    c2 = -lw3.a
    c3 = lw2.t/ca
    c4 = 1-lw2.r*lw3.r
    c5 = lw1.r
    c6 = lw2.t*lw1.r
    c7 = lw3.r
    c8 = lw2.r/ca
    c9 = lw2.t*lw3.r
    c10 = 1-lw1.r*lw2.r

    V1 = lw1.a * sigma * (c2*c3*c4 + c2*c8*c9)
    V2 = lw2.a * sigma * (c2 + c2*c3*c4*c5 + c2*c3*c6*c7 + \
            c2*c8*c9*c5 + c2*c8*c10*c7)
    V3 = lw3.a * sigma * (c1 + c2*c3*c6 + c2*c8*c10)
    return np.array((V1, V2, V3))

def sw_ref_sfc(sw_layers:tuple, Q=341.0):
    """
    Get the total amount of shortwave radiation reflected by the surface

    :@param sw_layers: 3-tuple of Layer objects representing upper, lower,
        and surface layers' shortwave radiative properties.
    """
    assert all(type(sw) is Layer for sw in sw_layers)
    assert all(sw.validate() for sw in sw_layers)
    sw1, sw2, sw3 = tuple(sw_layers)
    c1 = sw1.t**2*sw2.t**2*sw3.r
    c2 = (1-sw1.r*sw2.r)*(1-sw2.r*sw3.r) - sw2.t**2*sw3.r*sw1.r
    return Q * c1/c2

def sw_ref_atmo(sw_layers:tuple, Q=341.0):
    """
    Get the total amount of shortwave radiation reflected by the atmosphere
    """
    assert all(type(sw) is Layer for sw in sw_layers)
    assert all(sw.validate() for sw in sw_layers)
    sw1, sw2, sw3 = tuple(sw_layers)
    c1 = sw1.t**2*sw2.r*(1-sw2.r*sw3.r)
    c2 = ((1-sw1.r*sw2.r)*(1-sw2.r*sw3.r)-sw2.t**2*sw3.r*sw1.r)
    return  Q * (sw1.r + c1/c2)

def emit_sfc(lw_layers:tuple, sw_layers:tuple=(None, None, None),
             skin_temp=None, Q=341.0):
    """
    Get the total amount of longwave emissions transmitted directly to space.

    This function is highly dependent on surface temperature. By default, the
    surface temperature is calculated using the layer parameters, but if
    skin_temp is defined, it replaces the ground temperature in the function,
    but uses the same layer parameters (rather than applying a boundary).

    :@param lw_layers: 3-tuple of Layer objects representing upper, lower,
        and surface layers' longwave radiative properties.
    :@param sw_layers: 3-tuple of Layer objects representing upper, lower,
        and surface layers' shortwave radiative properties.
    :@param skin_temp: Surface temperature; see note above.
    """
    sigma = 5.670373e-8
    assert all(type(lw) is Layer for lw in lw_layers)
    assert all(type(sw) is Layer for sw in sw_layers if not sw is None)
    assert all(lw.validate() for lw in lw_layers)
    assert all(sw.validate() for sw in sw_layers if not sw is None)
    assert len(sw_layers) == len(lw_layers) == 3

    if skin_temp is None:
        if not all([not s is None for s in sw_layers]):
            raise ValueError(f"If you don't provide a skin temperature, " + \
                    "you must define shortwave layers sw1, sw2, and sw3.")
        skin_temp = layer_temps(lw_layers,sw_layers, Q)[2]
    lw1, lw2, lw3 = tuple(lw_layers)
    c1 = lw1.t*lw2.t*lw3.a*sigma*skin_temp**4
    c2 = (1-lw1.r*lw2.r)*(1-lw2.r*lw3.r) - lw2.t**2*lw3.r*lw1.r
    return c1/c2

def emit_atmo(lw_layers:tuple, sw_layers:tuple, Q=341.0):
    """
    Total longwave emissions from the atmosphere into space

    :@param lw_layers: 3-tuple of Layer objects representing upper, lower,
        and surface layers' longwave radiative properties.
    :@param sw_layers: 3-tuple of Layer objects representing upper, lower,
        and surface layers' shortwave radiative properties.
    :@param Q: Average surface insolation (ie S_0/4 in W/m^2)
    """
    sigma = 5.670373e-8
    assert all(type(lw) is Layer for lw in lw_layers)
    assert all(lw.validate() for lw in lw_layers)
    assert len(lw_layers)==3
    lw1, lw2, lw3 = lw_layers
    sw1, sw2, sw3 = sw_layers
    temps = layer_temps(lw_layers, sw_layers, Q)
    c1 = 1 + lw1.t*lw2.r*(1-lw2.r*lw3.r) + lw2.t**2*lw3.r
    c2 = lw1.t*(1+lw1.r*lw2.r*(1-lw2.r*lw3.r)+lw2.t*lw3.r*(1-lw1.r*lw2.t))
    return np.array([sigma * lw1.a * temps[0]**4 * c1,
                     sigma * lw2.a * temps[1]**4 * c2,
                     0])

def layer_sw_abs(sw_layers:tuple, Q=341.0):
    """
    Get the shortwave absorption in W/m^2 for each atmospheric level given
    the radiative charactaristics of each level and insolation Q in W/m^2

    :@param sw_layers: 3-tuple of valid Layer objects representing shortwave
        radiative properties from the top layer down
    :@param Q: Shortwave insolation in W/m^2
    """
    assert all(type(sw) is Layer for sw in sw_layers)
    assert all(sw.validate() for sw in sw_layers)
    sw_transforms = (sw_abs_l1, sw_abs_l2, sw_abs_l3)
    return np.array([f(*sw_layers, Q) for f in sw_transforms])

def layer_temps(lw_layers:tuple, sw_layers:tuple, Q=341.0):
    """
    Provided 3-tuples of layers corresponding to short-wave and long-wave
    radiative parameters, calculate the temperature of each layer under
    insolation defined by Q.

    :@param lw_layers: 3-tuple of Layer objects representing upper, lower,
        and surface layers' longwave radiative properties.
    :@param sw_layers: 3-tuple of Layer objects representing upper, lower,
        and surface layers' shortwave radiative properties.
    """
    assert all(type(sw) is Layer for sw in sw_layers)
    assert all(type(lw) is Layer for lw in lw_layers)
    assert all(sw.validate() for sw in sw_layers)
    assert all(lw.validate() for lw in lw_layers)
    assert len(sw_layers) == len(lw_layers) == 3

    sw_transforms = (sw_abs_l1, sw_abs_l2, sw_abs_l3)
    lw_transforms = (lw_emit_l1, lw_emit_l2, lw_emit_l3)
    S = layer_sw_abs(sw_layers, Q)
    M = np.array([f(*lw_layers) for f in lw_transforms])
    return np.matmul(np.linalg.inv(M), S)**(1/4)

if __name__=="__main__":
    lw = (Layer(.096,.899,.005), Layer(.740,.045,.215), Layer(.926,.000,.074))
    sw = (Layer(.020,.942,.038), Layer(.171,.584,.245), Layer(.868,.000,.132))

    sw_abs = layer_sw_abs(sw)
    temps = layer_temps(lw,sw)
    sfc_to_space = emit_sfc(lw, sw)
    atmo_to_space = emit_atmo(lw, sw)

    print(f"Layerwise absorption (W/m^2):  {sw_abs}")
    print(f"Layerwise temps (K):           {temps}")
    print()
    print(f"Absorbed by surface:           {sw_abs[2]:.3f}")
    print(f"Absorbed by atmosphere:        {sum(sw_abs[:2]):.3f}")
    print()
    print(f"Reflected by surface:          {sw_ref_sfc(sw, Q=1):.5f}")
    print(f"Reflected by atmosphere:       {sw_ref_atmo(sw):.3f}")
    print()
    print(f"Net surface heat loss:         {sw_abs[2]:.3f}")
    print()
    print(f"Sfc LW to space (W/m^2):       {sfc_to_space:.3f}")
    print(f"Sfc LW to atmosphere (W/m^2):  {sw_abs[2]-sfc_to_space:.3f}")
    print(f"Atmo LW to space (W/m^2):      {sum(atmo_to_space):.3f}")
    #print(f"Atmo ref power:            {sw_ref_atmo(sw)}")
    # print(f"Sfc ref power:             {sw_ref_sfc(sw)}")
