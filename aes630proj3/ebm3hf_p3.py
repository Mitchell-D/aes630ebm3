"""
:module ebm3: an energy-based 3-layer radiative transfer model based on a
    transformation matrix relating layerwise temperature to layerwise
    absorption of solar insolation. The matrix parameters are solved for
    exactly using constants derived from the radiative transfer equation.
"""
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from krttdkit.visualize import TextFormat as TF
from krttdkit.visualize import geoplot as gp

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
        if not abs(sum(components)-1) <= tolerance:
            raise ValueError(f"Layer components {components} must sum to 1")
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

    #V1 = lw1.a * sigma * (c1-c2*c3-c2*c7*c8)
    V1 = lw1.a * sigma * (c1 - c2 * (c3+c7*c8))
    #V2 = lw2.a * sigma * (-c2-c2*c3*c4-c2*c3*c6-c2*c7*c9-c2*c7*c10)
    V2 = lw2.a * sigma * -c2*(1 + c3*(c4+c6) + c7*(c9+c10))
    #V3 = lw3.a * sigma * (-c2*c3*c5-c2*c7)
    V3 = lw3.a * sigma * -c2*(c3*c5+c7)
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
             heat_pct=0, skin_temp=None, Q=341.0):
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
        skin_temp = layer_temps(lw_layers,sw_layers, Q=Q)[2]
    lw1, lw2, lw3 = tuple(lw_layers)
    c1 = lw1.t*lw2.t*lw3.a*sigma*skin_temp**4
    c2 = (1-lw1.r*lw2.r)*(1-lw2.r*lw3.r) - lw2.t**2*lw3.r*lw1.r
    return c1/c2

def emit_atmo(lw_layers:tuple, sw_layers:tuple, heat_pct=0, Q=341.0):
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
    temps = layer_temps(lw_layers, sw_layers, Q=Q)
    c1 = 1 + lw1.t*lw2.r*(1-lw2.r*lw3.r) + lw2.t**2*lw3.r
    c2 = lw1.t*(1+lw1.r*lw2.r*(1-lw2.r*lw3.r)+lw2.t*lw3.r*(1-lw1.r*lw2.t))
    return np.array([sigma * lw1.a * temps[0]**4 * c1,
                     sigma * lw2.a * temps[1]**4 * c2,
                     0])

def layer_sw_abs(sw_layers:tuple, heat_pct=0, Q=341.0):
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
    S = np.array([f(*sw_layers, Q) for f in sw_transforms])
    ## Add heat fluxes
    return S + np.array([0, heat_pct*Q, -1*heat_pct*Q])


def layer_temps(lw_layers:tuple, sw_layers:tuple, heat_pct=0, Q=341.0):
    """
    Provided 3-tuples of layers corresponding to short-wave and long-wave
    radiative parameters, calculate the temperature of each layer under
    insolation defined by Q.

    :@param lw_layers: 3-tuple of Layer objects representing upper, lower,
        and surface layers' longwave radiative properties.
    :@param sw_layers: 3-tuple of Layer objects representing upper, lower,
        and surface layers' shortwave radiative properties.
    :@param heat_pct: Percentage of incident shortwave irradiance which is
        ultimately incorporated into vertical heat flux
    """
    assert all(type(sw) is Layer for sw in sw_layers)
    assert all(type(lw) is Layer for lw in lw_layers)
    assert all(sw.validate() for sw in sw_layers)
    assert all(lw.validate() for lw in lw_layers)
    assert len(sw_layers) == len(lw_layers) == 3

    sw_transforms = (sw_abs_l1, sw_abs_l2, sw_abs_l3)
    lw_transforms = (lw_emit_l1, lw_emit_l2, lw_emit_l3)
    S = layer_sw_abs(sw_layers, heat_pct=heat_pct, Q=Q)
    M = np.array([f(*lw_layers) for f in lw_transforms])
    return np.matmul(np.linalg.inv(M), S)**(1/4)

def q1a(sw_layers, lw_layers, heat_pct=0, Q=341.0):
    """
    Temperatures if L2 solar reflectivity increases as solar abs decreases
    """
    d_abs, n_abs = .002, 18
    new_l2s = [Layer(.171-i*d_abs, .584, .245+i*d_abs) for i in range(n_abs)]
    temps = [
            layer_temps(
                lw_layers=lw_layers,
                sw_layers=(sw_layers[0], L, sw_layers[2]),
                heat_pct=heat_pct,
                Q=Q
                ) for L in new_l2s
            ]
    temp_lines = list(map(np.asarray, zip(*temps)))
    temp_diffs = [(T[-1]-T[0])/(d_abs*n_abs) for T in temp_lines]
    slope_str = f"$T_s = {temp_diffs[2]:.3f}\\rho_2 + {new_l2s[0].r}$"
    #print([(L[0], L[-1]) for L in temp_lines])
    gp.plot_lines(
            domain=[L.r for L in new_l2s],
            #ylines=temp_lines,
            #labels=["Layer 1", "Layer 2 (modified)", "Surface"],
            ylines=[temp_lines[2]],
            labels=[f"Surface (L2 modified)\n{slope_str}"],
            plot_spec={
                "title":"Layer 2 SW ref/trans vs EBM3HF Equilibrium Sfc Temp",
                "xlabel":"Layer 2 Shortwave Reflectivity",
                "ylabel":"Equilibrium Temperature (K)",
                "yrange":(284,288),
                "grid":True,
                },
            image_path=Path("p1a.png"),
            show=False,
            )
    print(f"Layerwise temp rates wrt inc. L2 SW ref:",temp_diffs)
    return temp_lines[2]

def q1b(sw_layers, lw_layers, heat_pct=0, Q=341.0):
    """
    Temperatures if L1 solar reflectivity increases as solar trans decreases
    """
    d_ref, n_ref = .002, 18
    new_l1s = [Layer(a=.020, t=.942-i*d_ref, r=.038+i*d_ref)
               for i in range(n_ref)]
    temps = [
            layer_temps(
                lw_layers=lw_layers,
                sw_layers=(L, sw_layers[1], sw_layers[2]),
                heat_pct=heat_pct,
                Q=Q
                ) for L in new_l1s
            ]
    temp_lines = list(map(np.asarray, zip(*temps)))
    temp_diffs = [(T[-1]-T[0])/(d_ref*n_ref) for T in temp_lines]
    slope_str = f"$T_s = {temp_diffs[2]:.3f}\\rho_1 + {new_l1s[0].r}$"
    gp.plot_lines(
            domain=[L.r for L in new_l1s],
            #ylines=temp_lines,
            #labels=["Layer 1 (modified)", "Layer 2", "Surface"],
            ylines=[temp_lines[2]],
            labels=[f"Surface (L1 Modified)\n{slope_str}"],
            plot_spec={
                "title":"Layer 1 SW ref/abs vs EBM3HF Equilibrium Sfc Temp",
                "xlabel":"Layer 1 Shortwave Reflectivity",
                "ylabel":"Equilibrium Temperature (K)",
                "yrange":(284,288),
                "grid":True,
                },
            image_path=Path("p1b.png"),
            show=False,
            )
    print(f"Layerwise temp rates wrt inc. L1 SW ref:",temp_diffs)
    #temp_diffs = np.average(np.vstack(list(map(np.diff, temp_lines))),axis=1)
    #temp_diffs /= d_ref
    return temp_lines[2]

def q2a(sw_layers, lw_layers, heat_pct=0, Q=341.0):
    """
    Temperatures if L2 infrared abs increases as infrared trans decreases
    """
    d_abs, n_abs = .002, 18
    new_l2s = [Layer(.740+i*d_abs, .045-i*d_abs, .215) for i in range(n_abs)]
    temps = [
            layer_temps(
                lw_layers=(lw_layers[0], L, lw_layers[2]),
                sw_layers=lw_layers,
                heat_pct=heat_pct,
                Q=Q
                ) for L in new_l2s
            ]
    temp_lines = list(map(np.asarray, zip(*temps)))
    sfc_warming = temp_lines[2]-temp_lines[2][0]
    temp_diffs = [(T[-1]-T[0])/(d_abs*n_abs) for T in temp_lines]
    slope_str = f"$T_s = {temp_diffs[2]:.3f}a_2 + {new_l2s[0].a}$"
    gp.plot_lines(
            domain=[L.a for L in new_l2s],
            #ylines=temp_lines,
            #labels=["Layer 1 (modified)", "Layer 2", "Surface"],
            ylines=[sfc_warming],
            labels=[f"Surface (L2 Modified)\n{slope_str}"],
            plot_spec={
                "title":"Layer 2 LW abs/trans vs Sfc Warming",
                "xlabel":"Layer 2 Thermal Absorptivity",
                "ylabel":"Increment of Warming (K)",
                "yrange":(-5,5),
                "grid":True,
                },
            image_path=Path("p2a.png"),
            show=False,
            )
    print(f"Layerwise temp rates wrt inc. L2 LW abs:",temp_diffs)
    return temp_lines[2]

def q2b(sw_layers, lw_layers, heat_pct=0, Q=341.0):
    """
    Temperatures if L1 solar reflectivity increases as solar trans decreases
    given increased L2 infrared absorptivity
    """
    d_ref, n_ref = .002, 18
    new_l2 = Layer(a=.770,t=.015,r=.215)
    new_l1s = [Layer(.020, .942-i*d_ref, .038+i*d_ref) for i in range(n_ref)]
    temps = [
            layer_temps(
                lw_layers=(lw_layers[0], new_l2, lw_layers[2]),
                sw_layers=(L, sw_layers[1], sw_layers[2]),
                heat_pct=heat_pct,
                Q=Q
                ) for L in new_l1s
            ]
    temp_lines = list(map(np.asarray, zip(*temps)))
    sfc_warming = temp_lines[2]-temp_lines[2][0]
    temp_diffs = [(T[-1]-T[0])/(d_ref*n_ref) for T in temp_lines]
    slope_str = f"$T_s = {temp_diffs[2]:.3f}\\rho_1 + {new_l1s[0].r}$"
    gp.plot_lines(
            domain=[L.r for L in new_l1s],
            #ylines=temp_lines,
            #labels=["Layer 1 (modified)", "Layer 2", "Surface"],
            ylines=[sfc_warming],
            labels=[f"Surface (L1 Modified)\n{slope_str}"],
            plot_spec={
                "title":"Modified Layer 1 SW ref/trans vs Sfc Warming",
                "xlabel":"Layer 1 Shortwave Reflectivity",
                "ylabel":"Increment of Warming (K)",
                "yrange":(-5,5),
                "grid":True,
                },
            image_path=Path("p2b.png"),
            show=False,
            )
    print(f"Layerwise temp rates wrt inc. L1 SW ref:",temp_diffs)
    return temp_lines[2]

def q3(sw_layers, lw_layers, heat_pct=0, Q=341.0):
    """
    Temperatures if L2 solar reflectivity increases as solar trans decreases
    given increased L2 infrared absorptivity
    """
    d_ref, n_ref = .002, 18
    new_l2_lw = Layer(a=.770,t=.015,r=.215)
    new_l2s = [Layer(.171, .584-i*d_ref, .245+i*d_ref) for i in range(n_ref)]
    temps = [
            layer_temps(
                lw_layers=(lw_layers[0], new_l2_lw, lw_layers[2]),
                sw_layers=(sw_layers[0], L, sw_layers[2]),
                heat_pct=heat_pct,
                Q=Q
                ) for L in new_l2s
            ]
    temp_lines = list(map(np.asarray, zip(*temps)))
    sfc_warming = temp_lines[2]-temp_lines[2][0]
    temp_diffs = [(T[-1]-T[0])/(d_ref*n_ref) for T in temp_lines]
    slope_str = f"$T_s = {temp_diffs[2]:.3f}\\rho_2 + {new_l2s[0].r}$"
    gp.plot_lines(
            domain=[L.r for L in new_l2s],
            #ylines=temp_lines,
            #labels=["Layer 1 (modified)", "Layer 2", "Surface"],
            ylines=[sfc_warming],
            labels=[f"Surface (L2 Modified)\n{slope_str}"],
            plot_spec={
                "title":"Modified Layer 2 SW ref/trans vs Sfc Warming",
                "xlabel":"Layer 2 Shortwave Reflectivity",
                "ylabel":"Increment of Warming (K)",
                "yrange":(-5,5),
                "grid":True,
                },
            image_path=Path("p3.png"),
            show=False,
            )
    print(f"Layerwise temp rates wrt inc. L2 SW ref:",temp_diffs)
    return temp_lines[2]

def q4(sw_layers, lw_layers, heat_pct=0, Q=341.0):
    """
    Temperatures if L3 solar reflectivity increases as solar abs decreases
    given increased L2 infrared absorption
    """
    d_ref, n_ref = .002, 18
    new_l2_lw = Layer(a=.770,t=.015,r=.215)
    new_l3s = [Layer(.868-i*d_ref, .0, .132+i*d_ref) for i in range(n_ref)]
    temps = [
            layer_temps(
                lw_layers=(lw_layers[0], new_l2_lw, lw_layers[2]),
                sw_layers=(sw_layers[0], sw_layers[1], L),
                heat_pct=heat_pct,
                Q=Q
                ) for L in new_l3s
            ]
    temp_lines = list(map(np.asarray, zip(*temps)))
    sfc_warming = temp_lines[2]-temp_lines[2][0]
    temp_diffs = [(T[-1]-T[0])/(d_ref*n_ref) for T in temp_lines]
    slope_str = f"$T_s = {temp_diffs[2]:.3f}\\rho_3 + {new_l3s[0].r}$"
    gp.plot_lines(
            domain=[L.r for L in new_l3s],
            #ylines=temp_lines,
            #labels=["Layer 1 (modified)", "Layer 2", "Surface"],
            ylines=[sfc_warming],
            labels=[f"Surface (L3 Modified)\n{slope_str}"],
            plot_spec={
                "title":"Modified Layer 3 SW ref/abs vs Sfc Warming",
                "xlabel":"Layer 3 Shortwave Reflectivity",
                "ylabel":"Increment of Warming (K)",
                "yrange":(-5,5),
                "grid":True,
                },
            image_path=Path("p4.png"),
            show=False,
            )
    temp_diffs = [(T[-1]-T[0])/(d_ref*n_ref) for T in temp_lines]
    print(f"Layerwise temp rates wrt inc. L3 SW ref:",temp_diffs)
    return temp_lines[2]


if __name__=="__main__":
    heat = .287 ## percentage of incident solar radiance
    ebm3_lw = (Layer(a=.096,t=.899,r=.005),
               Layer(a=.740,t=.045,r=.215),
               Layer(a=.926,t=.000,r=.074))
    ebm3_sw = (Layer(a=.020,t=.942,r=.038),
               Layer(a=.171,t=.584,r=.245),
               Layer(a=.868,t=.000,r=.132))

    print(layer_temps(ebm3_lw, ebm3_sw, heat_pct=heat, Q=341))
    q1a(ebm3_sw, ebm3_lw, heat_pct=heat)
    q1b(ebm3_sw, ebm3_lw, heat_pct=heat)
    q2a(ebm3_sw, ebm3_lw, heat_pct=heat)
    q2b(ebm3_sw, ebm3_lw, heat_pct=heat)
    q3(ebm3_sw, ebm3_lw, heat_pct=heat)
    q4(ebm3_sw, ebm3_lw, heat_pct=heat)
