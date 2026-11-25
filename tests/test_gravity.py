"""The EGM2008 gravity field: coefficients, the recursions, sizes at GNSS radii."""

import numpy as np
import pytest

from orbit_explorer import ephemeris, gravity, j2

GALILEO_KM = 29600.0


@pytest.fixture(scope="module")
def field():
    return gravity.load_field(degree=8)


def sphere(radius_km: float, step_deg: float = 5.0) -> np.ndarray:
    lat, lon = np.meshgrid(
        np.radians(np.arange(-85, 90, step_deg)),
        np.radians(np.arange(0, 360, step_deg)),
    )
    xyz = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )
    return radius_km * xyz.reshape(-1, 3)


def only(field, keep) -> gravity.GravityField:
    """The field with only the coefficients (n, m) for which keep(n, m)."""
    C, S = field.C.copy(), field.S.copy()
    for n in range(field.degree + 1):
        for m in range(n + 1):
            if not keep(n, m):
                C[n, m] = S[n, m] = 0.0
    return gravity.GravityField("part", field.gm, field.radius_km, field.degree, C, S)


def test_the_stored_model(field):
    assert field.name == "EGM2008"
    assert field.gm == pytest.approx(398600.4415)
    assert field.radius_km == pytest.approx(6378.1363)
    # J2 = -C20 (normalised -4.8417e-4 times sqrt(5))
    assert -field.C[2, 0] == pytest.approx(1.0826e-3, rel=1e-4)
    assert field.C[0, 0] == 1.0


def test_more_than_the_file_has_is_refused():
    with pytest.raises(ValueError, match="degree 13 asked"):
        gravity.load_field(degree=13)


def test_a_subset_loads_as_the_same_field(tmp_path):
    path = tmp_path / "egm_n4.gfc"
    gravity.main([str(gravity.EGM2008), "--max-degree", "4", "-o", str(path)])
    small, full = gravity.load_field(path, 4), gravity.load_field(degree=4)
    np.testing.assert_array_equal(small.C, full.C)
    np.testing.assert_array_equal(small.S, full.S)
    assert "max_degree                  4" in path.read_text()


def test_point_mass_and_j2_match_the_j2_module(field):
    r = np.array([[7000.0, 1200.0, -3000.0], [20000.0, -15000.0, 18000.0]])
    earth = j2.Earth("EGM2008", field.gm, field.radius_km, -field.C[2, 0])
    zonal_j2 = only(field, lambda n, m: n == 0 or (n, m) == (2, 0))
    np.testing.assert_allclose(
        gravity.acceleration(r, zonal_j2), j2.acceleration(r, True, earth), rtol=1e-13
    )
    point = only(field, lambda n, m: n == 0)
    np.testing.assert_allclose(
        gravity.acceleration(r, point), j2.acceleration(r, False, earth), rtol=1e-13
    )


@pytest.mark.parametrize(
    "r", [[20000.0, -15000.0, 18000.0], [7000.0, 1200.0, -3000.0], [10.0, 20.0, 7100.0]]
)
def test_acceleration_is_the_gradient_of_the_potential(field, r):
    """The recursions against the series summed directly (near a pole too)."""
    r = np.array(r)
    h = 1e-3
    gradient = [
        (gravity.potential(r + h * e, field) - gravity.potential(r - h * e, field))
        / (2 * h)
        for e in np.eye(3)
    ]
    a = gravity.acceleration(r, field)[0]
    assert np.abs(gradient - a).max() < 1e-8 * np.linalg.norm(a)


def test_beyond_j2_the_field_outweighs_sunlight_at_galileo(field):
    r = sphere(GALILEO_KM)
    rest = gravity.acceleration(
        r, only(field, lambda n, m: (n, m) not in ((0, 0), (2, 0)))
    )
    size = np.linalg.norm(rest, axis=1).max()
    assert 3.5e-10 < size < 5e-10
    srp = ephemeris.srp_acceleration_km_s2(0.02)
    j3 = ephemeris.zonal_acceleration_km_s2(3, ephemeris.J3, GALILEO_KM)
    assert size > 3 * srp and size > 5 * j3


def test_degrees_above_8_add_centimetres_a_day():
    r = sphere(GALILEO_KM)
    a8 = gravity.acceleration(r, gravity.load_field(degree=8))
    a12 = gravity.acceleration(r, gravity.load_field(degree=12))
    assert ephemeris.drift_km(np.linalg.norm(a12 - a8, axis=1).max()) < 1e-4  # 10 cm
