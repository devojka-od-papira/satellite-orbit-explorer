"""Each figure function must write a non-empty PNG and close its figure."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from orbit_explorer.plots import (
    plot_geo_vs_igso,
    plot_ground_tracks,
    plot_speeds,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("plot", [plot_ground_tracks, plot_speeds, plot_geo_vs_igso])
def test_writes_png_and_closes_figure(plot, tmp_path):
    plt.close("all")
    path = tmp_path / "figure.png"
    plot(path)
    assert path.read_bytes().startswith(PNG_SIGNATURE)
    assert plt.get_fignums() == []
