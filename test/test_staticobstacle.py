import numpy as np

from murseco.obstacle.static import StaticObstacle
import murseco.utility.io


def test_staticobstacle_pdf():
    borders = (-2, 2, -1, 1)
    x, y = np.meshgrid(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7))
    pdf = StaticObstacle(borders).pdf.pdf_at(x, y)
    mask_expected = np.logical_and(np.logical_and(-2 <= x, x <= 2), np.logical_and(-1 <= y, y <= 1))
    pdf_expected = np.asarray(np.ones_like(x) * mask_expected, dtype=float)
    assert np.array_equal(pdf, pdf_expected)


def test_stats_gmm_obstacle_json():
    staticobstacle_1 = StaticObstacle((-5.23, 5, -2, 8.51))
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/staticobstacle_test.json")
    staticobstacle_1.to_json(cache_path)
    staticobstacle_2 = StaticObstacle.from_json(cache_path)
    assert staticobstacle_1.summary() == staticobstacle_2.summary()
    assert staticobstacle_1.pdf.pdf_at(0.0, 0.0) == staticobstacle_2.pdf.pdf_at(0.0, 0.0)
