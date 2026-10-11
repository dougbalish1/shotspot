import pytest

from shotspot.core.presets import PRESETS


@pytest.fixture(params=list(PRESETS))
def preset(request):
    """Every preset scenario, at zero noise."""
    return PRESETS[request.param]()
