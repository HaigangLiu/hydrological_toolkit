import pandas as pd
import pytest

from ..hydrological_toolbox.data import util

test_phrases = [('gage', ), ('calcium', ), ('discharge', )]
test_site_numbers = [('02110400', 'BUCK CREEK NEAR LONGS, SC'),
                     ('02110500', 'WACCAMAW RIVER NEAR LONGS, SC'),
                     ('02110550', 'WACCAMAW RIVER ABOVE CONWAY, SC')]
test_addresses = [('1523 Greene St, Columbia, SC, 29201', ),
                  ('800 State St, West Columbia, SC, 29169',)]
state_abbreviation_and_fullname = [('SC', 'South Carolina'),
                                   ('NY', 'New York'),
                                   ('NC', 'North Carolina'),
                                   ('MA', 'Massachusetts'),
                                   ('PR', 'Puerto Rico')]


@pytest.mark.parametrize('phrase', test_phrases)
def test_find_hydrological_variable_by_keyword(phrase):
    (phrase, ) = phrase
    df = util.find_hydrological_variable_by_keyword(phrase)
    assert len(df) >= 1
    assert 'GROUP' in df and 'DESCRIPTION' in df and 'CASRN' in df
    assert 'SRSNAME' in df and 'UNIT' in df
    for idx, row in df.iterrows():
        assert phrase in row['DESCRIPTION'].lower()


def test_get_x_y_projections():
    columbia_sc = [34.0007, -81.0348]
    charleston_sc = [32.7765, -79.9311]
    greenville_sc = [34.8526, -82.3940]

    locations_test = pd.DataFrame([columbia_sc, charleston_sc, greenville_sc],
                                  columns=['lat', 'lon'])
    x_y = util.get_x_y_projections(locations_test)
    assert len(x_y) == 3
    assert isinstance(x_y, pd.DataFrame)
    assert 'x' in x_y and 'y' in x_y and 'lat' in x_y and 'lon' in x_y

    # check validity of the values
    assert x_y.lat.min() >= 32  # should not be too far away
    assert x_y.lon.min() >= -83  # should not be too far away

    assert x_y.lat.max() <= 35  # should not be too far away
    assert x_y.lon.max() <= -79  # should not be too far away


@pytest.mark.parametrize('site_numbers', test_site_numbers)
def test_get_site_info(site_numbers):
    site_code, site_description = site_numbers
    result_per_site = util.get_site_info(site_code)
    assert isinstance(result_per_site, dict)

    content = result_per_site[site_code]
    assert isinstance(content, tuple)
    # verify the locations have the correct name
    assert content[0] == site_description


@pytest.mark.parametrize('address', test_addresses)
def test_get_lat_lon_from_address(address):
    (address, ) = address
    latitude, longitude = util.get_lat_lon_from_address(address)

    # verify the validity from the service
    # both places are in South Carolina
    assert 30 < latitude < 35
    assert -83 < longitude < -78


@pytest.mark.parametrize('names', state_abbreviation_and_fullname)
def test_get_state_name_from_abbreviation(names):
    abbreviation, fullname = names
    assert util.get_state_name_from_abbreviation(abbreviation) == fullname
    with pytest.raises(ValueError):
        util.get_state_name_from_abbreviation('South Carolina')
    with pytest.raises(KeyError):
        util.get_state_name_from_abbreviation('JK')











