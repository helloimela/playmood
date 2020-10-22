import unittest
from weather import Weather

class WeatherTest(unittest.TestCase):
    def test_fetch_data(self):
        urlYr = 'http://www.yr.no/place/Sweden/Stockholm/Stockholm/varsel_time_for_time.xml'
        curr_weather = Weather(0);
        curr_weather.fetch_data_from_url(urlYr);
        curr_weather.print()