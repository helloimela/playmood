import urllib
import xmltodict  # pip install xmltodict

class Weather:
    def __init__ (self, age_hour):
        self.age_hour = age_hour
        self.condition = -99 # 1: clear sky, 2: fair, etc
        self.temp = -99
        self.precipitation = -99
        self.windspeed = -99

    def fetch_data_from_url(self, url):
        file = urllib.request.urlopen(url)
        tempfile = file.read()
        file.close()
        dict = xmltodict.parse(tempfile)
        data = dict['weatherdata']['forecast']['tabular']['time'][self.age_hour]
        self.condition = int(data['symbol']['@number'])
        self.temp = int(data['temperature']['@value'])
        self.precipitation = float(data['precipitation']['@value'])
        self.windspeed = float(data['windSpeed']['@mps'])

    def print(self):
        print("Age in hour:%d condition:%d temp:%dC precipitation:%.1f windspeed:%.1f mps" % \
              (self.age_hour, self.condition, self.temp, self.precipitation, self.windspeed))
