import requests

class AMap:
    def __init__(self, key):
        self.key = key
        self.session = requests.Session()

    def geocode(self, address, city=None):
        url = 'http://restapi.amap.com/v3/geocode/geo'
        params = {
            'address': address,
            'key': self.key,
        }
        if city:
            params['city'] = city
        req = self.session.get(url, params=params)
        try:
            resp = req.json()
        except Exception as e:
            raise
        if resp['status']:
            return resp['geocodes'][0]
        else:
            raise RuntimeError(resp['info'])