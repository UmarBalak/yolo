def hit_weather():
    import http.client

    conn = http.client.HTTPSConnection("the-weather-api.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "42047b4750msh234436c97fc8894p1a2e84jsn3df40fb61052",
        'x-rapidapi-host': "the-weather-api.p.rapidapi.com"
    }

    conn.request("GET", "/api/weather/mumbai", headers=headers)

    res = conn.getresponse()
    data = res.read()

    return data.decode("utf-8")

# print(hit_weather())