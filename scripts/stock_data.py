import http.client

conn = http.client.HTTPSConnection("twelve-data1.p.rapidapi.com")

headers = {
    'X-RapidAPI-Key': "ead4f2f77amsh30ac4b37d5bdb2dp19f535jsn4dd2ffd76c28",
    'X-RapidAPI-Host': "twelve-data1.p.rapidapi.com"
    }

conn.request("GET", "/time_series?interval=1h&symbol=AAPL&format=csv&outputsize=5000", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))