import requests

base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0."

for year in range(2019, 2020):
  url = f"{base_url}{year}.days_p05.nc"
  response = requests.get(url)

  if response.status_code == 200:
    with open(f"chirps-{year}.days_p05.nc", "wb") as file:
      file.write(response.content)
      print(f"Downloaded chirps-{year}.days_p05.nc")
  else:
    print(f"Error downloading {url} (status code: {response.status_code})")

print("Download complete!")