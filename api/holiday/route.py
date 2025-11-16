import requests

API_URL = "https://api.iapp.co.th/data/thai-holidays/holidays/range"
API_KEY = "PCx7qg91X4Obd4wEbz2X2loOrIO5Q5st"

def get_thai_holidays(start_date: str, end_date: str):
    params = {
        "apikey": API_KEY,
        "start_date": start_date,
        "end_date": end_date
    }

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        # ได้เฉพาะวันที่
        return [item["date"] for item in data.get("holidays", [])]

    except requests.RequestException as e:
        print("Error calling holidays API:", e)
        return []

if __name__ == "__main__":
    holidays = get_thai_holidays("2025-12-05","2025-12-31")
    print("จำนวนวันหยุด:", len(holidays))
    print(holidays)
