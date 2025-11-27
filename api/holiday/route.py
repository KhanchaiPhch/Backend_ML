import requests

API_URL = "https://api.iapp.co.th/data/thai-holidays/holidays/range"
API_KEY = "PCx7qg91X4Obd4wEbz2X2loOrIO5Q5st"

def get_thai_holidays(start_date: str, end_date: str, return_count=False):
    params = {
        "apikey": API_KEY,
        "start_date": start_date,
        "end_date": end_date
    }

    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        holidays = [item["date"] for item in data.get("holidays", [])]

        # ถ้าอยากให้คืนจำนวนวัน ก็คืน len
        return len(holidays) if return_count else holidays

    except requests.RequestException as e:
        print("Error calling holidays API:", e)
        
        # คืนค่าเป็น 0 หรือ [] (ขึ้นกับสิ่งที่ต้องการ)
        return 0 if return_count else []


if __name__ == "__main__":
    holidays = get_thai_holidays("2025-12-05","2025-12-31")
    print("จำนวนวันหยุด:", len(holidays))
    print(holidays)
