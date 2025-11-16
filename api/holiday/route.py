import requests

API_URL = "https://api.iapp.co.th/data/thai-holidays/holidays"
API_KEY = "PCx7qg91X4Obd4wEbz2X2loOrIO5Q5st"

def get_thai_holidays(days_after=3):
    """
    ดึงรายการวันหยุดไทยจาก API iApp
    :param days_after: จำนวนวันหลังจากวันนี้ที่ต้องการดึงข้อมูล
    :return: dict {date_string: holiday_name} เช่น {"2025-01-01": "วันปีใหม่"}
    """
    params = {
        "apikey": API_KEY,
        "days_after": days_after
    }

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        # สมมติ data['holidays'] เป็น list ของ dict ที่มี 'date' และ 'name'
        holidays = {item["date"]: item["name"] for item in data.get("holidays", [])}
        return holidays
    except requests.RequestException as e:
        print("Error while calling API:", e)
        return {}

if __name__ == "__main__":
    holidays = get_thai_holidays()
    print("จำนวนวันหยุด:", len(holidays))
    print(holidays)
