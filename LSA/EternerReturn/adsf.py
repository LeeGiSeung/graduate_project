import requests

url = "https://open-api.bser.io/v1/data/hash"

headers = {
    "accept": "application/json",
    "x-api-key": "bp7bOUefZK5EPnH3bvNq366a5SSjf3QB9nFhbHQx"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    # 응답 데이터를 처리하는 코드를 추가하세요
    print(data)
else:
    print(f"API 요청 실패. 응답 코드: {response.status_code}")
    print(response.text)
