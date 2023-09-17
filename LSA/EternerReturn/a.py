import requests

# API 엔드포인트 URL 설정
endpoint_url = 'https://open-api.bser.io/v1/user/nickname?query=[매화검존이기승]'

# 요청 헤더 설정
headers = {
    'accept': 'application/json',
    'x-api-key': 'bp7bOUefZK5EPnH3bvNq366a5SSjf3QB9nFhbHQx'
}

params = {
    'Nickname' : '매화검존이기승'
}

# OPTIONS 메서드를 사용하여 API 엔드포인트에 대한 정보를 요청
response = requests.options(endpoint_url, headers=headers, params=params)

# 응답 확인
if response.status_code == 200:
    # API 엔드포인트에 대한 정보를 출력
    print(response.text)
else:
    print(f"API 요청 실패. 응답 코드: {response.status_code}")
    print(response.text)
