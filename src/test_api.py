import requests, os

os.environ["USER_AGENT"] = "OSRS-Flipper/1.0 (contact: you@example.com)"

response = requests.get(
    "https://prices.runescape.wiki/api/v1/osrs/latest",
    headers={"User-Agent": os.environ["USER_AGENT"]}
)

print("Status code:", response.status_code)
print("Keys returned:", list(response.json().keys()))
