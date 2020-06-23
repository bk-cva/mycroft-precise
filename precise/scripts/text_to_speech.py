import base64
import requests
import json


headers = {
    'x-origin': 'https://explorer.apis.google.com',
    'content-type': 'application/json',
    'Content-Type': 'text/plain',
}

params = (
    ('key', 'AIzaSyAa8yy0GdcGPHdtD083HiGGx_S0vMPScDM'),
    ('alt', 'json'),
)

data_request = {
    "input": {
        "text": "Chào bạn"
    },
    "voice": {
        "languageCode": "vi-VN",
        "name": "vi-VN-Wavenet-A"
    },
    "audioConfig": {
        "audioEncoding": "LINEAR16",
        "pitch": 1,
        "speakingRate": 1,
        "sampleRateHertz": 16000
    }
}


def text_to_bytes(text: str):
    data_request['input']['text'] = text
    response = requests.post('https://texttospeech.googleapis.com/v1/text:synthesize',
                             headers=headers, params=params, json=data_request)
    data = json.loads(response.text)
    base64_message = data['audioContent']
    base64_bytes = base64_message.encode('ascii')
    message_bytes = base64.b64decode(base64_bytes)

    return message_bytes
