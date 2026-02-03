from google import genai
from google.genai import types

paths = [
    'face.jpg',
    'glasses1.webp',
    'glasses2.webp',
    'glasses3.webp'
]

PROMPT = '''
These images are a face and three pairs of glasses. Decide which pair of glasses
looks best on the face. Write a response in the format

CHOICE: <The number 0, 1 or 2, indicating which glasses are the best.>

DESCRIPTION: <A text description of why you made that choice.>
'''

image_bytes = []

for path in paths:
    with open(path, 'rb') as f:
        image_bytes.append(f.read())

client = genai.Client()
response = client.models.generate_content(
    model='gemini-3-flash-preview',
    contents=[
        types.Part.from_bytes(
            data=image_bytes[0],
            mime_type='image/jpeg',
        ),
        types.Part.from_bytes(
            data=image_bytes[1],
            mime_type='image/webp',
        ),
        types.Part.from_bytes(
            data=image_bytes[2],
            mime_type='image/webp',
        ),
        types.Part.from_bytes(
            data=image_bytes[3],
            mime_type='image/webp',
        ),
        PROMPT
    ]
)

print(response.text)
