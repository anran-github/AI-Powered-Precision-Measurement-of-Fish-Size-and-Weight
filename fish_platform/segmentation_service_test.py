import requests

# Replace with your Flask server URL and endpoint
API_URL = "http://127.0.0.1:5000/segment"

# Path to the image file you want to test
IMAGE_PATH = "/media/anranli/DATA/data/fish/Final G2 12_11_24/2.JPG"

def test_api(image_path):
    with open(image_path, 'rb') as img_file:
        # Send a POST request to the API with the image file
        response = requests.post(
            API_URL,
            files={'image': img_file}
        )
    
    # Check the response
    if response.status_code == 200:
        print("API Response:")
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_api(IMAGE_PATH)
