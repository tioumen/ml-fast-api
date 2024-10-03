from fastapi.testclient import TestClient
from main import app
import json

# Initialize a FastAPI client
client = TestClient(app)

# Test your root endpoint
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello stranger! This API allow you to evaluate the quality of red wine. Go to the /docs for more details."}

# Test your predict endpoint
def test_predict():
    response = client.post(
        "/predict",
        headers={'accept': 'application/json', 'Content-Type': 'application/json'},
        json={'alcohol':9.4, 'volatile_acidity': 0.7},
    )
    assert response.status_code == 200
    assert response.json() == json.dumps({
        'prediction': 0,
        'probability': [0.7759315070304591, 0.22406849296954087]
    })