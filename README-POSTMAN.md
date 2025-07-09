# Testing the Cybersecurity Threat Detection API with Postman

This guide will help you test the API endpoints using Postman, a popular API testing tool.

## Prerequisites

1. Ensure the Flask application is running (`cd frontend && python app.py`)
2. Install [Postman](https://www.postman.com/downloads/)

## Available Endpoints

The application exposes the following API endpoints for testing:

### 1. Single Prediction

- **Endpoint**: `http://localhost:5000/api/predict`
- **Method**: POST
- **Content-Type**: application/json

#### Sample Request Body (Normal Traffic):
```json
{
  "Flow Duration": 5220,
  "Flow Bytes/s": 4096.7,
  "Flow Packets/s": 12.3,
  "Fwd Packet Length Max": 1024,
  "Bwd Packet Length Min": 64,
  "Flow IAT Mean": 125.6,
  "Fwd IAT Mean": 130.2,
  "Bwd IAT Mean": 121.8,
  "Fwd PSH Flags": 0,
  "Bwd PSH Flags": 0,
  "Fwd URG Flags": 0,
  "Bwd URG Flags": 0,
  "Fwd Header Length": 160,
  "Bwd Header Length": 160,
  "Down/Up Ratio": 1.2,
  "Packet Length Std": 245.3,
  "Packet Length Variance": 60158.09,
  "Destination Port": 443
}
```

#### Sample Request Body (Attack Traffic):
```json
{
  "Flow Duration": 958,
  "Flow Bytes/s": 35621.4,
  "Flow Packets/s": 983.7,
  "Fwd Packet Length Max": 1420,
  "Bwd Packet Length Min": 0,
  "Flow IAT Mean": 12.8,
  "Fwd IAT Mean": 13.5,
  "Bwd IAT Mean": 0,
  "Fwd PSH Flags": 1,
  "Bwd PSH Flags": 0,
  "Fwd URG Flags": 1,
  "Bwd URG Flags": 0,
  "Fwd Header Length": 320,
  "Bwd Header Length": 0,
  "Down/Up Ratio": 0.01,
  "Packet Length Std": 12.5,
  "Packet Length Variance": 156.25,
  "Destination Port": 22
}
```

### 2. Batch Prediction

- **Endpoint**: `http://localhost:5000/api/batch-predict`
- **Method**: POST
- **Content-Type**: application/json

#### Sample Request Body:
```json
[
  {
    "Flow Duration": 5220,
    "Flow Bytes/s": 4096.7,
    "Flow Packets/s": 12.3,
    "Fwd Packet Length Max": 1024,
    "Bwd Packet Length Min": 64,
    "Fwd PSH Flags": 0,
    "Fwd URG Flags": 0,
    "Down/Up Ratio": 1.2,
    "Destination Port": 443
  },
  {
    "Flow Duration": 958,
    "Flow Bytes/s": 35621.4,
    "Flow Packets/s": 983.7,
    "Fwd Packet Length Max": 1420,
    "Bwd Packet Length Min": 0,
    "Fwd PSH Flags": 1,
    "Fwd URG Flags": 1,
    "Down/Up Ratio": 0.01,
    "Destination Port": 22
  }
]
```

### 3. API Health Check

- **Endpoint**: `http://localhost:5000/api/health`
- **Method**: GET

### 4. Test Prediction (Simplified)

- **Endpoint**: `http://localhost:5000/api/test-predict`
- **Method**: POST
- **Content-Type**: application/json

This endpoint is specifically designed for Postman testing and returns more simplified results than the main prediction endpoint.

## Setting Up Postman

1. **Create a new Collection**:
   - Click "New" > "Collection"
   - Name it "Cybersecurity Threat Detection API"

2. **Create requests for each endpoint**:
   - Right-click on the collection > "Add Request"
   - Configure each request with the appropriate HTTP method and URL
   - Add the sample JSON data to the request body for POST requests

3. **Set Headers**:
   - For POST requests, set the Content-Type header to "application/json"

## Testing Tips

1. **Variables**: Create environment variables in Postman for the base URL to easily switch between development and production environments.

2. **Automation**: Use Postman's Collection Runner to run multiple tests in sequence.

3. **Testing Different Attack Scenarios**: Modify the following fields to simulate different attack types:
   - Increase `Flow Packets/s` above 500 to simulate DoS attacks
   - Set both `Fwd PSH Flags` and `Fwd URG Flags` to 1 to simulate TCP flag manipulation
   - Set `Destination Port` to common vulnerable ports (22, 23, 3389) to simulate targeting
   - Decrease `Flow Duration` while increasing `Flow Bytes/s` to simulate data exfiltration

4. **Response Analysis**: Pay attention to:
   - The `prediction` field ("Normal" or "Attack")
   - The `prediction_probability` value (higher means more confident)
   - For attack detections, look at the `indicators` field for what triggered the detection

## Postman Collection JSON

You can also import this prepared Postman collection:

```json
{
  "info": {
    "name": "Cybersecurity Threat Detection API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Single Prediction (Normal)",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "url": "http://localhost:5000/api/predict",
        "body": {
          "mode": "raw",
          "raw": "{\n  \"Flow Duration\": 5220,\n  \"Flow Bytes/s\": 4096.7,\n  \"Flow Packets/s\": 12.3,\n  \"Fwd Packet Length Max\": 1024,\n  \"Bwd Packet Length Min\": 64,\n  \"Flow IAT Mean\": 125.6,\n  \"Fwd IAT Mean\": 130.2,\n  \"Bwd IAT Mean\": 121.8,\n  \"Fwd PSH Flags\": 0,\n  \"Bwd PSH Flags\": 0,\n  \"Fwd URG Flags\": 0,\n  \"Bwd URG Flags\": 0,\n  \"Fwd Header Length\": 160,\n  \"Bwd Header Length\": 160,\n  \"Down/Up Ratio\": 1.2,\n  \"Packet Length Std\": 245.3,\n  \"Packet Length Variance\": 60158.09,\n  \"Destination Port\": 443\n}"
        }
      }
    },
    {
      "name": "Single Prediction (Attack)",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "url": "http://localhost:5000/api/predict",
        "body": {
          "mode": "raw",
          "raw": "{\n  \"Flow Duration\": 958,\n  \"Flow Bytes/s\": 35621.4,\n  \"Flow Packets/s\": 983.7,\n  \"Fwd Packet Length Max\": 1420,\n  \"Bwd Packet Length Min\": 0,\n  \"Flow IAT Mean\": 12.8,\n  \"Fwd IAT Mean\": 13.5,\n  \"Bwd IAT Mean\": 0,\n  \"Fwd PSH Flags\": 1,\n  \"Bwd PSH Flags\": 0,\n  \"Fwd URG Flags\": 1,\n  \"Bwd URG Flags\": 0,\n  \"Fwd Header Length\": 320,\n  \"Bwd Header Length\": 0,\n  \"Down/Up Ratio\": 0.01,\n  \"Packet Length Std\": 12.5,\n  \"Packet Length Variance\": 156.25,\n  \"Destination Port\": 22\n}"
        }
      }
    },
    {
      "name": "Batch Prediction",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "url": "http://localhost:5000/api/batch-predict",
        "body": {
          "mode": "raw",
          "raw": "[\n  {\n    \"Flow Duration\": 5220,\n    \"Flow Bytes/s\": 4096.7,\n    \"Flow Packets/s\": 12.3,\n    \"Fwd Packet Length Max\": 1024,\n    \"Bwd Packet Length Min\": 64,\n    \"Fwd PSH Flags\": 0,\n    \"Fwd URG Flags\": 0,\n    \"Down/Up Ratio\": 1.2,\n    \"Destination Port\": 443\n  },\n  {\n    \"Flow Duration\": 958,\n    \"Flow Bytes/s\": 35621.4,\n    \"Flow Packets/s\": 983.7,\n    \"Fwd Packet Length Max\": 1420,\n    \"Bwd Packet Length Min\": 0,\n    \"Fwd PSH Flags\": 1,\n    \"Fwd URG Flags\": 1,\n    \"Down/Up Ratio\": 0.01,\n    \"Destination Port\": 22\n  }\n]"
        }
      }
    },
    {
      "name": "API Health",
      "request": {
        "method": "GET",
        "url": "http://localhost:5000/api/health"
      }
    },
    {
      "name": "Test Prediction",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "url": "http://localhost:5000/api/test-predict",
        "body": {
          "mode": "raw",
          "raw": "{\n  \"Flow Duration\": 958,\n  \"Flow Bytes/s\": 35621.4,\n  \"Flow Packets/s\": 983.7,\n  \"Fwd PSH Flags\": 1,\n  \"Fwd URG Flags\": 1,\n  \"Destination Port\": 22\n}"
        }
      }
    }
  ]
}
```

Save this JSON as `cybersecurity-api.postman_collection.json` and import it into Postman. 