# API Documentation for Fake News Detection Model

## Overview
This API allows users to submit news articles and receive predictions on whether the articles are classified as "Fake News" or "True News".

## Endpoints

### 1. Predict News
- **URL**: `/`
- **Method**: `POST`
- **Request Body**:
  - `news`: (string) The news article to be classified.
  
- **Response**:
  - `prediction`: (string) The classification result ("Fake News" or "True News").
  - `news_article`: (string) The original news article submitted.

### Example Request
```json
{
  "news": "This is a sample news article."
}
```

### Example Response
```json
{
  "prediction": "True News",
  "news_article": "This is a sample news article."
}
```

## Error Handling
- If the input is empty, the API will return an error message indicating that the input cannot be empty.
- If the input is not a valid string, a `ValueError` will be raised.

## Unit Tests
Unit tests should be written to ensure the functionality of the model and the API endpoints. Tests should cover:
- Input validation
- Prediction accuracy
- Error handling

## Conclusion
This API provides a simple interface for detecting fake news using a trained machine learning model. Ensure to validate inputs and handle errors gracefully.
