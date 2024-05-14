# Ortho AI Backend

This repository contains the Flask app that hosts the pretrained model for the Ortho project.

## Requirements
- [Python 3](https://www.python.org/downloads/)
- pip (usually included with Python 3)
- PyTorch

**NOTE** If you are willing to start locally it's not recommended to use `pip install pytorch`, isntead install it from [Official PyTorch Site](https://pytorch.org/get-started/locally/).

## Installation

To run the app, make sure you have the following packages installed:

- cv2
- torch
- flask
- werkzeug.utils
- facenet_pytorch
- PIL
- numpy
- torchvision

You can install these packages using pip:
```bash
pip install cv2 flask werkzeug.utils facenet-pytorch Pillow numpy torchvision
```

## Running the App
1. Open a terminal in the project directory (where this README is located).
2. Start the Flask development server:
```bash
python app.py
```
3. Access the app
    - Local development: http://127.0.0.1:5000/ (default port)
    - Custom port (optional):
    ```bash
    # In ortho-ai-backend.py, modify the `app.run()` line:
    app.run(debug=True, host='0.0.0.0', port=8080)  # Replace 8080 with your desired port
    ```

## Usage
- Firstly: you need to validate that everthing is by scanning test image
    ```bash
    curl -X POST -F "file=@./test.jpg" http://localhost:5000/scan
    ```
    you well receive a response like the following, anything else means there is an issue with your installation
    ```json
    {
        "image": base64 image,
        "predictions": [
            {
            "classes": [
                7.06945575075224e-05,
                6.261473572521936e-06,
                0.9999229907989502
            ],
            "description": "the smile is considered gummy and may require lip repositioning surgery to reduce the amount of gum tissue that is exposed when smiling.",
            "labels": [
                "Low",
                "Medium",
                "High"
            ],
            "predictedIndex": 2,
            "title": "Lipline"
            },
            ...
        ]
    }
    ```
- Visit [Ortho Backend](repository) repository for the full usage

## Additional Notes
- Ensure the model.pt file is placed in the project directory or a known location accessible by the app.
- Refer to the documentation of cv2, torch, Flask, facenet-pytorch, and other libraries for detailed usage information.
- For production deployment, explore options beyond the built-in Flask development server (e.g., Gunicorn, uWSGI).