# CANai: Capture, Analyze, Notify with AI
### An AI-powered object detection system that streams video, detects objects, and records events.

---

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/your-username/canai.git
cd canai
```

### 2. Create a Virtual Environment (Recommended)

```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install poetry
poetry install
```

### 4. Add Model Weights

Create a directory named `models`, add model, and rename model in `configs/app_config.yaml`.

## Usage

### Running the Application

#### **1. Using a RealSense Camera**
```sh
python main.py --camera realsense
```

#### **2. Using a Webcam**
```sh
python main.py --camera webcam
```

Press **`q`** to stop the application.

---