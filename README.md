# Object Detection with YOLO

This repository contains code and resources for object detection using the YOLO (You Only Look Once) algorithm.

## Directory Structure

```plaintext
Object-Detection-with-YOLO/
├── README.md
├── data/
│   └── your_dataset/
│       ├── images/
│       └── annotations/
├── models/
│   └── yolov3.weights
├── notebooks/
│   └── training_notebook.ipynb
├── src/
│   ├── train.py
│   ├── detect.py
│   ├── utils.py
│   └── config.py
└── requirements.txt
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/aeshaharshad/Object-Detection-with-YOLO.git
   cd Object-Detection-with-YOLO
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the model:
```bash
python src/train.py --data data/your_dataset
```

To run detection:
```bash
python src/detect.py --weights models/yolov3.weights --source data/images
```

## Contributing

Feel free to submit a pull request if you have improvements or bug fixes!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
