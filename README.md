# UOD Object Detection Pipeline

A modular and scalable pipeline for object detection tasks, leveraging PyTorch and COCO datasets. This project focuses on single-class classification using custom anchors and tailored loss functions.

---

## 🚀 Features

- **Anchor Generation:** Flexible anchor generation with configurable feature map sizes, scales, aspect ratios, and strides.
- **Dataset Integration:** Seamlessly integrates with COCO datasets for training and evaluation.
- **Customizable Loss Functions:** Support for single-label classification loss and other tailored objectives.
- **Efficient Training:** Implements multi-GPU support and automatic learning rate scheduling.
- **Performance Evaluation:** Comprehensive metrics based on COCO evaluation standards (e.g., AP, AP50).
- **Visualization:** Annotation and model output visualization for better insights.

---

## 🛠️ Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/uod-object-detection.git
    cd uod-object-detection
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install COCO API:**
    ```bash
    pip install pycocotools
    ```

---

## 📂 Project Structure

## ⚙️ Configuration

Key parameters are defined in `modules/config.py`:

- **Paths:**
  - `TRAIN_IMAGES_PATH`: Directory for training images.
  - `TEST_IMAGES_PATH`: Directory for testing images.
  - `TRAIN_ANNOTATIONS_PATH`: Path to training annotations.
  - `TEST_ANNOTATIONS_PATH`: Path to testing annotations.

- **Model and Training:**
  - `BATCH_SIZE`: Batch size for training and testing.
  - `LEARNING_RATE`: Learning rate for the optimizer.
  - `EPOCHS`: Number of training epochs.
  - `IMAGE_SIZE`: Target size for input images.
  - `FEATURE_MAP_SIZES`, `STRIDES`, `SCALES`, `ASPECT_RATIOS`: Anchor generation parameters.

---

## 🏃‍♂️ Usage

1. **Prepare your datasets:**
   Ensure COCO-style datasets are available and correctly linked in `config.py`.

2. **Run the pipeline:**
    ```bash
    python main.py
    ```

3. **Monitor progress:**
   Training logs and visualizations are available on **[Weights & Biases](https://wandb.ai/)**.

---

## 🔍 Visualization

Leverage the `visualize_annotations` module to inspect ground truth annotations and model predictions. Outputs are saved to disk for review.

---

## 📈 Evaluation

Model performance is measured using COCO evaluation metrics:
- **AP (Average Precision):** Standard metric for object detection tasks.
- **AP50:** Precision at IoU threshold of 50%.

Metrics are logged and visualized on Weights & Biases.

---

## 🔧 Troubleshooting

- **Out-of-memory errors:** Reduce batch size or use a smaller image size in `config.py`.
- **Anchor mismatch:** Ensure `FEATURE_MAP_SIZES`, `SCALES`, and `ASPECT_RATIOS` align with your dataset's scale.
- **COCO annotation issues:** Verify that annotation files conform to COCO standards.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Submit issues or feature requests.
- Fork the repo and create a pull request with your enhancements.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---
