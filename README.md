# 🧠 MNIST Neural Network Trainer (TensorFlow + Docker)

This project trains a simple neural network on the MNIST handwritten digits dataset using TensorFlow.  
It is packaged in a clean, modular Python script and fully containerized with Docker.

Use it for testing TensorFlow on CPU or GPU, or as a starting point for your own ML experiments.

---

## 📁 Project Structure

```
mnist-docker/
├── Dockerfile              # CPU-compatible Docker image
├── Dockerfile.gpu          # Optional GPU version
├── requirements.txt        # Python dependencies
├── first_tf_try_clean.py   # Main training script
└── README.md               # You're here!
```

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Docker installed ([Get Docker](https://www.docker.com/get-started))
- (Optional) NVIDIA GPU with drivers + [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU use

---

## 🐳 Run with Docker

### ✅ CPU (Default)
```bash
docker build -t mnist-trainer .
docker run --rm mnist-trainer
```

### ⚡ GPU (Optional)
```bash
docker build -f Dockerfile.gpu -t mnist-trainer-gpu .
docker run --rm --gpus all mnist-trainer-gpu
```

> TensorFlow will automatically detect and use your GPU if available (no need to modify code).

---

## 💾 Output

- Model is trained for 10 epochs on MNIST
- Accuracy and validation accuracy are printed
- Model is saved as `mnist_model.h5`
- Sample prediction + training curve is plotted (requires local display)

---

## 🔍 Want to Modify or Extend?

You can easily customize:
- Model architecture (`build_model()` in `first_tf_try_clean.py`)
- Training parameters (`train_model(...)`)
- Dataset (e.g. switch to Fashion-MNIST)

---

## 🧠 License

MIT License — feel free to use, fork, and build on it!

---

## 🙌 Credits

Made with ❤️ using TensorFlow, Docker, and Python 3.
Implemented by Genchang Peng, genchangp@gmail.com
