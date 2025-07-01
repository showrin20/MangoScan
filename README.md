# 🥭 MangoScan – Mango Species Classifier

A deep learning–based mango variety recognition system built using **PyTorch** and **Streamlit**, trained on the *Mangifera 2012* dataset.

#### 📌 Highlights:

* ✅ **95.05%** test accuracy using **ShuffleNet v2 x1.0**
* 🧠 Classifies **10 Bangladeshi mango varieties**:
  *Amrapali, Bari-4, Bari-7, Fazlee, Harivanga, Kanchon Langra, Katimon, Langra, Mollika, Nilambori*
* 📈 Interactive dashboard with:

  * Top-3 prediction insights
  * Real-time probability charts using **Chart.js**
  * Training history visualizations (loss & accuracy curves)
* 📦 Lightweight model (\~2.3M parameters) for efficient inference
* 🧪 Compared against MobileNet v2, EfficientNet-Lite0, and SqueezeNet
* 🌐 Web UI built with Tailwind-style theming & accessible components

#### 📂 Dataset:

* **Source**: Bharati et al., *Data in Brief*, 2025
* **DOI**: [10.1016/j.dib.2025.111560](https://doi.org/10.1016/j.dib.2025.111560)
* **Size**: 2012 labeled mango images
* **Preprocessing**: Stratified sampling, normalization, and augmentation

#### 🚀 Technologies:

`Python` • `PyTorch` • `TorchVision` • `Streamlit` • `Chart.js` • `PIL` • `Tailwind CSS`

