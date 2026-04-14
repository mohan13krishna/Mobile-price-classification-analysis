# 📱 Mobile Price Classification Analysis       
         
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)          
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)                
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)                                     
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()                                                      
                                         
## 🚀 Project Overview                        
      
This project implements and compares **6 different machine learning classification algorithms** to predict mobile phone price ranges. The analysis provides comprehensive performance evaluation with detailed visualizations to determine the most effective approach for multi-class classification.
   
> **🎯 Key Achievement:** Reached a maximum accuracy of 96.5% with Logistic Regression.
## 🎓 Academic Information
     
**📚 Course:** Machine Learning (Semester 3-2)  
**🏛️ Institution:** ACE Engineering College   
**👩‍🏫 Faculty Guide:** Mrs. Swaroopa Mam, Assistant Professor  
**👨‍💻 Student:** [Mohan Krishna Thalla] - GitHub: [mohan13krishna](https://github.com/mohan13krishna)

## 🔧 Technologies Used

- **Python 3.x**
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn

## 🤖 Algorithms Implemented

| Algorithm | Output File | Key Strength |
|-----------|-------------|--------------|
| 🔵 **Logistic Regression** | `Output-2.png` | Probability estimates & interpretability |
| 🌳 **Decision Tree** | `Output-3.png` | Rule interpretability & feature importance |
| 🌲 **Random Forest** | `Output-4.png` | Reduces overfitting & robust performance |
| ⚡ **Support Vector Machine** | `Output-5.png` | Effective in high dimensions |
| 👥 **K-Nearest Neighbors** | `Output-6.png` | Non-parametric & local patterns |
| 🎯 **Naive Bayes** | `Output-7.png` | Fast training & probabilistic approach |

## 📊 Dataset Information

- **Training Samples:** 2000 | **Test Samples:** 1000
- **Features:** 20 mobile specifications (battery, RAM, camera, connectivity, etc.)
- **Target:** `price_range` (0: Low Cost, 1: Medium, 2: High, 3: Very High)
- **Preprocessing:** Feature scaling and stratified validation splitting

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mohan13krishna/Mobile-price-classification-analysis.git
   cd Mobile-price-classification-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Run the analysis:**
   ```bash
   python classification_analysis.py
   ```

## 🏆 Performance Results

| Rank | Algorithm | Accuracy | Performance Grade |
|------|-----------|----------|------------------|
| **1st** 🏆 | **Logistic Regression** | **96.5%** | **A+** ⭐⭐⭐⭐⭐ |
| **2nd** 🥈 | **SVM** | **89.5%** | **A-** ⭐⭐⭐⭐ |
| **3rd** 🥉 | **Random Forest** | **88.0%** | **B+** ⭐⭐⭐⭐ |
| 4th | Decision Tree | 83.0% | B ⭐⭐⭐ |
| 5th | Naive Bayes | 81.0% | B- ⭐⭐⭐ |
| 6th | KNN | 50.0% | F ⭐ |

### Key Insights:
- **Logistic Regression** dominates with 96.5% accuracy across all price categories
- **Linear methods** significantly outperform instance-based approaches
- **Feature scaling** proves critical for optimal performance

## 📋 Generated Outputs

| Output File | Content | Purpose |
|-------------|---------|---------|
| `Output-1.png` | Test Table | Initial data validation |
| `Output-2.png` | Logistic Regression CM | Linear classification analysis |
| `Output-3.png` | Decision Tree CM | Tree-based prediction |
| `Output-4.png` | Random Forest CM | Ensemble method performance |
| `Output-5.png` | SVM CM | Support vector classification |
| `Output-6.png` | KNN CM | Nearest neighbor analysis |
| `Output-7.png` | Naive Bayes CM | Probabilistic classification |
| `Output-8.png` | Performance Summary | Comparative analysis table |

## 📁 Project Structure

```
Mobile-price-classification-analysis/
├── 📄 classification_analysis.py    # Main analysis script
├── 📊 train.csv                    # Training dataset
├── 📊 test.csv                     # Testing dataset  
├── 📖 README.md                    # Documentation
└── 📁 Results/                     # Generated visualizations
    ├── Output-1.png               # Test Table
    ├── Output-2.png               # Logistic Regression CM
    ├── Output-3.png               # Decision Tree CM
    ├── Output-4.png               # Random Forest CM
    ├── Output-5.png               # SVM CM
    ├── Output-6.png               # KNN CM
    ├── Output-7.png               # Naive Bayes CM
    └── Output-8.png               # Performance Summary
```

## 🤝 Contributing

Contributions welcome! Ideas for improvement:
- Add more algorithms (XGBoost, LightGBM)
- Implement hyperparameter tuning
- Create interactive visualizations
- Add cross-validation analysis

## ⭐ Show Your Support

If this project helped you:
- ⭐ **Star** the repository
- 🍴 **Fork** for your experiments  
- 📢 **Share** with ML enthusiasts

---

## 📞 Contact

**👨‍💻 Developer:** [Mohan Krishna Thalla]  
**🔗 GitHub:** [mohan13krishna](https://github.com/mohan13krishna)  


---

<div align="center">

### 🌟 "Transforming Data into Insights, One Algorithm at a Time" 🌟

**Made with ❤️ for Machine Learning Education**

</div>
