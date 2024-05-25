# Health Recommender System with Skin and Lung Cancer Detection

## Overview

This project is a Health Recommender System designed to assist in the early detection and management of health conditions, specifically focusing on Skin and Lung Cancer. The system leverages various recommendation techniques and integrates advanced machine learning models for cancer detection to provide comprehensive healthcare recommendations.

## Features

1. **Recommendation Techniques:**
   - Implemented various recommendation techniques with a primary focus on content-based filtering.
   - The system suggests potential diseases and corresponding treatment plans based on user-reported symptoms.

2. **Cancer Detection:**
   - Integrated Convolutional Neural Network (CNN) models for Skin and Lung Cancer detection.
   - Utilized TensorFlow and Keras frameworks for building and training the models.

## Technologies Used

- **Programming Languages:** Python
- **Libraries and Frameworks:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Flask 
- **Tools:** Jupyter Notebook, Git, VS Code

## Installation

To get a local copy up and running, follow these simple steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/asad-ishtiaque/health-recommender-system.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd health-recommender-system/
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Note:** You may need to change directories for models and datasets depending on your paths.

1. **Run the application (for HRS):**
   ```bash
   python symptoms_api.py 
   ```
2. **Run the application (for Detections):**
   ```bash
   python lung_cancer_detect.py  or skin_cancer_detect.py
   ```

3. **Interact with the system:**
   - Input your symptoms to receive disease and treatment recommendations.
   - Upload medical images for Skin and Lung Cancer detection.



## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Asad Ishtiaque - [asad.ishtiaque.ai@gmail.com](mailto:asad.ishtiaque.ai@gmail.com)

Project Link: [https://github.com/asad-ishtiaque/health-recommender-system](https://github.com/asad-ishtiaque/health-recommender-system)

---

Feel free to reach out with any questions or feedback! Thank you for checking out this project.
