# Mindrove Armband EMG Classifier

This project focuses on developing a classifier for Electromyography (EMG) signals captured using the Mindrove Armband. The classifier aims to interpret muscle activity for various applications, such as prosthetic control or gesture recognition.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Collection](#data-collection)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [Contact](#contact)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/liadolier99/Mindrove_armband_EMG_classifier.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd Mindrove_armband_EMG_classifier
    ```

3. **(Optional) Create and activate a virtual environment:**

    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use 'env\Scripts\activate'
    ```

4. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Data Collection:**

    - Ensure the Mindrove Armband is properly connected and calibrated.
    - Use the provided script data analysis/Data_Recording.py.
2. **labeling**

   - The labeling proccess need to be done by hand, using the video of each record.
     
3. **Feature Extraction:**

    - Utilize the script data analysis/feature_extraction.py to extract relevant features from the raw EMG data.

4. **Model Training:**

    - Access the data analysis/feature_extraction.py to create SVM modle.
    - Train the classifier using the extracted features.
    - Adjust hyperparameters as necessary to optimize performance.

4. **Evaluation:**

    - Evaluate the trained model using validation datasets.
    - Analyze performance metrics to assess classifier accuracy.

## Data Analysis

For in-depth data analysis, refer to the `data analysis` directory, which contains scripts  for exploratory data analysis and visualization.

## Contributing

Contributions are welcome! To contribute:

1. **Fork the repository.**
2. **Create a new branch:**

    ```bash
    git checkout -b feature/YourFeatureName
    ```

3. **Commit your changes:**

    ```bash
    git commit -m 'Add a descriptive commit message'
    ```

4. **Push to the branch:**

    ```bash
    git push origin feature/YourFeatureName
    ```

5. **Open a pull request.**


## Contact

For questions or collaboration:

- **Name:** Liad Olier
- **Email:** [Liad.Olier@campus.technion.ac.il](mailto:your.email@example.com)
- **GitHub:** [liadolier99](https://github.com/liadolier99)

- **Name:** Roee Savion
- **Email:** [roee.savion@campus.technion.ac.il](mailto:your.email@example.com)
- **GitHub:** [roeesavion](https://github.com/roeesavion)

---

By following this template and customizing it with specific details from your project, you'll provide clear guidance to users and contributors, enhancing the project's accessibility and collaborative potential.
::contentReference[oaicite:0]{index=0}
 
