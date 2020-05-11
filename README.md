# Demo Link
* [Demo](https://drive.google.com/open?id=1GfbK2KapEMfR1jq3f-BpkzKXOf_33_k5)
# Introduction
Corona (Covid-19) virus is a global pandemic threatening all countries with its virality and quick spread, so much so that there is not enough specialists or equipment to effectively detect patients with covid-19 to be able to supply proper measures and treatments for them.

so this project is aimed at facilitating the process of filtering potential covid-19 patients for further inspections as pneumonia is a common symptom for those affected with covid-19 (similar to SARS-Cov and MERS-Cov).

Only 18% of the patients demonstrate normal chest radiographs or CT when mild or early in the disease course, but this decreases to 3% in severe disease, of patients with COVID-19 requiring hospitalization, 69% had an abnormal chest radiograph at the initial time of admission, and 80% had radiographic abnormalities sometime during hospitalization.

Although pneumonia isn’t a symptom that certainly identifies a corona patient it is a common factor and in this current period it is a symptom worth considering.

So, this project’s goal is to be able to quickly identify pneumonia patients without the aid of doctors and specialists in an effort to speed up check-ups and perhaps even allowing for less crowding at clinics and hospitals.

# Project Folders
* GUI: This folder is the main folder and it contains the following files:
  * dialogue.ui: This file contains the user interface generated through qt creator representing the main app's ui.
  * main.py: This file contains the main logic behind the app's functionality.
  * main.pyproject: This file is the qt project file.
  * model86.h5: This file contains the trained model that is used in the main.py file to predict the dignostic of a selected image.
  
* training: This folder contains the files used in the training process of the model:
  * augmentation.py: This file augments the images in the dataset to by applying rotations and random zooms to generate a bigger dataset for training.
  * model.py: This file is used for processing the images, training the classification model and saving it to model86.h5 

* examples: This folder contains the following two subfolders:
  * Normal: Contains images of healthy patients' xrays.
  * Pneumonia: Contains images of pneumonic patients' xrays.

# Testing the project
1. Download the repo.
2. Navigate to the GUI folder.
3. In a terminal or a cmd prompt run: python main.py
4. After the program starts up just browse for the image to test ("examples" folder exists for simplicity) and the result will show up in the middle of the ui window.
