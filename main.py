########### Imports ###########
import sys
from PySide2.QtWidgets import QApplication, QFileDialog
from PySide2.QtCore import QFile, QObject, SIGNAL, SLOT
from PySide2.QtUiTools import QUiLoader
from keras.models import load_model
from cv2 import imread, resize, cvtColor, filter2D, medianBlur
from cv2 import COLOR_BGR2GRAY
from numpy import expand_dims, array

########### Global Variables ###########
model = load_model('model86.h5')
img = ''
########### Helper Functions ###########

def dignose_case(image_path):
    ''' Process Image and Predict'''
    global img
    img = imread(image_path)
    img = resize(cvtColor(img, COLOR_BGR2GRAY), (150, 150))

    kernel = array([[-1/9,-1/9,-1/9], [-1/9,1,-1/9], [-1/9,-1/9,-1/9]])
    img = filter2D(img, -1, kernel)
    img = medianBlur(img, 3)

    img = img / 255.
    img = expand_dims(img, 2)
    img = expand_dims(img, 0)

    prediction = model.predict(img)
    prediction = 1*(prediction >= 0.65)
    
    if prediction == 1:
        main_window.label.setText('Has Pneumonia')
    elif prediction == 0:
        main_window.label.setText('No Pneumonia')

def browse_image():
    ''' Find Image and Dignose'''
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(main_window,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
    if fileName:
        dignose_case(image_path=fileName)

########### GUI ###########
if __name__ == "__main__":
    app = QApplication([])

    ########### Load UI's XML ###########
    ui_file=QFile('dialog.ui')
    ui_file.open(QFile.ReadOnly)

    loader = QUiLoader()
    main_window = loader.load(ui_file)

    ########### Save UI Objects ###########
    browse_btn = main_window.btn_browse
    cancel_btn = main_window.btn_cancel

    ########### Close UI File ###########
    ui_file.close()

    ########### Show UI ###########
    main_window.show()

    ########### Add Button Actions ###########
    QObject.connect(browse_btn, SIGNAL('clicked()'), browse_image)
    QObject.connect(cancel_btn, SIGNAL('clicked()'), app, SLOT('quit()'))

    sys.exit(app.exec_())
