# Indian-Sign-Language-to-speech-device
developed and deployed Indian sign language ml model to edge device that converts hand gesture to alphabet character and then to speech.
# Task : Indian Sign Language Recognition(gesture recognition) and Conversion to speech

![sign.png](https://github.com/HarshRajTiwary/Indian-Sign-Language-to-speech-device/blob/main/sign.png)

## Objective:
Develop instructional materials for students to learn machine learning using
Jupyter Notebook and other ML tools. The aim is to accurately recognize
ISL(Indian Sign Language Recognition) gestures, convert them into
corresponding text, and then convert the text into speech.

## Tools and Technologies:
  * **Jupyter Notebook:** For data preprocessing, model development, and
training.

  * **Programming Language:** Python - Widely used for its simplicity,
extensive libraries, and ecosystem support.

## Starting the Exciting Journey for this project by following these

To make this interesting project we will be following some steps and sub-steps so that our project can be made without any errors. This is the most useful project which we are going to make, which deals with the real life problem of deaf and dumb people.

So let us start this exciting journey with the following steps:
* **Step1:** Making our system ready for this project, for that you need to download some libraries of python using `pip` command.
  * OpenCV ----> `pip install opencv-python` ----> open-source library for computer vision, machine learning, and image processing.
  * PIL ----> `pip install pillow` ----> Pillow provides the ImageDraw module that provides simple 2D graphics for Image objects.
  * tensorflow ----> `pip install tensorflow` ----> used to build and train deep learning models as it facilitates the creation of computational graphs and efficient execution on various hardware platforms
  * sklearn ----> `pip install scikit-learn` used here for spliting data into test and train.
  * pyttsx3 ----> `pip install pyttsx3` used to convert text to speech.

  Do not worry you are not supposed to copy these commends and install one by one in your terminal because you will see these commands installed at the time of there use in this notebook file only, so chill....
* **Step2:** Making Hardware ready for this project,for that we will be following some sub steps which are as followed:
  * Getting the Arduino code
  * Interfacing the Hardware
  * Installing the code to the Hardware
  * Getting the esp32_url for this project
* **Step3:** Now after sucessfully installing these libraries of python, you need to import these libraries for this project.
* **Step4:** Data collection and processing, this very steps have some sub steps which is needed to be followed:
  * **Data Collection:** Capture image form esp32-cam and save them with there class name(in our project it is A to Z gestures).
  * **Data storing:** Now after creating your data with there class name you need to store these data into csv file, in our case we are taking the image data of any pixel, resizing it to 40x40, spliting the data into test and train and after this we will be storing these data in to csv files named as `train.csv` and `test.csv`.
  * **Data Loading:** We will be loading these csv files in our code.
  * **Data Processing:** Now its time to process the data for this project which will be helpful for us to develop the model and test it for the accuracy.
  * **Debugging:** Debugging the data is most essential before model development as it debugs our process and alearts us when there is any error.
  * **Previewing Data:** After some more data processing now we will be previewing the data in grayscale as this gives us the correct image data for developing our model.
* **Step5:** Developing our ML model.
* **step6:** Deploying this ML model using esp32-cam, for this we will be following some sub steps which are as follows:
  * some necessary system commands
  * Import some libraries
  * Load ML model
  * Connect to esp32-cam
  * Prediction
  * Converting the predicted value to text
  * Converting the text to speech

and yes if you follow these steps, you will be getting your project ready.

### Interfacing the Hardware

See the beloy diagram and interface the esp32-cam as ot is (Do not forget to connect GPIO0 pin to GND).

Now connect the FTDI to the computer through the usb cable
* 5V ---> VCC
* GND ---> GND
* U0T ---> Rx
* U0R ---> Tx
* IO0 ---> GND

pin diagram for your ease.

![interfacing.png](https://github.com/HarshRajTiwary/Indian-Sign-Language-to-speech-device/blob/main/interfacing.jpg)

### Installing code to Hardware

Follow these steps to install code to your Hardware:
* **Step 1:** Open the example code and do some necessary changes so that it can be used in our project, which are as follows:
  * Comment the line no 17 which is `#define CAMERA_MODEL_ESP_EYE  // Has PSRAM`
  * uncomment the line no 25 which is `#define CAMERA_MODEL_AI_THINKER // Has PSRAM`
  * change the ssid and password
  ```
  const char *ssid = "**********";
  const char *password = "**********";
  ```

Now Install the code on esp32cam by selecting the correct board and port to which the esp32-cam is connected. To do so

Go to `Tools---> Board---> esp32---> AI Thinker ESP32-CAM`

Go to `Tools---> Port---> your port`

and hit the upload button (it will take some time to verify and upload the code to the esp32-cam).

if it shows any error the for debugging follow these steps:
* Press the reset button present on the esp32-cam module.
* Check the connection for any loose fitting.


Your Hardware is ready with the code now, only you need to get your esp32_url to get the camera access for this project. To do so you need to follow some steps which are as follows:
* Open the serial monitor
* Set the baud rate to `115200 baud`
* Remove the jumper wire which is connected to IO0 to GND of esp32-cam module
* Press the reset button present on the esp32-cam module, you will notice some url is printed on the serial monitor.
* Copy this url and paste on the browser of your PC (ensure that your pc is also connected to the same hotspot to which the esp32-cam module is connected).
* Now serach for the start stream button red in colour.
* Click that button for starting the video stream, now video will start streaming on the web page
* Now right click on the video displayed and go to open in new tab and cose the previous tab
* Copy the url of the current page and thats it you got your esp32 url.

## About Author

### Developed by Harsh Raj
Contacts :
- [Email](mailto:developerharshraj@gmail.com)
- [LinkedIn](https://in.linkedin.com/in/harsh-raj-416a0b27b)
- [GitHub](https://github.com/HarshRajTiwary)

### Happy Learning
