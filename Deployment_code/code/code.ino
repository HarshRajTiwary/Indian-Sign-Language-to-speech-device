#include <Arduino.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "image_provider.h"
#include "model_settings.h"
#include "model_data.cpp"
#include <WiFi.h>

// // CMSIS include paths
// #include <arm_math.h>
// #include <cmsis_compiler.h>
// #include <arm_nnfunctions.h>

const char* ssid = "Mi_11x"; // Replace with your WiFi SSID
const char* password = "98765432@#@"; // Replace with your WiFi password

WiFiServer server(80);
WiFiClient live_client;
WiFiClient resp_client;
bool iscam = false;
bool isresp = false;

String index_html = "<meta charset=\"utf-8\"/>\n" \
                    "<style>\n" \
                    "#content {\n" \
                    "display: flex;\n" \
                    "flex-direction: column;\n" \
                    "justify-content: center;\n" \
                    "align-items: center;\n" \
                    "text-align: center;\n" \
                    "min-height: 100vh;}\n" \
                    "</style>\n" \
                    "<body bgcolor=\"#000000\"><div id=\"content\"><h2 style=\"color:#ffffff\">Gesture Recognition Device</h2><div>" \
                    "<img src=\"video\"></div><div style=\"color:#ffffff\">Made by Harsh Raj<img src=\"resp\"></div></div></body>";

namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;

  constexpr int kTensorArenaSize = 50 * 1024; // Adjusted size
  uint8_t tensor_arena[kTensorArenaSize];
}

void StreamCam(uint8_t *buf, size_t len) {
  live_client.print("--frame\n");
  live_client.print("Content-Type: image/jpeg\n\n");
  live_client.flush();
  live_client.write(buf, len);
  live_client.flush();
  live_client.print("\n");
  live_client.flush();
}

void liveResp(uint8_t *buf, size_t len) {
  resp_client.print("--mframe\n");
  resp_client.print("Content-Type: image/jpeg\n\n");
  resp_client.flush();
  resp_client.write(buf, len);  
  resp_client.print("\n");
  resp_client.flush();
}

void setup() {
  Serial.begin(9600);
  WiFi.begin(ssid, password);
  Serial.println("");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  String IP = WiFi.localIP().toString();
  Serial.println("IP address: " + IP);
  index_html.replace("server_ip", IP);
  server.begin();

  model = tflite::GetModel(tf_lite_quant_model_tflite); // Assuming g_model is defined in model_data.cpp
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  tflite::MicroMutableOpResolver resolver;
  resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D, tflite::ops::micro::Register_CONV_2D());
  resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED());
  resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX, tflite::ops::micro::Register_SOFTMAX());

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  Serial.println("Setup ok.");
}

void http_resp() {
  WiFiClient client = server.available();                    
  if (client.connected()) {     
    String req = "";
    while(client.available()){
      req += (char)client.read();
    }
    int addr_start = req.indexOf("GET") + strlen("GET");
    int addr_end = req.indexOf("HTTP", addr_start);
    if (addr_start == -1 || addr_end == -1) {
      Serial.println("Invalid request " + req);
      return;
    }
    req = req.substring(addr_start, addr_end);
    req.trim();
    Serial.println("Request: " + req);
    client.flush();

    String s;
    if (req == "/") {
      s = "HTTP/1.1 200 OK\n";
      s += "Content-Type: text/html\n\n";
      s += index_html;
      s += "\n";
      client.print(s);
      client.stop();
    } else if (req == "/video") {
      live_client = client;
      live_client.print("HTTP/1.1 200 OK\n");
      live_client.print("Content-Type: multipart/x-mixed-replace; boundary=frame\n\n");
      live_client.flush();
      iscam = true;
    } else if (req == "/resp") {
      resp_client = client;
      resp_client.print("HTTP/1.1 200 OK\n");
      resp_client.print("Content-Type: multipart/x-mixed-replace; boundary=mframe\n\n");
      resp_client.flush();
      isresp = true;
    } else {
      s = "HTTP/1.1 404 Not Found\n\n";
      client.print(s);
      client.stop();
    }
  }       
}

void loop() {
  if (iscam && isresp) {
    if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels, input->data.uint8, StreamCam)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
    }
    if (kTfLiteOk != interpreter->Invoke()) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    }
    TfLiteTensor* output = interpreter->output(0);

    int size = 5;
    uint8_t* inference_results = (uint8_t*)malloc(size * sizeof(uint8_t));
    if (inference_results == nullptr) {
      Serial.println("Memory allocation failed");
      return;
    }

    // Initialize the inference results
    inference_results[0] = output->data.uint8[0];
    inference_results[1] = output->data.uint8[1];
    inference_results[2] = output->data.uint8[2];
    inference_results[3] = output->data.uint8[3];
    inference_results[4] = output->data.uint8[4];
    processInferenceResults(inference_results, size);
    free(inference_results);
    delay(1000);
  } else {
    http_resp();
  }
}

void processInferenceResults(uint8_t* output_data, int size) {
  if (size < 5) return;

  uint8_t a_score = output_data[0];
  uint8_t b_score = output_data[1];
  uint8_t c_score = output_data[2];
  uint8_t d_score = output_data[3];
  uint8_t empty_score = output_data[4];

  uint8_t max_score = a_score;
  char max_char = 'A';

  if (b_score > max_score) {
    max_score = b_score;
    max_char = 'B';
  }
  if (c_score > max_score) {
    max_score = c_score;
    max_char = 'C';
  }
  if (d_score > max_score) {
    max_score = d_score;
    max_char = 'D';
  }
  if (empty_score > max_score) {
    max_score = empty_score;
    max_char = 'Empty';
  }

  Serial.println(max_char);
}

// Setup WiFi
void setupWiFi() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
}

// Setup camera
void setupCamera() {
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;

  if(psramFound()){
    config.frame_size = FRAMESIZE_QQVGA;  // Use a valid frame size macro
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QQVGA;  // Use a valid frame size macro
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Initialize the camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
}

// Load TensorFlow Lite model
void loadModel() {
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version not supported!");
    return;
  }

  static tflite::TflmOpResolver resolver;
  static tflite::MicroAllocator static_allocator(tensor_arena, kTensorArenaSize);
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, &static_allocator, error_reporter);
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded successfully");
}


// Capture image and preprocess
bool captureImage() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return false;
  }

  memcpy(input->data.int8, fb->buf, input->bytes);
  esp_camera_fb_return(fb);

  return true;
}

// Make prediction
void predictGesture() {
  if (captureImage()) {
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed");
      return;
    }

    // Get the highest scoring label
    int8_t* predictions = output->data.int8;
    int max_index = 0;
    int max_value = predictions[0];
    for (int i = 1; i < output->dims->data[1]; i++) {
      if (predictions[i] > max_value) {
        max_value = predictions[i];
        max_index = i;
      }
    }

    Serial.print("Predicted gesture: ");
    Serial.println(max_index);  // Add label mapping as needed
  }
}

