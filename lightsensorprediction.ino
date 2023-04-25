#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "model.h"
#include "Adafruit_VL53L0X.h"

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
  constexpr int kTensorArenaSize = 5 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

Adafruit_VL53L0X lox = Adafruit_VL53L0X();

void setup() {
  while (!Serial);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version not matched");
  }

  static tflite::MicroMutableOpResolver<2> mmopres;
  mmopres.AddFullyConnected();
  mmopres.AddLogistic();
  static tflite::MicroInterpreter static_interpreter(model, mmopres, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("Allocate Tensor failed");
    while (1);
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  if (!lox.begin()) {
    Serial.println(F("Failed to boot VL53L0X"));
    while (1);
  }
  
  lox.startRangeContinuous();
}

void loop() {
  float vals[1][20];
  if (lox.isRangeComplete()) {
    for (int i = 0; i < 20; i++) {
      vals[0][i] = lox.readRange();
      delay(100);
    }
  }
  memcpy(model_input->data.f, vals, 20 * sizeof(float));
  interpreter->Invoke();
  float outputs[1];
  memcpy(outputs, model_output->data.f, sizeof(float));
  Serial.println(outputs[0]);
}
