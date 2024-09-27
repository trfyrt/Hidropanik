#include "model2.h"
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

#define PH_PIN 12
#define TDS_PIN 13
#define RELAY_PIN1 20
#define RELAY_PIN2 21

#define ARENA_SIZE 10000  // Increase arena size if needed
#define TF_NUM_OPS 2      // Define the number of operations

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;  // Model initialization

// Input samples (4 features each)
float x0[4] = {1.587163783, 96.90713677, 5.412836217, 800};   // Input for first prediction
float x1[4] = {1.678363096, 568.0195563, 5.321636904, 1400};  // Input for second prediction
float x2[4] = {0.5677674638, 215.4064595, 6.432232536, 1200}; // Input for third prediction

// Expected outputs for reference (can be used for comparison)
float expectedOutputs[4][2] = {
    {2.081860084, 14.2},  // Expected output for x0
    {2.046783425, 25.0},  // Expected output for x1
    {2.473935591, 21.4},   // Expected output for x2
};

void setup() {
    Serial.begin(115200);
    delay(3000);  // Wait for serial connection
    Serial.println("_TENSORFLOW HIDROPONIK_");

    // Setup model
    tf.setNumInputs(4);    // 4 input features
    tf.setNumOutputs(2);   // 2 continuous output values (for regression)
    tf.resolver.AddFullyConnected();
    tf.resolver.AddSoftmax();

    // Initialize the model and check for errors
    if (!tf.begin(model_tflite).isOk()) {
        Serial.println("Error initializing the model.");
        Serial.println(tf.exception.toString());
        while (1);  // Halt execution on failure
    }
}

void predictAndPrint(float* input, float* expectedOutput) {
    if (!tf.predict(input).isOk()) {
        Serial.println("Error predicting");
        Serial.println(tf.exception.toString());
        return;
    }

    // Print the expected output for comparison
    Serial.print("Expected output: ");
    for (int i = 0; i < tf.numOutputs; i++) {
        Serial.print(expectedOutput[i]);
        if (i < tf.numOutputs - 1) {
            Serial.print(", ");  // Print a comma between expected values
        }
    }
    Serial.println();

    // Print the predicted output values
    Serial.print("Predicted output: ");
    for (int i = 0; i < tf.numOutputs; i++) {
        Serial.print(tf.output(i));
        if (i < tf.numOutputs - 1) {
            Serial.print(", ");  // Print a comma between predicted values
        }
    }
    Serial.println();

    // Print the time taken for prediction
    Serial.print("Prediction time: ");
    Serial.print(tf.benchmark.microseconds());
    Serial.println("us");
}

void loop() {
    // Predict and print results for x0, x1, x2
    predictAndPrint(x0, expectedOutputs[0]);  // Input x0, expected output {2.081860084, 14.2}
    predictAndPrint(x1, expectedOutputs[1]);  // Input x1, expected output {2.046783425, 25.0}
    predictAndPrint(x2, expectedOutputs[2]);  // Input x2, expected output {2.473935591, 21.4}

    delay(1000);  // Optional delay between predictions
}
