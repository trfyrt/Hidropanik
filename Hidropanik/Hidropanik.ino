#include <WiFi.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <LiquidCrystal_I2C.h>
#include "model2.h"
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

#define ARENA_SIZE 10000  // Increase arena size if needed
#define TF_NUM_OPS 2      // Define the number of operations

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;  // Model initialization


// Wi-Fi credentials
const char* ssid = "H's Galaxy A52s 5G";       // Replace with your Wi-Fi SSID
const char* password = "hpbaru12";             // Replace with your Wi-Fi password
bool start = true;

// Initialize NTP client and LCD display
WiFiUDP ntpUDP;
const long utcOffsetInSeconds = 8 * 3600;      // UTC+8 time offset
NTPClient timeClient(ntpUDP, "pool.ntp.org", utcOffsetInSeconds, 60000); // Update every 60 seconds
LiquidCrystal_I2C lcd(0x27, 16, 2);            // LCD at I2C address 0x27, 16 columns and 2 rows

// TDS sensor configurations
#define TdsSensorPin 13
#define VREF 3.3                             // Analog reference voltage (Volt) of the ADC
#define SCOUNT 30                            // Number of sample points

#define relayup 35
#define relaydown 36
#define relaynut 37
int analogBuffer[SCOUNT];                    // Array to store the analog values from ADC
int analogBufferTemp[SCOUNT];
int analogBufferIndex = 0, copyIndex = 0;
float averageVoltage = 0, tdsValue = 0, temperature = 25;

// pH sensor configurations
int pH_Value;                                // Variable to store pH sensor reading
float pH_Voltage;                            // Variable to store converted voltage value

void setup() {
  // Initialize serial monitor
  Serial.begin(115200);

  // Initialize LCD
  lcd.begin();
  lcd.backlight();

  pinMode(relayup, OUTPUT);
  pinMode(relaydown, OUTPUT);
  pinMode(relaynut, OUTPUT);
  
  // Connect to Wi-Fi
  connectToWiFi();

  // Initialize NTP client
  timeClient.begin();
  
  // Set TDS sensor pin mode
  pinMode(TdsSensorPin, INPUT);
  
  // Set pH sensor pin mode
  pinMode(12, INPUT);  // pH sensor connected to pin GPIO34 (ADC input)

  delay(3000);  // Wait for serial connection
    Serial.println("TENSORFLOW HIDROPONIK");

    // Setup model
    tf.setNumInputs(4);    // 5 input features
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

void loop() {

//   Update the time from the NTP server
  timeClient.update();

  
  // Display current time (HH:MM:SS) from NTP on LCD
  displayRealTime();

  // Get real time
  String time = getRealTime();
  
  float pH = readPhSensor();  // Get the pH voltage value
  displayPhValue(pH);         // Display the pH voltage on the LCD
  
  float tds = readTdsSensor();  // Get the TDS value
  displayTdsValue(tds);         // Display the TDS value on the LCD

  //prediction start here
  if (time=="12:30:00" or time=="12:35:00" or time=="12:40:00" or time=="12:45:00" or time=="12:50:00" or time=="12:55:00" or time=="13:00:00" or time=="13:05:00" or time=="13:10:00" or time=="13:15:00" or time=="13:20:00" or start){
      bool start = false;
      float x0[5] = {pH, tds, 6, 600};   // Input for first prediction
      predictAndPrint(x0);

      float pumppH = tf.output(0) * 1000; //pompa ph
      float pumpTDS = tf.output(1) * 1000; //pompa tds
    if (pumppH>0){
      digitalWrite(relayup, LOW);    // Writing value "LOW" to the pin
      delay(pumppH);                  // Delay in miliseconds 
      digitalWrite(relayup, HIGH);   // Writing value "HIGH" to the pin
      delay(1000);
    }
    if (pumppH<0){
      float pumppHlast = pumppH *-1;
      digitalWrite(relaydown, LOW);    // Writing value "LOW" to the pin
      delay(pumppHlast );                  // Delay in miliseconds 
      digitalWrite(relaydown, HIGH);   // Writing value "HIGH" to the pin
      delay(1000);
    }
    if (pumpTDS>0){
      digitalWrite(relaynut, LOW);    // Writing value "LOW" to the pin
      delay(pumpTDS);                  // Delay in miliseconds 
      digitalWrite(relaynut, HIGH);   // Writing value "HIGH" to the pin
      delay(1000);
    } 
  }
  
  delay(1000);  // Update every second
}

// Function to connect to Wi-Fi
void connectToWiFi() {
  lcd.setCursor(0, 0);
  lcd.print("Connecting...");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Connected!");
  delay(2000);
  lcd.clear();
}

// Function to display the real-time from NTP on the LCD
void displayRealTime() {
  lcd.setCursor(0, 0);

  // Format time as HH:MM:SS for UTC+8
  int hours = timeClient.getHours();
  int minutes = timeClient.getMinutes();
  int seconds = timeClient.getSeconds();

  // Display formatted time on LCD
  if (hours < 10) lcd.print("0");
  lcd.print(hours);
  lcd.print(":");

  if (minutes < 10) lcd.print("0");
  lcd.print(minutes);
  lcd.print(":");

  if (seconds < 10) lcd.print("0");
  lcd.print(seconds);
}

String getRealTime() {
  // Format time as HH:MM:SS for UTC+8
  int hours = timeClient.getHours();
  int minutes = timeClient.getMinutes();
  int seconds = timeClient.getSeconds();

  // Create the time string
  String timeString = "";

  // Format the hours
  if (hours < 10) timeString += "0";
  timeString += String(hours);
  timeString += ":";

  // Format the minutes
  if (minutes < 10) timeString += "0";
  timeString += String(minutes);
  timeString += ":";

  // Format the seconds
  if (seconds < 10) timeString += "0";
  timeString += String(seconds);

  // Return the time string
  return timeString;
}

// Function to read TDS sensor value and display it on the LCD
// Function to read TDS sensor and return TDS value
float readTdsSensor() {
  static unsigned long analogSampleTimepoint = millis();
  if (millis() - analogSampleTimepoint > 40U) { // Every 40 milliseconds, read the analog value from the ADC
    analogSampleTimepoint = millis();
    analogBuffer[analogBufferIndex] = analogRead(TdsSensorPin); // Read the analog value and store it in the buffer
    analogBufferIndex++;
    if (analogBufferIndex == SCOUNT) {
      analogBufferIndex = 0;
    }
  }

  // Calculate average voltage and TDS value every 800 milliseconds
  static unsigned long calculationTimepoint = millis();
  if (millis() - calculationTimepoint > 800U) {
    calculationTimepoint = millis();
    
    // Copy buffer data for calculation
    for (int copyIndex = 0; copyIndex < SCOUNT; copyIndex++) {
      analogBufferTemp[copyIndex] = analogBuffer[copyIndex];
    }

    // Calculate average voltage
    averageVoltage = getMedianNum(analogBufferTemp, SCOUNT) * (float)VREF / 4096.0;

    // Temperature compensation
    float compensationCoefficient = 1.0 + 0.02 * (temperature - 25.0);
    float compensationVoltage = averageVoltage / compensationCoefficient;

    // Convert voltage to TDS value (ppm)
    float tdsValue = (133.42 * compensationVoltage * compensationVoltage * compensationVoltage
                      - 255.86 * compensationVoltage * compensationVoltage
                      + 857.39 * compensationVoltage) * 0.354;

    // Return the calculated TDS value
    return tdsValue;
  }

  // If not enough time has passed for a new calculation, return the last known TDS value
  return tdsValue;
}

// Function to display the TDS value on the LCD
void displayTdsValue(float tdsValue) {
  lcd.setCursor(0, 1);     // Set the cursor position on the LCD
  lcd.print("TDS: ");      // Print "TDS: " label
  lcd.print(tdsValue, 0);  // Print the TDS value with no decimal places
}

// Function to read pH sensor and return pH voltage
float readPhSensor() {
  // Read the analog value from the pH sensor (connected to GPIO34)
  int pH_Value = analogRead(12);

  // Convert analog value to voltage (assuming 12-bit ADC)
  float pH_Voltage = pH_Value * (3.3 / 4096.0);

  // Return the calculated pH voltage
  return pH_Voltage;
}

// Function to display the pH voltage on the LCD
void displayPhValue(float pH_Voltage) {
  lcd.setCursor(8, 1);         // Set the cursor position on the LCD (next to TDS)
  lcd.print("pH: ");           // Print "pH: " label
  lcd.print(pH_Voltage, 2);    // Print the pH voltage with 2 decimal places
}

// Function to predict and print the output
void predictAndPrint(float* input) {
    if (!tf.predict(input).isOk()) {
        Serial.println("Error predicting");
        Serial.println(tf.exception.toString());
        return;
    }

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



// Function to get the median number in an array
int getMedianNum(int bArray[], int iFilterLen) {
  int bTab[iFilterLen];
  for (byte i = 0; i < iFilterLen; i++) {
    bTab[i] = bArray[i];
  }
  int i, j, bTemp;
  for (j = 0; j < iFilterLen - 1; j++) {
    for (i = 0; i < iFilterLen - j - 1; i++) {
      if (bTab[i] > bTab[i + 1]) {
        bTemp = bTab[i];
        bTab[i] = bTab[i + 1];
        bTab[i + 1] = bTemp;
      }
    }
  }
  if ((iFilterLen & 1) > 0) {
    bTemp = bTab[(iFilterLen - 1) / 2];
  } else {
    bTemp = (bTab[iFilterLen / 2] + bTab[iFilterLen / 2 - 1]) / 2;
  }
  return bTemp;
}
