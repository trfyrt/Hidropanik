# Edge AI Smart Hydroponics System
This project implements a smart Nutrient Film Technique (NFT) hydroponics automation system driven by Edge AI running on an ESP32-S3 microcontroller. Unlike traditional IoT setups that rely on cloud connectivity or act as passive monitoring tools, this system executes an Artificial Neural Network (ANN) model locally and in real-time.

---

## Key Features
* **Zero Human Intervention**: The system independently analyzes water conditions and performs corrective actions (nutrient and pH dosing) without requiring manual intervention.

* **Edge AI Inference**: Artificial Intelligence model processing runs 100% locally on the ESP32-S3 using efficient embedded libraries, keeping the system fully responsive even without an internet connection.

* **Time-Duration Prediction**: Instead of just outputting standard target parameters, the AI directly predicts the exact operational runtime of the DC pumps (in seconds) to prevent chemical over-correction in the reservoir.

* **Stabilized Sensor Readings**: Equipped with a smoothing algorithm based on a data buffer `SCOUNT 30` to filter out erratic voltage fluctuations, combined with a post-activation mixing delay to ensure fluids are fully blended before the next reading.

* **NTP Time Synchronization**: Integration with Network Time Protocol (NTP) to track chronological context (vegetative growth week) as a critical input parameter for the AI model.

---

## Hardware Schematic
<p align="center">
  <img src="https://github.com/user-attachments/assets/d049368f-f249-40fb-99ca-8645ba123da9" alt="">
</p>

---

## System Workflow
1. Setup: The ESP32-S3 boots up, connects to Wi-Fi, and syncs the current plant growth week.

2. Read: It takes 30 continuous samples from the pH and TDS sensors to calculate a clean, stable average.

3. Think: The embedded AI model `model2.h` processes the sensor data locally on the chip without needing internet.

4. Act: The AI decides exactly how many seconds the pumps need to run to add nutrients or balance pH, then executes the relays and pauses briefly to let the water mix.

---

## Developer Team
1. [Michael Christianto Sawitto](https://github.com/Mic-03)
2. [Hainzel Kemal](https://github.com/HainzelK)
3. [Aaron Jevon Benedict Kongdoh](https://github.com/trfyrt)
4. Kasmir Syariati

© Tim Gabut 2026
