#include <Servo.h>

// Define pins
const int trigPin = 9;
const int echoPin = 10;
const int servoPin = 6;
const int activationDistance = 10; // Distance threshold in cm

Servo myServo;
bool graspInitiated = false;
bool releaseInitiated = false;

void setup() {
    Serial.begin(9600); // Start Serial communication
    pinMode(trigPin, OUTPUT);
    pinMode(echoPin, INPUT);
    myServo.attach(servoPin);
    myServo.write(0); // Start servo at initial position
    Serial.println("Awaiting speech command...");
}

void loop() {
    // Check for speech command input via Serial
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        if (command == "Grasp") {
            graspInitiated = true;
            releaseInitiated = false;
            Serial.println("Grasp command received. Monitoring object distance...");
        } else if (command == "Release") {
            releaseInitiated = true;
            graspInitiated = false;
            Serial.println("Release command received. Servo resetting...");
            myServo.write(0); // Move servo back to initial position
        }
    }
    
    // If grasp mode is initiated, monitor ultrasonic sensor
    if (graspInitiated) {
        int distance = getDistance();
        Serial.print("Distance: ");
        Serial.print(distance);
        Serial.println(" cm");

        // If object is within activation distance, actuate servo
        if (distance > 0 && distance <= activationDistance) {
            Serial.println("Object detected! Actuating servo...");
            myServo.write(90); // Move servo to grasp position
            delay(2000);
        }
    }
    
    delay(100); // Small delay to avoid excessive Serial output
}

// Function to get distance from ultrasonic sensor
int getDistance() {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    
    long duration = pulseIn(echoPin, HIGH);
    int distance = duration * 0.034 / 2; // Convert time to distance (cm)
    return distance;
}