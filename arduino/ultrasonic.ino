// Define pins for the ultrasonic sensor
const int trigPin = 9;
const int echoPin = 10;

// Variables to store distance and duration values
long duration;
float distance;
float avgDistance;
const int numReadings = 5;
float readings[numReadings];
int index = 0;

void setup() {
    Serial.begin(9600); // Initialize serial communication
    pinMode(trigPin, OUTPUT);
    pinMode(echoPin, INPUT);
    
    // Initialize readings array
    for (int i = 0; i < numReadings; i++) {
        readings[i] = 0;
    }
}

float getDistance() {
    // Clear the trigPin
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    
    // Send a 10-microsecond pulse to trigger pin
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    
    // Read the echoPin, and calculate the duration (time taken for the echo)
    duration = pulseIn(echoPin, HIGH);
    
    // Convert duration to distance (in cm)
    float dist = duration * 0.034 / 2;
    
    // Limit range to avoid incorrect readings
    if (dist < 2 || dist > 400) {
        return -1; // Invalid reading
    }
    return dist;
}

float getAverageDistance() {
    float sum = 0;
    int count = 0;
    for (int i = 0; i < numReadings; i++) {
        if (readings[i] > 0) { // Ignore invalid readings
            sum += readings[i];
            count++;
        }
    }
    return (count > 0) ? sum / count : -1;
}

void loop() {
    // Read current distance
    distance = getDistance();
    
    // Store reading in the array
    readings[index] = distance;
    index = (index + 1) % numReadings;
    
    // Calculate moving average
    avgDistance = getAverageDistance();
    
    // Print distance readings
    Serial.print("Current Distance: ");
    if (distance == -1) {
        Serial.println("Invalid reading");
    } else {
        Serial.print(distance);
        Serial.println(" cm");
    }
    
    Serial.print("Average Distance: ");
    if (avgDistance == -1) {
        Serial.println("Not enough data");
    } else {
        Serial.print(avgDistance);
        Serial.println(" cm");
    }
  
}
