# VOCA: Voice-Activated Prosthetic Device
# OVERVIEW  
VOCA is a voice-activated prosthetic device that uses depth sensing and speech recognition for object grasping and release. It integrates an Arduino system, Intel RealSense depth camera, and ultrasonic sensors to detect distances and execute grasping commands.

# ABSTRACT 
This project presents Voca, a non-invasive, voice-activated prosthetic arm designed to provide affordable, intuitive, and emotional resonant solutions for individuals with upper limb loss. Traditional prosthetics are often prohibitively expensive, uncomfortable due to invasive control methods, and emotionally detached from their users. Voca addresses these challenges through the integration of natural language processing for seamless voice control, AI-based autonomous grasping for functional versatility, and a 3D-printed, customizable design that prioritizes comfort and accessibility, all under a $1500 budget. The system uses a microcontroller, depth camera, and tactile sensors to enable multi-articulation, haptic feedback, and real-time object detection. Results from prototype testing demonstrated successful grasping and user-directed actuation in varied conditions. Voca reimagines prosthetic technology not only as a tool for mobility, but as meaningful extensions of a person’s will, laying the foundation for a more compassionate future in assistive technology."


# REPOSITORY STRUCTURE

**voice/** - Voice activation code.

Run python speech.py (requires Azure API setup).


**cad/** 
– CAD files for the prototype and 4-bar linkage system.


**arduino/** 
– Arduino code for servo control.


**camera/** 
– Intel RealSense depth camera testing code.


**basic_grasping_code/**
– PyBullet simulation for grasping and movement functions.

# MATERIALS

Arduino Kit

3D-printed prosthetic materials

4-bar linkage system

Spherical ball manipulator

Intel RealSense depth camera

Ultrasonic sensor

Microphone

Screws

More...

# FUNCTIONALITY

**Voice Commands:**
Simple grasp/release via speech recognition (Azure API).

**Depth Sensing:**
Uses Intel RealSense + ultrasonic sensor to detect objects.

**Automated Grasping:**
When an object is within a set distance, VOCA initiates a grasp.

# SETUP

Install dependencies.

Set up Azure API for speech recognition.

Connect Arduino and RealSense camera.

Run voice activation → python speech.py.

Test grasping using PyBullet simulation.
