# VOCA: Voice-Activated Prosthetic Device
# OVERVIEW  
VOCA is a voice-activated prosthetic device that uses depth sensing and speech recognition for object grasping and release. It integrates an Arduino system, Intel RealSense depth camera, and ultrasonic sensors to detect distances and execute grasping commands.

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
