# Goosebusters

![Goosebusters](https://i.imgur.com/q4iTbps_d.png?maxwidth=520&shape=thumb&fidelity=high)  
A fun and practical solution for keeping pesky Canadian geese at bay! **Goosebusters** is a final project for the University of Waterloo Software Engineering Term 1A. It uses a Raspberry Pi 5 to control a servo, water pump, and Raspberry Pi Camera V2, creating a system capable of targeting and spraying water at geese.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Geese are great... but not when they're invading your space! **Goosebusters** offers a humane and environmentally friendly way to deter Canadian geese using an electrical water gun deterrent system. This project combines hardware and software to automate the process of keeping geese at bay.

---

## Features

- **Raspberry Pi 5 Integration**: Acts as the brain of the system, processing input and controlling hardware.
- **Servo Motor**: Adjusts the aim of the water pump for precise targeting.
- **Water Pump Control**: Activates the pump to spray water at geese.
- **Camera**: Takes visual input and uses OpenCV's live video and image processing to identify and locate geese.

---

## Hardware Requirements

- Raspberry Pi 5
- Raspberry Pi Camera Module 2 (or any other)
- Servo motor
- Water pump
- Power supply for Raspberry Pi and peripherals (A lithium-ion battery and an outlet with an adapter)
- Tubing for water pump
- Mounting hardware
- Adapters (if needed)
- A monitor for the Raspberry Pi

---

## Software Requirements

- Python 3
- Raspberry Pi OS 
- GPIO library for Raspberry Pi
- gpiozero for Servo control
- picamera2 for Raspberry Pi Camera input
- OpenCV
- Mediapipe

---

## Contributing
Claire Guo: claire.jl.guo@gmail.com
Ashley Du: ashley.yj3@gmail.com
Adrian Luk: lukchitung@gmail.com
Amelia Song: amelia.j.s1110@gmail.com
Bernard Lim: bintawachild@gmail.com

---

## License
Copyright 2023 Ashley Du, Claire Guo, Bernard Lim, Adrian Luk, Amelia Song. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at 
http://www.apache.org/licenses/LICENSE-2.0
