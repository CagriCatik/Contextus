# PCB Design Overview

## Design Objectives

The board is intended to host an ESP‑32 microcontroller along with a suite of peripheral sensors (temperature, humidity, pressure, light, motion, etc.) and a user interface that includes a display, buttons, and a USB‑to‑serial bridge. The design must accommodate a 2‑layer prototype for rapid validation and a 4‑layer final version that delivers robust power distribution, reduced noise, and controlled‑impedance routing for high‑speed signals. The primary constraints are cost, manufacturability, and signal integrity.

## Prototype vs. Final Board

### 2‑Layer Prototype

- **Purpose**: Quick, inexpensive validation of the schematic and component placement.  
- **Stackup**: Two copper layers (top and bottom) with no dedicated power or ground planes.  
- **Advantages**: Lower cost, simpler fabrication, faster turnaround.  
- **Limitations**: Higher trace impedance, limited routing space, poorer power‑plane isolation, and increased susceptibility to EMI.  
- **Trade‑off**: Acceptable for early functional testing but not suitable for the final product where noise and power integrity are critical. [Verified]

### 4‑Layer Final

- **Purpose**: Deliver a production‑grade board with reliable power distribution, controlled impedance for high‑speed signals, and reduced EMI.  
- **Stackup**: Typically two signal layers with an inner power plane and an inner ground plane, providing a reference plane for impedance control.  
- **Advantages**: Improved noise immunity, better thermal management, and the ability to route differential pairs with length matching.  
- **Limitations**: Higher cost and more complex fabrication.  
- **Trade‑off**: The cost increase is justified by the