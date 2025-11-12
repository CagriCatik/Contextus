# Operational Requirements and Component Selection

## Overview

The design of a custom ESP32‑based IoT board demands a careful balance between functional performance, manufacturability, and cost. The board must support wireless communication, a rich set of peripherals (sensors, storage, display, user interface), and robust power management while remaining suitable for production in a modern PCB fabrication house such as NextPCB. The following sections detail the key design decisions, constraints, and best practices that guided the project.

## Signal Conditioning and Power Integrity

### Decoupling Capacitors

Decoupling capacitors are placed as close as possible to the power pins of every integrated circuit. This placement is mandated by the ESP32 datasheet and the BME280 sensor specification, which both recommend a 0.1 µF ceramic capacitor in parallel with a 10 µF tantalum capacitor near the supply pins. The dual‑value approach filters both high‑frequency switching noise (handled by the ceramic) and low‑frequency supply droop (handled by the tantalum). The proximity of the capacitor to the ESP32’s VDD pin (C8 in the schematic) is critical; the datasheet specifies a maximum lead length of a few millimeters to maintain a low source impedance.  
[Verified]

### Pull‑Up Resistors

I²C buses, such as the one used by the BME280, require pull‑up resistors on the SDA and SCL lines. The chosen values (typically 4.7 kΩ) ensure reliable communication while keeping the bus capacitance within the limits specified by the I²C standard. These resistors are placed near the sensor to reduce trace length and associated parasitic capacitance.  
[Verified]

### Capacitor Sizing and Placement

The board uses a mix of ceramic and tantalum capacitors of varying capacitance values. The selection of each capacitor’s value is driven by the device datasheet and the expected load profile. For example, the SD card module requires a 1 µF ceramic and a 10 µF tantalum to handle both high‑frequency switching and low‑frequency voltage sag. Placement is critical: all decoupling capacitors are routed within a few millimeters of the corresponding ICs, and the largest capacitors are positioned on the inner ground plane to provide a low‑impedance return path.  
[Verified]

## Component Sourcing and BOM Management

### Supplier Selection

To guarantee manufacturability, most critical components were sourced from SnapMagic, an online distributor that provides real‑time stock information. By selecting parts that are in stock, the design team can avoid supply chain bottlenecks during production. The BOM includes the manufacturer part number, supplier reference, and a direct link to the component page, simplifying procurement for the assembly house.  
[Verified]

### 3‑D Models for Mechanical Verification

Each component on the board has an associated 3‑D model, typically obtained from the same distributor or from the manufacturer’s library. These models are imported into the PCB editor and used to generate a 3‑D representation of the assembled board. Visual inspection of the 3‑D model allows early detection of mechanical fit issues, such as protruding pins or insufficient clearance for the USB‑C connector.  
[Verified]

## Schematic Organization

### Multi‑Page Design

Given the density of the design, the schematic is split across several pages:

1. **Power Management** – battery charging, voltage regulators, and power distribution.
2. **Core Module** – ESP32, SD card, USB‑C, and flash memory.
3. **Sensors** – BME280, other analog/digital sensors.
4. **User Interface** – OLED display, buttons, and unused GPIO breakouts.

This modular approach keeps each page manageable, reduces the risk of schematic errors, and facilitates collaboration among team members.  
[Verified]

## Layout Strategy

### Four‑Layer Stack‑Up

The board uses a four‑layer stack‑up:

- **Top Layer (Layer 1)** – Signal routing for high‑speed and low‑speed signals.
- **Bottom Layer (Layer 4)** – Additional signal routing and ground return for high‑frequency signals.
- **Inner Layer 2** – Dedicated ground plane, providing a low‑impedance return path and shielding.
- **Inner Layer 3** – 3.3 V plane, supplying logic power to the ESP32 and peripheral ICs.

This configuration offers several advantages: improved signal integrity, reduced EMI, and efficient power distribution. The ground plane also serves as a thermal spreader for high‑current traces.  
[Verified]

### Copper Zones and Power Distribution

Large copper zones are defined for the ground and 3.3 V planes. The ground zone surrounds the entire board, ensuring that all power‑plane vias and power traces have a direct, low‑impedance connection to the plane. The 3.3 V zone is placed on the second inner layer, with a dedicated pad for the ESP32’s VDD pin. Power traces that carry significant current (e.g., the battery charger output or the USB‑C VBUS line) are routed as wide, low‑impedance traces and are connected to the 3.3 V plane via multiple vias to minimize voltage drop.  
[Inference]

### Trace Widths

Trace widths are chosen based on the expected current and the impedance requirements of the signal. Power traces are wider to accommodate the ESP32’s peak current draw, while data lines are kept narrow enough to maintain the desired impedance but wide enough to avoid excessive resistance. The trace width for the USB‑C VBUS line, for instance, is significantly wider than the data lines to handle the higher current.  
[Inference]

### Via Placement and Types

Through‑hole vias are used for standard signal connections, while micro‑vias are employed to connect the top and bottom signal layers to the inner planes. Blind or buried vias are avoided to keep the design simple and reduce cost. The board’s design rule set enforces minimum via diameter and clearance to meet the fabrication house’s DFM guidelines.  
[Inference]

## Manufacturing Considerations

### Design for Manufacturability (DFM)

The layout follows DFM guidelines such as:

- **Minimum Pad Size** – Ensuring pads meet the assembly house’s solder paste stencil requirements.
- **Clearance Rules** – Maintaining creepage and clearance distances for high‑voltage traces (e.g., USB‑C VBUS).
- **Via Density** – Limiting the number of vias per unit area to avoid drill collisions.
- **Component Footprint Alignment** – Using standard footprints (e.g.,