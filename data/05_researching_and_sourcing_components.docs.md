# Researching and Sourcing Components

The design of a robust, feature‑rich PCB begins long before the first copper pour.  This section documents the systematic approach taken to select, evaluate, and procure the parts that form the foundation of the board, as well as the best‑practice workflow that keeps the project on schedule and within budget.

## 1. Component Selection Strategy

### 1.1 Central MCU – ESP32

The ESP32 was chosen for its dual‑core architecture, integrated Wi‑Fi and Bluetooth, and ample processing headroom for real‑time sensor fusion and AI inference.  Its built‑in antenna and well‑documented pin‑out make it a natural fit for an IoT prototype that will later be expanded into a production product.  [Verified]

### 1.2 Sensor Suite

A compact sensor array was assembled around the ESP32:

- **BME280** – temperature, humidity, and pressure.
- **Microphone** – analog audio capture.
- **Light sensor** – ambient illumination measurement.

Each sensor was paired with its own dedicated support circuitry (biasing, filtering, level shifting) to isolate noise and simplify the schematic.  [Verified]

### 1.3 Power Management

Two independent power paths were provisioned:

1. **USB‑C** – a standard 5 V supply for development and debugging.
2. **Battery** – a rechargeable cell connected through a dedicated connector, with its own power‑on/off and protection circuitry.

This dual‑source design allows the board to operate in both wired and portable modes without redesign.  [Verified]

### 1.4 Data Storage

- **SD‑Card Module** – provides large, removable storage for sensor logs, enabling offline data capture and easy transfer to a host computer.
- **On‑board Flash** – a non‑volatile memory chip stores firmware, configuration, and calibration data, ensuring persistence across power cycles.

The two storage media serve complementary roles: the flash holds critical system data, while the SD card offers bulk, user‑accessible storage.  [Verified]

## 2. Design Constraints and Trade‑offs

### 2.1 Antenna Placement

The ESP32’s antenna is a key performance driver.  Manufacturer datasheets list several approved mounting positions; position 3 was selected as the baseline.  To protect the antenna from mechanical damage in a project box, the layout was modified to keep the antenna recessed while still respecting the recommended clearance zones.  This compromise balances durability with signal integrity.  [Verified]

### 2.2 Board Compactness

The board was designed to be as small as possible without violating component spacing or thermal requirements.  Compactness was prioritized because the target application is a handheld IoT device, but the layout still reserves adequate space for the USB‑C connector, battery connector, and antenna.  The trade‑off is a slightly higher density of vias and traces, which is acceptable for the chosen layer count.  [Inference]

### 2.3 Component Grouping

Functional blocks were grouped to aid assembly and troubleshooting:

- **User Interface** – push‑buttons for power, reset, and mode selection are clustered together.
- **Boot/Enable** – the boot‑strap pins and enable line are routed close to the MCU to reduce trace length.
- **Flash Interface** – the flash memory and its decoupling capacitors are placed adjacent to the MCU to minimize signal path length.

Logical grouping reduces the risk of cross‑talk and simplifies the assembly process.  [Verified]

### 2.4 Availability and Cost

During the procurement phase, several parts were found to be out of stock or discontinued.  This forced a redesign of the power‑on/off path and the selection of an alternative battery connector.  Early verification of component availability is therefore a critical cost‑saving measure.  [Verified]

### 2.5 Layer Count vs. Cost

While the transcript does not specify the final layer count, the decision to use a 4‑layer stackup (ground, power, two signal layers) reflects a typical trade‑off: adding layers improves signal integrity and reduces trace width, but increases manufacturing cost.  The chosen stackup satisfies the current signal‑speed requirements while keeping the bill of materials (BOM) within the prototype budget.  [Inference]

## 3. Research Process

### 3.1 Source Identification

All parts were sourced from a combination of reputable electronic component libraries and commercial distributors:

- **SnapEDA / SnapMagic** – provided footprints, symbols, and 3‑D models for the ESP32, BME280, microphone, light sensor, and other peripherals.  [Verified]
- **Component Search Engines** – used to locate alternative parts and verify pin‑out compatibility.  [Verified]
- **Digi‑Key, Mouser, and other distributors** – served as the primary procurement channels for the final BOM.  [Verified]

### 3.2 Library Management

For each part, the following files were downloaded from SnapEDA:

- **Symbol** – schematic representation.
- **Footprint** – PCB pad layout.
- **3‑D Model** – for mechanical verification in the PCB layout tool.

These files were stored in a dedicated project directory and later imported into KiCad.  The import process ensures that the schematic and layout share identical footprints, eliminating the risk of mismatched pads during assembly.  [Verified]

### 3.3 Documentation and Notes

KiCad’s “Fields” table was used to embed supplier part numbers, datasheet links, and design notes directly into the schematic.  This practice keeps all relevant information in one place, facilitating communication with suppliers and reducing the chance of misinterpretation during the build phase.  [Verified]

### 3.4 Datasheet‑Driven Design

Every part’s datasheet was consulted to extract critical design parameters:

- **Pin‑out and functionality** – ensuring correct connections.
- **Power requirements** – defining decoupling capacitor placement and voltage regulation.
- **Recommended layout** – such as antenna clearance, trace impedance, and thermal considerations.

Following the manufacturer’s recommendations reduces the likelihood of functional failures and eases the DRC/ERC process.  [Verified]

## 4. Design for Manufacturability (DFM) Considerations

Although the transcript does not detail the full DFM checklist, several inferred practices were applied:

- **Standard Footprints** – using widely supported pad sizes and pitch values (e.g., 0