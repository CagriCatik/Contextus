# PCB Design Overview – ESP32‑C3‑02 Prototype

## 1. System Architecture

The board integrates an ESP32‑C3‑02 module, a Li‑Po battery charger IC, a low‑dropout regulator, and power‑management peripherals. The design is split into hierarchical sheets: the top‑level sheet contains the USB‑C connector, charger IC, LDO, battery connector, and status LEDs; a child sheet hosts the ESP32 module and its associated hardware. This hierarchical approach simplifies the top‑level schematic, eases signal routing, and supports modular design for future revisions.

## 2. Power Management

### 2.1 USB‑C Interface

- **Connector Placement**: The USB‑C connector is positioned to allow the VBUS pin to feed the charger IC. A 5 V label is attached to the VBUS net, and a 3.3 V label is used for the regulator output. The VBUS pin is routed to the charger IC’s input, ensuring a clean power path.  
  *[Verified]*

- **Input Filtering**: A 0.1 µF ceramic capacitor and a 0.1 µF electrolytic capacitor are placed in tandem close to the charger IC’s 2 V output pins. The electrolytic is positioned nearer the IC pins to reduce inductance and improve transient response.  
  *[Verified]*

- **Power Indicator**: A 3.3 V‑driven LED with a current‑limiting resistor is wired to ground. When power is present, the LED illuminates, providing a quick visual cue. The LED is labeled “Power LED” for easy identification in the layout.  
  *[Verified]*

### 2.2 Low‑Dropout Regulator (LDO)

- **Input/Output Labels**: The regulator receives 5 V and delivers 3.3 V. The output is marked with a dedicated 3.3 V symbol, and the input with a 5 V label, ensuring consistency across the schematic.  
  *[Verified]*

- **Filtering Strategy**: Two capacitors (0.1 µF ceramic and 0.1 µF electrolytic) are placed in tandem near the regulator’s VOUT pins. This proximity reduces output ripple and improves noise performance.  
  *[Verified]*

- **Ground Plane**: Separate ground symbols are used for the regulator and the filtering capacitors to maintain clear reference planes and avoid inadvertent ground loops.  
  *[Verified]*

### 2.3 Battery Connector

- The battery connector is oriented with pins facing inward. A VBAT net is routed to the connector’s positive pin, and a ground net to the negative pin.  
  *[Verified]*

### 2.4 Charging Status LEDs

- **Red LED**: Indicates charging power is good (connected to STAT 2).  
- **Blue LED**: Indicates charging status (connected to STAT 2).  
- **Green LED**: Indicates battery charged (connected to STAT 1).  
  *[Verified]*

- The LEDs are repositioned to match their intended functions, and the associated nets are labeled (“Charging Power Good”, “Charged”, “Charging”) to aid later layout stages.  
  *[Verified]*

## 3. Hierarchical Design

- A hierarchical shade symbol is created for the ESP32 module and its peripherals. This symbol is placed on a separate sheet named “ESP32 C3‑02”.  
  *[Verified]*

- The hierarchical structure allows the top‑level sheet to reference the ESP32 sheet via the shade symbol, promoting modularity and simplifying future revisions.  
  *[Verified]*

## 4. Design Decisions & Trade‑offs

| Decision | Rationale | Impact |
|----------|-----------|--------|
| **Use of 5 V and 3.3 V labels** | Consistency across schematic and layout; reduces confusion during DRC/DRC. | Improves trace identification and reduces errors. |
| **Separate ground symbols** | Enhances clarity; prevents accidental ground loops. | Slight increase in component count but improves reliability. |
| **Capacitor placement** | Electrolytic closer to regulator pins to reduce inductance. | Improves filtering performance; [Inference] |
| **LED labeling** | Facilitates quick status identification in layout. | Reduces design time and debugging effort. |
| **Hierarchical sheets** | Enables modular design and easier maintenance. | Simplifies complex board management. |

## 5. Lessons Learned

- **Labeling is critical**: Adding descriptive labels for nets (e.g., “Charging Power Good”, “Charged”) greatly simplifies later stages of layout and verification.  
  *[Verified]*

- **Component proximity matters**: Placing filtering capacitors close to the regulator pins reduces trace length and improves noise performance.  
  *[Inference]*

- **Use of power flags**: When a component’s pin is a power input, attaching a power flag informs the ERC tool that the designer is aware, preventing false error messages.  
  *[Verified]*

- **Re‑routing LEDs for status indication**: Adjusting LED connections to match intended status signals (charging vs. charged) ensures correct visual feedback.  
  *[Verified]*

- **Consistent labeling across sheets**: Maintaining uniform net names across hierarchical sheets aids traceability and reduces confusion during multi‑sheet design.  
  *[Verified]*

## 6. Recommendations for Future Iterations

- **Controlled impedance for USB‑C**: While not explicitly addressed, the differential pair routing for USB‑C signals should consider length matching and impedance control to maintain signal integrity.  
  *[Speculation]*

- **DFM considerations**: The use of through‑hole vias for power and ground planes is acceptable for a low‑cost board; however, if higher density is required, microvias could be explored.  
  *[Speculation]*

- **Power‑plane design**: A dedicated ground plane beneath the LDO and USB‑C traces would improve noise immunity and reduce EMI.  
  *[Speculation]*

- **Thermal management**: The LDO and charger IC may dissipate significant power; adding thermal vias or copper pours around these components could mitigate hotspot formation.  
  *[Speculation]*

These insights provide a solid foundation for the subsequent layout and manufacturing stages, ensuring that the final board meets performance, reliability, and cost objectives.