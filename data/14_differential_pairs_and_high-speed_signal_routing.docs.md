# Differential Pairs and High‑Speed Signal Routing

## USB Differential Pair Routing

The USB interface requires two tightly coupled differential traces, DP and DM, that must be routed with identical lengths and controlled impedance. The designer began by selecting the differential‑pair tool (keyboard shortcut **6**) and attempted to connect the pads on the bottom layer. The initial attempt failed because the pads were too close together and the clearance rule was violated. By unlocking and shifting the USB connector component slightly to the left, the designer created the necessary space and successfully routed the pair. The traces were then refined using the **D** key, which allows 45°‑mode dragging, to achieve a more symmetrical layout and to avoid crossing the VBUS pad. The VBUS net was routed separately, using a via to connect the pad to the VBUS plane, rather than forcing the differential pair to weave around it. This approach preserves the integrity of the differential pair while satisfying the VBUS routing requirement.  
[Verified]

## I²C (I²C) Routing

The I²C bus (SCL and SDA) operates at up to 400 kHz in standard mode and 1 MHz in fast mode. Although strict length matching is not critical for I²C, the designer still aimed for minimal skew to maintain signal integrity. The SDA and SCL traces were routed on the top layer, with vias used to transition to the bottom layer where necessary. The designer moved the SPI component (U6) downward to isolate the I²C traces from high‑speed signals, thereby reducing crosstalk. The SCL trace was longer than SDA, so the designer increased SDA’s length by routing it around the ESP32 area and then fine‑tuned the length using the length‑tuning tool (shortcut **7**). By iteratively adding and removing small segments, the designer achieved a length difference of less than 0.6 mm, well within acceptable tolerances for I²C.  
[Verified] [Inference]

## Length Matching and Skew

Length matching is critical for differential pairs and for any high‑speed bus where timing skew can affect data integrity. The designer measured the routed lengths: SCL was 68.29 mm and SDA was 52.3 mm. Using the length‑tuning tool, the SDA trace was extended to 68.9 mm, bringing the two traces within 0.6 mm of each other. This process involved drawing additional segments, then trimming them with the same tool until the desired length was reached. The final lengths were verified by the DRC, confirming that the skew was acceptable for the target data rates.  
[Verified] [Inference]

## Design Tools and Shortcuts

The workflow relied heavily on keyboard shortcuts to accelerate routing:

- **6** – Select differential‑pair tool.  
- **X** – Draw a straight trace.  
- **D** – Drag in 45° mode, useful for creating clean, orthogonal routes.  
- **V** – Place a via.  
- **7** – Length‑tune a single track.

These shortcuts, combined with component unlocking and repositioning, allowed the designer to quickly resolve clearance issues and refine trace geometry.  
[Verified] [Inference]

## Practical Layout Decisions

Several practical decisions were made to balance performance, manufacturability, and board space:

1. **Component Repositioning** – Moving the USB connector and the SPI component created clearance for differential pairs and isolated sensitive I²C traces from high‑speed signals.  
2. **Layer Management** – The designer routed most high‑speed signals on the top layer, using vias to transition to the bottom layer only when necessary, thereby reducing the number of layer changes and potential impedance discontinuities.  
3. **Via Placement** – Vias were placed close to the trace ends to minimize stub length, and the designer used standard through‑hole vias rather than blind or buried vias, simplifying manufacturing.  
4. **Trace Width and Spacing** – While exact dimensions are not specified, the designer maintained consistent spacing between differential pairs and adhered to the board’s clearance rules, ensuring manufacturability and signal integrity.  
5. **Isolation of Sensitive Nets** – I²C traces were routed away from the SPI bus and other high‑speed lines, reducing crosstalk and improving noise immunity.  

These decisions reflect a trade‑off between board area, component density, and signal quality.  
[Inference] [Speculation]

## Best Practices and Lessons Learned

- **Use the Differential‑Pair Tool Early** – It automatically enforces equal trace lengths and spacing, reducing manual effort.  
- **Maintain Adequate Clearance** – Always check clearance rules before routing; moving components can resolve conflicts without compromising the design.  
- **Employ 45° Routing for Orthogonal Traces** – The **D** key facilitates clean, orthogonal routing, which helps maintain controlled impedance.  
- **Length‑Tune Critical Nets** – Even for buses where skew is not critical, small mismatches can accumulate; use the length‑tuning tool to fine‑tune trace lengths.  
- **Isolate Sensitive Nets** – Keep low‑speed, high‑noise‑tolerant nets (e.g., I²C) away from high‑speed buses to reduce crosstalk.  
- **Document Component Movements** – Record any repositioning of components, as it may affect other nets or thermal considerations.  
- **Validate with DRC/ERC** – After routing, run design rule checks to ensure all clearance, impedance, and net connectivity constraints are satisfied.  

By following these practices, designers can achieve reliable high‑speed signal routing while maintaining manufacturability and board performance.