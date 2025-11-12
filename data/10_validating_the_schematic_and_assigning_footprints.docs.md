# Footprint Assignment and Schematic Validation

## Footprint Selection Strategy

The design team opted for 0805 surface‑mount devices for the majority of resistors and decoupling capacitors. This choice reflects a common compromise between board density and manufacturing cost: 0805 components are small enough to support moderate component counts while still being inexpensive and readily available for automated pick‑and‑place assembly. [Verified]  
For high‑frequency decoupling, tandem (bypass) capacitors were selected, and a 1.0 mm test‑point pad was used for each test point. These pad dimensions are typical for test points that must accommodate probing equipment without compromising the surrounding signal integrity. [Inference]  
Standard 2.54 mm pitch connectors were chosen for the JST and B2B connectors to guarantee compatibility with off‑the‑shelf housings and to simplify the assembly process. [Verified]  

## Efficient Footprint Assignment in KiCad

KiCad’s schematic editor allows the user to assign footprints to symbols in a single operation. By selecting multiple symbols and applying a filter in the footprint browser, the designer could narrow the search to the desired 0805 or tandem capacitor footprints, thereby reducing visual clutter and speeding up the assignment process. [Verified]  
The filter mechanism is particularly useful when a project contains a large number of footprint libraries; it limits the displayed options to those that match the current search string, making it easier to locate the correct package. [Verified]  
After assigning footprints, the designer saved the project, ensuring that the footprint associations were persisted for the subsequent layout stage. [Verified]  

## Electrical Rule Check (ERC) Workflow

Before proceeding to the layout editor, the schematic was subjected to an Electrical Rule Check. ERC automatically flags several classes of violations:

1. **Unconnected power pins** – The check reported that a power pin was not driven by a power net. The designer resolved this by duplicating the power‑flag symbol and connecting it to the appropriate net. This action informs the ERC engine that the net is a power supply, preventing further power‑pin warnings. [Verified]  
2. **Missing ground connections** – A missing ground pin was identified. The designer added a ground symbol and connected the pin, satisfying the ERC requirement that all ground pins be tied to a common ground net. [Verified]  
3. **Library footprint configuration** – ERC warned that certain footprint libraries were not included in the current configuration. Although the footprints existed within the project file, the configuration did not reference the library, leading to a warning. The designer chose to ignore this warning until layout validation. [Verified]  
4. **Symbol–library mismatch** – A copy of a symbol in the project did not match the original library definition, likely due to a prior edit that moved a label. Since the mismatch did not affect functionality, the warning was suppressed. [Verified]  
5. **Pin‑type conflicts** – ERC reported that pins of type “B directional” and “power output” were connected in a way that could potentially cause a conflict. After inspecting the connections, the designer confirmed that the net assignments were correct and suppressed the warning. [Verified]  

The ERC interface offers several mechanisms to manage warnings:

- **Ignore** – Temporarily suppresses a warning for the current session.  
- **Exclude** – Adds a permanent exclusion that can be reinstated later via the exclusion list.  
- **Ignore all** – Suppresses all warnings of the same type, useful when a pattern of benign warnings is present.  
These options allow designers to focus on critical violations while deferring non‑critical issues to the layout stage. [Verified]  

## Best Practices for Footprint Assignment

- **Use multi‑select** to assign identical footprints to many symbols at once, reducing repetitive work. [Verified]  
- **Apply footprint filters** to limit the search space to relevant packages, improving navigation speed. [Verified]  
- **Verify that each symbol’s footprint matches the library definition** before proceeding to layout; mismatches can cause ERC and DRC failures. [Verified]  
- **Mark power nets with a power‑flag symbol**