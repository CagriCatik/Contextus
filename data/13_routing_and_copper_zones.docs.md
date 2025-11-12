# Routing and Copper Zones

## Overview

The routing phase of a multilayer board is where the abstract schematic is translated into a physical trace network that satisfies electrical, thermal, and manufacturability requirements. This chapter focuses on the manual routing approach, the systematic creation of copper zones for reference planes, and the practical techniques used to connect component pads to those planes. The goal is to achieve low‑impedance, low‑skew power delivery while maintaining clean, manufacturable geometry for high‑speed signals that will follow.

## Manual Routing vs. Auto‑Router

Choosing manual routing over an auto‑router is a deliberate design decision that trades off speed for control. Auto‑routers can quickly fill dense nets but often produce sub‑optimal trace widths, via placement, and impedance‑matching for high‑speed signals. Manual routing allows the designer to:

- **Enforce consistent trace widths** for power and ground planes, ensuring predictable impedance and current capacity.  
- **Place vias strategically** to minimize via count and avoid blind or buried via usage unless necessary.  
- **Match differential pair lengths** and maintain skew within tight tolerances.  

The designer acknowledges that the auto‑router may still be used for low‑priority nets or as a fallback, but the primary routing effort remains manual. [Verified]

## Copper Plane Strategy

### Ground Plane

A single, continuous ground plane is created on the second inner layer (the reference plane). The plane is named `gnd` and is filled with a solid fill to maximize copper area and reduce impedance. The plane is also used as a reference for all other power zones, providing a low‑impedance return path for high‑current nets. Naming conventions such as `gnd D J2` or `gnd D top one` help keep the search panel organized and avoid confusion when multiple zones share the same net. [Verified]

### Power Planes

Three primary power zones are defined:

- **3.3 V Plane** – Encompasses all 3.3 V components and is placed on the first inner layer.  
- **5 V Plane** – Covers the 5 V supply, often used for the regulator and USB VBUS.  
- **VBUS Plane** – Dedicated to the USB VBUS line, ensuring a clean return path for USB power.  

Each plane is created as a copper zone with solid fill, which improves current carrying capacity and reduces the risk of impedance discontinuities compared to a hatch fill. The zones are named descriptively (e.g., `3.3 volt top one`, `5 volt top one`) to aid navigation in the search panel. [Verified]

### Zone Creation Process

1. **Tool Selection** – The zone tool is chosen from the layer stack (inner or outer) depending on the target plane.  
2. **Naming** – Each zone is given a clear, descriptive name that reflects its net and location.  
3. **Fill Style** – Solid fill is preferred over hatch for power planes because it provides a continuous copper area, reducing resistance and improving thermal performance.  
4. **Enclosure** – The zone is drawn to encompass all relevant pads and component footprints, ensuring that the plane covers the entire area needed for the net.  
5. **Fill** – The `b` key (or equivalent) is used to apply the fill, which automatically connects the zone to any pads or vias that intersect it.  

These steps are repeated for each power plane and for the ground plane. [Verified]

## Connecting Pads to Zones

### Via Placement

Pads on the top layer that need to connect to an inner plane are linked via through‑hole or buried vias. The designer uses a **VR** (via) to connect the pad to the zone, ensuring that the via’s net matches the zone’s net. If a via is not automatically connected, the designer can manually change the via’s net property to the correct plane. This approach guarantees that the pad is electrically tied to the reference plane, providing a low‑impedance path. [Verified]

### Thermal Reliefs

Thermal reliefs are applied to pads that carry significant current. The designer selects a via size of 0.5 mm with a 0.3 mm clearance to provide a robust connection while still allowing the pad to dissipate heat effectively. Solid fill zones also help reduce the need for extensive thermal reliefs because the plane itself provides a large copper area for heat sinking. [Inference]

### Via Stackup

The stackup includes two inner layers: the first inner layer for the ground plane and the second inner layer for the power planes. Vias that connect top layer pads to these inner planes are typically through‑hole or buried, depending on the board’s manufacturing capabilities. The designer ensures that the via diameter and pad clearance meet the design rules for the chosen manufacturing process. [Speculation]

## Design Rules and Constraints

| Parameter | Typical Value | Rationale |
|-----------|---------------|-----------|
| **Clearance** | 0.2 mm (adjusted for pad inclusion) | Ensures pads are fully connected to zones without violating DRC. |
| **Trace Width** | Minimum 0.2 mm for high‑current tracks | Provides sufficient current capacity while staying within board size constraints. |
| **Via Diameter** | 0.3 mm | Balances current carrying capability and manufacturability. |
| **Thermal Relief** | 0.5 mm pad with 0.3 mm clearance | Allows heat dissipation while maintaining pad integrity. |
| **Fill Style** | Solid | Improves electrical performance and reduces impedance discontinuities. |

These rules are enforced through DRC and ERC checks, ensuring that all nets meet clearance, width, and impedance requirements. [Verified]

## Practical Tips

- **Locking Groups** – Once a zone or via group is finalized, it is locked to prevent accidental movement during subsequent routing steps.  
- **Search Panel** – The search panel’s “Zones” tab is used to review all created zones, verify their net assignments, and rename them if necessary.  
- **Adjusting Clearance** – If a pad is not included in a zone due to clearance settings, reducing the clearance to 0.2 mm can resolve the issue. This demonstrates the importance of fine‑tuning design rules for specific layout scenarios. [Inference]  
- **Solid Fill Preference** – Using solid fill for power planes yields a lower resistance path compared to hatch fill, which is especially important for high‑current nets such as VBAT and USB power. [Verified]

## Differential Pairs and High‑Speed Signals

After establishing the copper planes and ensuring all power and ground connections are robust, the next routing phase focuses on differential pairs and high‑speed clock signals. These traces require careful length matching, controlled impedance, and minimal skew. The groundwork laid by the copper zones—particularly the solid ground plane and power planes—provides a stable reference for these high‑speed signals, reducing noise and ensuring signal integrity. [Speculation]

## Summary

A disciplined approach to copper zone creation, pad‑to‑zone connectivity, and manual routing yields a board that meets electrical performance targets while remaining manufacturable. Solid fill zones, strategic via placement, and clear naming conventions collectively contribute to a reliable, high‑density PCB design. The subsequent routing of differential pairs and high‑speed signals will build upon this foundation to achieve the desired signal integrity.