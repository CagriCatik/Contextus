# Setting Up KiCad 9 for the Project

## Project Initialization

When starting a new PCB design in KiCad 9, the first step is to create a dedicated project folder.  
KiCad will generate a set of files (`*.kicad_pro`, `*.sch`, `*.kicad_pcb`, etc.) inside this folder.  
If the folder already contains files, KiCad will warn you and allow you to continue; this is useful when you want to reuse an existing workspace for a new design.  
Naming the project clearly (e.g., `esp32_demo_project`) helps keep the workspace organized and makes it easier to locate the files later.  
[Verified]

## Library Management

### Footprints

KiCad supports both global and project‑specific libraries.  
For a design that uses many unique components, it is often preferable to keep the footprints in a **project‑specific** library.  
This approach prevents accidental reuse of the wrong footprint in other projects and keeps the global library set lean.  
The workflow is:

1. Open *Preferences → Manage Footprint Libraries*.
2. Choose **Project Specific Libraries**.
3. Browse to the project’s `libraries/footprints` directory and add the folder.
4. Give the library a descriptive nickname (e.g., `esp32_project_libs`) so that it is immediately recognizable in the library selector.  
[Verified]

### Symbols

The same principle applies to schematic symbols.  
Because the design uses a mix of generic and custom symbols, the symbols are also stored in a project‑specific library.  
When adding symbols, KiCad requires selecting each file individually; bulk selection is not supported in the current UI.  
[Verified]

### Source of Libraries

All footprints and symbols were sourced from reputable online repositories (e.g., SnapEDA) and accompanied by datasheets.  
Reading the datasheets before adding a component to the schematic is a best practice that ensures correct pin‑out, footprint dimensions, and electrical constraints.  
[Verified]

## Preferences and Grid Settings

### General Preferences

- **High‑Quality Analysis**: Enabled to improve the visual fidelity of schematic and PCB drawings.  
- **Annotation Options**: Default settings were accepted; these control how component references are generated automatically.  
- **Crosshairs**: The *four‑window crosshair* mode was selected in both the schematic and PCB editors. This feature aligns the cursor to the center of the active window, simplifying placement and routing.  
[Verified]

### Grid Configuration

KiCad allows multiple grid presets.  
The following grid values were chosen:

- **Schematic**: 150 µm, 25 µm, 10 µm (fast switching)  
- **PCB**: 50 µm, 25 µm (fast switching)  

These grids provide a balance between precision for fine components and speed for larger layout tasks.  
[Verified]

### Editing Options

The default editing options were left unchanged.  
These options govern how components snap, how the cursor behaves, and how the editor responds to user actions.  
[Verified]

## Plugin Selection

KiCad’s plugin ecosystem extends the core functionality.  
The following plugins were installed via the *Content Manager*:

| Plugin | Purpose | Recommendation |
|--------|---------|----------------|
| **Interactive HTML BOM** | Generates a web‑friendly bill of materials | Essential for tracking parts and ordering |
| **PCB Action Tools** | Quick access to common PCB actions | Useful for rapid design iteration |
| **Round Tracks** | Rounds the corners of routed traces | Improves aesthetics and reduces stress concentration |
| **Free Routing** | Provides a free‑form routing tool | Helpful for complex or irregular routing paths |
| **HQ DFM** | Performs a high‑quality design‑for‑manufacturability check | Must be run before finalizing the design for fabrication |

At a minimum, the *Free Routing* and *HQ DFM* plugins are recommended for any serious PCB project.  
[Verified]

## BOM Preparation

The bill of materials (BOM) was compiled before any schematic work began.  
Key points:

- **Partial BOM**: Initially only major components (MCU, power ICs, diodes, buttons) were listed.  
- **Generic Parts**: Resistors and capacitors were chosen as generic values to simplify prototyping.  
- **Data‑Sheet Review**: All parts were verified against their datasheets to confirm pin‑outs, package dimensions, and electrical characteristics.  
- **Time Investment**: Approximately 5–6 hours were spent gathering footprints, symbols, and datasheets.  
- **Iterative Refinement**: The BOM is expected to evolve as the schematic is fleshed out.  
[Verified]

## Best Practices and Lessons Learned

| Practice | Rationale | Evidence |
|----------|-----------|----------|
| **Use Project‑Specific Libraries** | Keeps component data isolated, reduces risk of cross‑project contamination | Explicit choice in the transcript |
| **Set Consistent Grids** | Facilitates accurate placement and routing | Grid values listed |
| **Enable High‑Quality Analysis** | Improves readability of schematic and PCB files | Preference setting |
| **Install Key Plugins Early** | Saves time during design and final checks | Plugin list |
| **Review Datasheets First** | Prevents costly mistakes in pin‑out and footprint selection | Time spent on datasheet review |
| **Iterate BOM** | Allows flexibility as design details emerge | BOM described as evolving |

These practices collectively reduce design errors, streamline the workflow, and improve the manufacturability of the final board.  
[Inference]