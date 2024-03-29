Vision-Based Localization
========================

This repository houses the source code for a versatile vision library that facilitates 3D localization using cameras.

**Key Features:**

- **Scalability:** Supports an arbitrary number of cameras, scales automatically to any configuration. (Just ensure that calibration target is visible to all camera's)
- **Handle large no. of points:** The computational complexity scales as O(n^2*k), where n represents the number of points and k denotes the optimization steps.
- **Flexible Feature Detection:** Currently tested with 4 Aruco markers as detection points, the library can work with any feature as long as it's visible in multiple cameras. This allows for customization and adaptation to different environments and requirements (like using IR LEDs, treating satellite face as features).

**Data Flow Visualization**

.. figure:: _figures/flow_chart.png
    :alt: Data Flow Chart
    :width: 400

.. toctree::
   processing_flow

.. toctree::
   function_reference
