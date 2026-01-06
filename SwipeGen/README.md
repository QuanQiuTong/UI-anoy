# Swiper

## Overview

**Swiper** is a **Vision-Language Model (VLM)–driven automated mobile application exploration system** that operates on real Android devices and collects structured UI interaction trajectories.

The system performs exploration using a **depth-2 interaction strategy**, consisting of:

* **Level 1 (L1):** Exploration of the application home screen
* **Level 2 (L2):** Conditional exploration of secondary pages triggered by L1 interactions

Interaction success is determined exclusively via **visual state changes** observed from screenshots, without relying on UI hierarchies, accessibility APIs, or application-specific instrumentation.

This design makes Swiper applicable to a wide range of real-world mobile applications, including those dominated by WebViews or custom-rendered UI components.

---

## Usage

1. Start the inference backend (`remote_server.py`)
2. Connect an Android device with USB debugging enabled
3. Configure target application package names in `run_qwen.py`
4. Run:

```bash
python run_qwen.py
```

---

## Core System Design

### Entry Point: `run_qwen.py`

The entire exploration process is orchestrated by `run_qwen.py`.
The **central algorithmic logic** is implemented in the following method:

```
AppExplorer.explore_app()
```

This function defines the full exploration procedure, including state initialization, interaction execution, branching logic, and result serialization.

---

## Core Exploration Algorithm (`explore_app`)

The exploration process is explicitly modeled as a **depth-2 interaction tree**, designed to balance UI coverage and execution controllability.

---

### Step 1: Application Reset

* The target application is force-stopped and relaunched.
* This guarantees that every exploration episode starts from a clean and reproducible initial state.

---

### Step 2: Level-1 Page Analysis (L1)

1. A screenshot of the home page is captured.
2. The screenshot is sent to the VLM inference server.
3. The VLM returns a set of candidate interaction regions, including:

   * **Clickable regions**
   * **Slidable regions**

To improve robustness against missed detections, the system **always injects global full-screen swipe actions** (vertical and horizontal), ensuring that exploration does not stall due to model omissions.

---

### Step 3: Level-1 Interaction Execution

For each Level-1 interaction region, the system performs the following sequence:

* Execute the corresponding interaction (tap or swipe) on the real device
* Capture screenshots **before** and **after** the action
* Compute pixel-level image differences to detect UI state changes

Only interactions that induce **significant visual changes** are considered effective and eligible for further exploration.

#### Visual Change Detection

Visual change detection is implemented in `device_controller.py` via:

```
calculate_image_diff(img1_path, img2_path, bbox=None, threshold)
```

Two complementary criteria are used:

* **Local change detection (object-level):**
  The interaction’s bounding box is compared with a relatively higher threshold to determine whether the interaction itself was successful.

* **Global change detection (page-level):**
  Full-screen comparison with a lower threshold (`threshold = 5e-3`) determines whether the interaction enables potential follow-up exploration.

---

### Step 4: Level-2 Exploration (L2)

Level-2 exploration is **conditionally triggered** and executed **only if** a Level-1 interaction produces a global visual change.

The procedure is as follows:

1. The post-interaction screenshot from L1 is re-analyzed by the VLM.
2. A limited number of child interactions (clicks and/or swipes) are selected.
3. Each child interaction is executed with automatic back navigation.
4. No further recursive exploration is performed.

This design ensures:

* Strictly bounded exploration depth
* No uncontrolled navigation drift
* Stable, tree-structured interaction traces

---

### Step 5: Termination and Serialization

After all Level-1 interactions have been processed:

* The exploration episode terminates
* All interaction data are serialized and saved in JSON format

---

## Prompt Design

The core prompt used for VLM inference is defined in `detect.py`.
It instructs the model to identify interactive regions from a UI screenshot and output structured region descriptions.

```text
Analyze the mobile application UI screenshot and identify all slidable regions,
such as lists, carousels, or page-level scrolling areas.
Output at most 6 regions.

For each region, provide:
1. category: "clickable" or "slidable"
2. type: semantic type (e.g., button, list, carousel)
3. direction: sliding direction (horizontal / vertical / both)
4. bbox: bounding box [x1, y1, x2, y2], with coordinates in [0, 1000]
5. description: semantic intent of the interaction
6. interaction: interaction type ("click", "long_press", or "swipe")

Output a JSON array only. Do not include any additional text.
```

---

## Output Format

Each exploration run produces a JSON report stored in the `logs/` directory.
The report contains:

* Application package name
* Timestamp
* Device metadata
* A hierarchical interaction tree

---

### Output Structure (Simplified)

```json
{
  "structure": "depth_2_tree",
  "results": {
    "l1_clicks": [...],
    "l1_slides": [...]
  }
}
```

---

### Output Characteristics

* All coordinates are normalized to `[0, 1000]`
* Each interaction record includes:

  * Action type (tap / swipe)
  * Semantic intent
  * Derived motion parameters (direction, distance, speed)
  * Binary success signal
* Level-2 interactions are strictly nested under their triggering Level-1 interaction

This structure is directly suitable for:

* Vision-language model supervision
* Imitation learning
* Offline reinforcement learning
* UI interaction analysis
