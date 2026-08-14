# SpiNNaker Hardware Deployment

This guide explains how to deploy our trained PyTorch SNN models onto the physical **SpiNNaker supercomputer** (University of Manchester) to evaluate their real-world hardware power consumption and inference capabilities.

## What are these programs and what do they do?

We provide two distinct deployment scripts to bridge our software training framework with the physical neuromorphic hardware:
*   `spinnaker_stress_test.py`: Used to rapidly map the architecture's topology onto the ARM cores to evaluate raw hardware allocation and baseline energy consumption.
*   `spinnaker_inference.py`: Used to load the actual trained PyTorch weights into the physical LIF neurons to evaluate true hardware inference activity.

You cannot simply run a PyTorch `.pth` file on SpiNNaker. Instead, these programs perform the following automatically:

1.  **Reads the PyTorch Weights:** It opens your pre-trained `.pth` file and extracts the synaptic weights.
2.  **Rebuilds the Brain:** It uses the `pyNN.spiNNaker` library to recreate your network (SpikingVGG, SpikingMLP) as a physical, biological circuit made of LIF (Leaky Integrate-and-Fire) neurons.
3.  **Hardware Mapping:** It calculates the routing and physically allocates the thousands of ARM cores and motherboards required to run your specific network.
4.  **Physical Simulation:** It sends a dummy stream of Poisson spikes (50Hz) into the camera layer for 20ms and lets the physical chips calculate the output.
5.  **Energy Profiling:** It records exactly how many Joules and Watts the ARM cores consumed to process the data, and securely saves this in a dynamically sorted CSV report.

---

## 🚀 How to use it (Step-by-Step)

### Step 1: Get your Pre-trained Model

Before running these scripts, you must have successfully trained a model using our `main.py` PyTorch framework. Locate your best weights file (usually named `model_best.pth` in your `saved_models/` folder).

### Step 2: Access the Hardware Portal

Log in to the Manchester SANDS Jupyter portal: [https://lab.jsc.ebrains.eu/](https://lab.jsc.ebrains.eu/) using your EBRAINS account.

### Step 3: Create the Exact Folder Structure

> **CRITICAL:** You MUST work inside the `work/` directory on the SANDS portal. Everything else is deleted when you log out.

Inside `work/`, you must create a folder named exactly `networks/`. This is where the scripts will look for your weights. Upload your weights into this folder and rename them strictly to match the scripts' expectations:

    work/
    ├── networks/                      <-- CREATE THIS FOLDER
    │   ├── nmnist_best.pth            <-- Upload your N-MNIST weights here
    │   ├── cifar10_dvs_best.pth       <-- Upload your CIFAR-10 DVS weights here
    │   ├── dvs_gesture_best.pth       <-- Upload your DVS Gesture weights here
    │   └── nepic_kitchens_best.pth    <-- Upload your N-EPIC Kitchens weights here
    │
    ├── spinnaker_stress_test.py       <-- Upload the stress test script
    └── spinnaker_inference.py         <-- Upload the inference script

*(Note: The scripts will automatically generate the required `.spynnaker.cfg` file to enable the hardware energy profiler!)*

### Step 4: Run the Simulation

Open the script you want to run (`spinnaker_stress_test.py` or `spinnaker_inference.py`).
At the very top of the script, change the `DATASET` variable to the network you want to deploy:

```python
DATASET = (
    "cifar10_dvs"  # Options: "nmnist", "cifar10_dvs", "dvs_gesture", "nepic_kitchens"
)
```

Run the script from your portal's terminal or Jupyter interface.
*Note: For large models like VGG5 or VGG8, the "Routing" phase can take a few minutes as the supercomputer calculates millions of cable connections.*

### Step 5: Read your Energy Results!

Once the simulation prints `Simulation completed with success`, the script will automatically organize your results into specialized directories based on the script you used.
Navigate to the automatically generated `reports/` folder:

    work/reports/
    ├── inference/
    │   └── cifar10_dvs/
    │       └── 2026-07-23-01-23-45-678910/
    │           └── energy_report.csv       <-- YOUR INFERENCE RESULTS ARE HERE
    └── stress_test/
        └── cifar10_dvs/
            └── 2026-07-23-01-23-45-678910/
                └── energy_report.csv       <-- YOUR STRESS TEST RESULTS ARE HERE

Open `energy_report.csv`. The most important scientific metric for your research is **`Simulation execution energy (active chips and cores only)`**. This tells you exactly how much energy your physical SNN consumed for a 20ms inference!