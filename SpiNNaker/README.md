# 🧠 SpiNNaker Hardware Deployment

This guide explains how to deploy our trained PyTorch SNN models onto the physical **SpiNNaker supercomputer** (University of Manchester) to evaluate their real-world hardware power consumption.

## 🎯 What is this program and what does it do?

The script `run_spinnaker.ipynb` is the bridge between our software training framework and the physical neuromorphic hardware.

You cannot simply run a PyTorch `.pth` file on SpiNNaker. Instead, this program does the following automatically:

1. **Reads the PyTorch Weights:** It opens your pre-trained `.pth` file and extracts the synaptic weights.
2. **Rebuilds the Brain:** It uses the `pyNN.spiNNaker` library to recreate your network (SpikingVGG, SpikingMLP) as a physical, biological circuit made of LIF (Leaky Integrate-and-Fire) neurons.
3. **Hardware Mapping:** It calculates the routing and physically allocates the thousands of ARM cores and motherboards required to run your specific network.
4. **Physical Simulation:** It sends a dummy stream of spikes into the camera layer for 20ms and lets the physical chips calculate the output.
5. **Energy Profiling:** It records exactly how many Joules and Watts the ARM cores consumed to process the data, and saves this in a CSV report.

## 🚀 How to use it (Step-by-Step)

### Step 1: Get your Pre-trained Model

Before running this script, you must have successfully trained a model using our `main.py` PyTorch framework.
Locate your best weights file (usually named `model_best.pth` in your `saved_models/` folder).

### Step 2: Access the Hardware Portal

Log in to the Manchester SANDS Jupyter portal: <https://lab.jsc.ebrains.eu/> using your EBRAINS account.

### Step 3: Create the Exact Folder Structure

> ⚠️ **CRITICAL:** You MUST work inside the `work/` directory on the SANDS portal. Everything else is deleted when you log out.

Inside `work/`, you must create a folder named exactly `networks/`. This is where the script will look for your weights. Upload your weights into this folder and rename them to match the script's expectations:

    work/
    ├── networks/                      <-- CREATE THIS FOLDER
    │   ├── nmnist_best.pth            <-- Upload your N-MNIST weights here
    │   ├── cifar10_best.pth           <-- Upload your CIFAR-10 weights here
    │   └── dvs-gesture_best.pth       <-- Upload your DVS weights here
    │
    ├── spinnaker.cfg                  
    └── run_spinnaker.ipynb      <-- Upload the deployment script

### Step 4: Enable the Energy Profiler

In the `work/` folder, create a text file named exactly `spinnaker.cfg`. Paste the following code inside to force the machine to measure physical electricity consumption:

    [Reports]
    write_energy_report = True

### Step 5: Run the Simulation

Open the `run_spinnaker.ipynb` file in the Jupyter interface.
At the very top of the script, change the `DATASET` variable to the network you want to deploy:

    DATASET = "cifar10"  # Options: "nmnist", "cifar10", "dvs_gesture"

Run the cell.
*Note: For large models like VGG5, the "Routing" phase can take 3 to 5 minutes as the supercomputer calculates millions of cable connections.*

### Step 6: Read your Energy Results!

Once the simulation prints `Simulation completed with success`, the script will automatically organize your results.
Navigate to the automatically generated `reports/` folder:

    work/reports/
    └── cifar10/
        └── 2026-07-23-14-05-07-806431/
            └── energy_report.csv       <-- YOUR RESULTS ARE HERE

Open `energy_report.csv`. The most important scientific metric for your research is **`Simulation execution energy (active chips and cores only)`**. This tells you exactly how much energy your SNN consumed for a 20ms inference!
