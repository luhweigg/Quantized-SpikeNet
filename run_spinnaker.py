import pyNN.spiNNaker as sim
import torch


def deploy_nmnist_inference(pth_path, nmnist_spike_times, simulation_time=16.0):
    state_dict = torch.load(pth_path)

    sim.setup(timestep=1.0)

    pop_in = sim.Population(2312, sim.SpikeSourceArray(spike_times=nmnist_spike_times))
    pop_hidden = sim.Population(256, sim.IF_curr_exp())
    pop_out = sim.Population(10, sim.IF_curr_exp())

    weights_hidden = state_dict["network.1.weight"].cpu().numpy()
    weights_out = state_dict["network.4.weight"].cpu().numpy()

    sim.Projection(
        pop_in, pop_hidden, sim.FromListConnector(format_weights(weights_hidden))
    )
    sim.Projection(
        pop_hidden, pop_out, sim.FromListConnector(format_weights(weights_out))
    )

    pop_out.record(["spikes"])

    sim.run(simulation_time)

    spikes = pop_out.get_data("spikes")
    sim.end()

    return spikes


def format_weights(weight_matrix):
    connector_list = []
    for i in range(weight_matrix.shape[1]):
        for j in range(weight_matrix.shape[0]):
            connector_list.append((i, j, weight_matrix[j, i], 1.0))
    return connector_list


if __name__ == "__main__":
    pth_file = "saved_models/nmnist/run_20260708_164417/nmnist_base.pth"

    mock_spike_times = [[] for _ in range(2312)]
    mock_spike_times[0] = [1.0, 5.0]
    mock_spike_times[150] = [2.0, 10.0]

    print(f"Chargement du modèle depuis : {pth_file}")

    results = deploy_nmnist_inference(pth_file, mock_spike_times, simulation_time=16.0)

    print("\nInférence terminée. Résultats (spikes par neurone de sortie) :")
    print(results.segments[0].spiketrains)
