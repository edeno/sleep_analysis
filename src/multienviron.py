import logging
import os

import numpy as np
from dask.distributed import Client
from replay_trajectory_classification import SortedSpikesClassifier
from replay_trajectory_classification.continuous_state_transitions import (
    RandomWalk, Uniform)
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.initial_conditions import \
    UniformOneEnvironmentInitialConditions
from replay_trajectory_classification.observation_model import ObservationModel
from src.load_data import load_data, load_sleep_data
from loren_frank_data_processing import make_epochs_dataframe
from src.parameters import PROCESSED_DATA_DIR, ANIMALS

logging.basicConfig(level="INFO",
                    format="%(asctime)s %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")


def is_common(epoch_key, neuron_info, common_ind):
    return neuron_info.index.isin([epoch_key + ind for ind in common_ind])


def get_common_ind(neuron_infos):
    neuron_keys = [
        set(neuron_info.reset_index(['animal', 'day', 'epoch']).index)
        for neuron_info in neuron_infos]
    return list(neuron_keys[0].intersection(*neuron_keys[1:]))


def return_st(env1, env2):
    if env1 == env2:
        return RandomWalk(env1)
    else:
        return Uniform(env1, env2)


def decode(epoch_keys, cur_env):
    logging.info('Loading data...')
    
    epoch_info = make_epochs_dataframe(ANIMALS)
    
    data = dict()
    for env, epoch_key in epoch_keys.items():
        if epoch_info.loc[epoch_key].type != 'sleep':
            data[env] = load_data(epoch_key)
        else:
            data[env] = load_sleep_data(epoch_key)

    Client(n_workers=6, threads_per_worker=4)
    state_names = np.sort(list(epoch_keys.keys()))
    environments = [
        Environment(environment_name=env_name,
                    track_graph=data[env_name]['track_graph'],
                    edge_order=data[env_name]['edge_order'],
                    edge_spacing=data[env_name]['edge_spacing'])
        for env_name in state_names]

    initial_conditions = UniformOneEnvironmentInitialConditions(cur_env)

    observation_models = [ObservationModel(state) for state in state_names]

    continuous_transition_types = [
        [return_st(env_name1, env_name2) for env_name2 in state_names]
        for env_name1 in state_names]

    classifier = SortedSpikesClassifier(
        environments=environments,
        initial_conditions_type=initial_conditions,
        observation_models=observation_models,
        continuous_transition_types=continuous_transition_types,
        sorted_spikes_algorithm='spiking_likelihood_glm',
        sorted_spikes_algorithm_params={
            'knot_spacing': 8.0
        },
    )

    common_ind = get_common_ind([env_data['neuron_info']
                                 for env_data in data.values()])

    position = []
    environment_labels = []
    is_training = []
    spikes = []

    for env, epoch_key in epoch_keys.items():
        position_info = data[env]['position_info']
        position.append(position_info.linear_position.values)

        n_time = position_info.shape[0]
        environment_labels.append(np.asarray([env] * n_time))

        is_training.append(position_info.speed > 4)

        spikes.append(data[env]['spikes'].iloc[:, is_common(
            epoch_key, data[env]['neuron_info'], common_ind)])

    position = np.concatenate(position)
    environment_labels = np.concatenate(environment_labels)
    is_training = np.concatenate(is_training)
    spikes = np.concatenate(spikes, axis=0)

    logging.info('Fitting model...')
    classifier.fit(
        position=position,
        spikes=spikes,
        environment_labels=environment_labels,
        is_training=is_training)

    results = classifier.predict(
        spikes=data[cur_env]['spikes'].iloc[:, is_common(
            epoch_keys[cur_env], data[cur_env]['neuron_info'], common_ind)],
        time=data[cur_env]['spikes'].index / np.timedelta64(1, 's'),
        use_gpu=True,
        state_names=state_names)

    logging.info('Saving results...')
    epoch_key = epoch_keys[cur_env]
    results_filename = os.path.join(
        PROCESSED_DATA_DIR,
        f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}"
        f"_{len(state_names):02d}_environment_results.nc"
    )
    results.to_netcdf(results_filename)
    model_filename = os.path.join(
        PROCESSED_DATA_DIR,
        f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}"
        f"_{len(state_names):02d}_environment_classifier.pkl"
    )
    classifier.save_model(model_filename)
    logging.info('Done!')

    return data, classifier, results, common_ind



def decode2D(epoch_keys, cur_env):
    logging.info('Loading data...')
    
    epoch_info = make_epochs_dataframe(ANIMALS)
    
    data = dict()
    for env, epoch_key in epoch_keys.items():
        if epoch_info.loc[epoch_key].type != 'sleep':
            data[env] = load_data(epoch_key)
        else:
            data[env] = load_sleep_data(epoch_key)

    Client(n_workers=6, threads_per_worker=4)
    state_names = np.sort(list(epoch_keys.keys()))
    environments = [
        Environment(environment_name=env_name)
        for env_name in state_names]

    initial_conditions = UniformOneEnvironmentInitialConditions(cur_env)

    observation_models = [ObservationModel(state) for state in state_names]

    continuous_transition_types = [
        [return_st(env_name1, env_name2) for env_name2 in state_names]
        for env_name1 in state_names]

    classifier = SortedSpikesClassifier(
        environments=environments,
        initial_conditions_type=initial_conditions,
        observation_models=observation_models,
        continuous_transition_types=continuous_transition_types,
        sorted_spikes_algorithm='spiking_likelihood_kde_gpu',
        sorted_spikes_algorithm_params={
            'position_std': 6.0
        },
    )

    common_ind = get_common_ind([env_data['neuron_info']
                                 for env_data in data.values()])

    position = []
    environment_labels = []
    is_training = []
    spikes = []

    for env, epoch_key in epoch_keys.items():
        position_info = data[env]['position_info']
        position.append(position_info[['x_position', 'y_position']].values)

        n_time = position_info.shape[0]
        environment_labels.append(np.asarray([env] * n_time))

        is_training.append(position_info.speed > 4)

        spikes.append(data[env]['spikes'].iloc[:, is_common(
            epoch_key, data[env]['neuron_info'], common_ind)])

    position = np.concatenate(position)
    environment_labels = np.concatenate(environment_labels)
    is_training = np.concatenate(is_training)
    spikes = np.concatenate(spikes, axis=0)

    logging.info('Fitting model...')
    classifier.fit(
        position=position,
        spikes=spikes,
        environment_labels=environment_labels,
        is_training=is_training)

    results = classifier.predict(
        spikes=data[cur_env]['spikes'].iloc[:, is_common(
            epoch_keys[cur_env], data[cur_env]['neuron_info'], common_ind)],
        time=data[cur_env]['spikes'].index / np.timedelta64(1, 's'),
        use_gpu=True,
        state_names=state_names)

    logging.info('Saving results...')
    epoch_key = epoch_keys[cur_env]
    results_filename = os.path.join(
        PROCESSED_DATA_DIR,
        f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}"
        f"_{len(state_names):02d}_environment_results2D.nc"
    )
    results.to_netcdf(results_filename)
    model_filename = os.path.join(
        PROCESSED_DATA_DIR,
        f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}"
        f"_{len(state_names):02d}_environment_classifier2D.pkl"
    )
    classifier.save_model(model_filename)
    logging.info('Done!')

    return data, classifier, results, common_ind