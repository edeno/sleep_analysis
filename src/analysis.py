import networkx as nx
import numpy as np
import pandas as pd

from loren_frank_data_processing.track_segment_classification import (
    get_track_segments_from_graph, project_points_to_segment)


def get_replay_info(results, ripple_spikes, ripple_times, position_info,
                    track_graph, sampling_frequency, probablity_threshold):
    '''

    Parameters
    ----------
    results : xarray.Dataset, shape (n_ripples, n_position_bins, n_states,
                                     n_ripple_time)
    ripple_spikes : pandas.DataFrame (n_ripples * n_ripple_time, n_neurons)
    ripple_times : pandas.DataFrame (n_ripples, 2)
    position_info : pandas.DataFrame (n_time, n_covariates)
    track_graph : networkx.Graph
    sampling_frequency : float
    probablity_threshold : float

    Returns
    -------
    replay_info : pandas.DataFrame, shape (n_ripples, n_covariates)

    '''
    try:
        duration = (
            (results.sum(['x_position', 'y_position']) > probablity_threshold)
            .sum('time') / sampling_frequency)
    except ValueError:
        duration = (
            (results.sum('position') > probablity_threshold)
            .sum('time') / sampling_frequency)
    duration = duration.acausal_posterior.to_dataframe().unstack(level=1)
    duration.columns = list(duration.columns.get_level_values('state'))
    duration = duration.rename(
        columns=lambda column_name: column_name + '_duration')
    is_category = (duration > 0.0).rename(columns=lambda c: c.split('_')[0])
    duration = pd.concat((duration, is_category), axis=1)
    duration['is_classified'] = np.any(duration > 0.0, axis=1)
    duration['n_unique_spiking'] = get_n_unique_spiking(ripple_spikes)
    duration['n_total_spikes'] = get_n_total_spikes(ripple_spikes)
    ripple_position_info = reshape_to_segments(position_info, ripple_times)
    duration['avg_actual_x_position'] = ripple_position_info.groupby(
        'ripple_number').x_position.mean()
    duration['avg_actual_y_position'] = ripple_position_info.groupby(
        'ripple_number').y_position.mean()
    duration['avg_actual_linear_position2'] = ripple_position_info.groupby(
        'ripple_number').linear_position2.mean()
    duration['avg_actual_speed'] = ripple_position_info.groupby(
        'ripple_number').speed.mean()

    metrics = pd.DataFrame(
        [get_replay_distance_metrics(
            results, ripple_position_info, ripple_number, position_info,
            track_graph, sampling_frequency, probablity_threshold)
         for ripple_number in ripple_times.index], index=ripple_times.index)

    return pd.concat((ripple_times, duration, metrics), axis=1)


def get_replay_distance_metrics(results, ripple_position_info, ripple_number,
                                position_info, track_graph,
                                sampling_frequency, probablity_threshold):
    posterior = (results
                 .sel(ripple_number=ripple_number)
                 .acausal_posterior
                 .dropna('time'))
    map_estimate = maximum_a_posteriori_estimate(posterior.sum('state'))

    actual_positions = (ripple_position_info
                        .loc[ripple_number, ['x_position', 'y_position']]
                        .values)
    actual_track_segment_ids = (ripple_position_info
                                .loc[ripple_number, 'track_segment_id']
                                .values.squeeze().astype(int))

    distance = calculate_replay_distance(
        track_graph, map_estimate, actual_positions,
        actual_track_segment_ids, position_info)

    metrics = {
        'avg_replay_distance': np.mean(distance),
        'avg_replay_speed': np.mean(np.abs(np.diff(distance)) /
                                    sampling_frequency),
        'avg_replay_velocity': np.mean(np.diff(distance) / sampling_frequency),
    }

    for state, probability in posterior.sum('position').groupby('state'):
        state_distance = distance[probability > probablity_threshold]
        metrics[f'{state}_avg_replay_distance'] = np.mean(state_distance)
        metrics[f'{state}_avg_replay_speed'] = np.mean(
            np.abs(np.diff(state_distance)) / sampling_frequency)
        metrics[f'{state}_avg_replay_velocity'] = np.mean(
            np.diff(state_distance) / sampling_frequency)
        metrics[f'{state}_max_probability'] = np.max(np.asarray(probability))

    return metrics


def get_n_unique_spiking(ripple_spikes):
    return (ripple_spikes.groupby('ripple_number').sum() > 0).sum(axis=1)


def get_n_total_spikes(ripple_spikes):
    return ripple_spikes.groupby('ripple_number').sum().sum(axis=1)


def maximum_a_posteriori_estimate(posterior_density):
    '''

    Parameters
    ----------
    posterior_density : xarray.DataArray, shape (n_time, n_x_bins, n_y_bins)

    Returns
    -------
    map_estimate : ndarray, shape (n_time,)

    '''
    try:
        stacked_posterior = np.log(posterior_density.stack(
            z=['x_position', 'y_position']))
        map_estimate = stacked_posterior.z[stacked_posterior.argmax('z')]
        map_estimate = np.asarray(map_estimate.values.tolist())
    except KeyError:
        map_estimate = posterior_density.position[
            np.log(posterior_density).argmax('position')]
        map_estimate = np.asarray(map_estimate)[:, np.newaxis]
    return map_estimate


def get_place_field_max(classifier):
    try:
        max_ind = classifier.place_fields_.argmax('position')
        return np.asarray(
            classifier.place_fields_.position[max_ind].values.tolist())
    except AttributeError:
        return np.asarray(
            [classifier.place_bin_centers_[gpi.result().argmax()]
             for gpi in classifier.ground_process_intensities_])


def get_linear_position_order(position_info, place_field_max):
    position = position_info.loc[:, ['x_position', 'y_position']]
    linear_place_field_max = []

    for place_max in place_field_max:
        min_ind = np.sqrt(
            np.sum(np.abs(place_max - position) ** 2, axis=1)).idxmin()
        linear_place_field_max.append(
            position_info.loc[min_ind, 'linear_position2'])

    linear_place_field_max = np.asarray(linear_place_field_max)
    return np.argsort(linear_place_field_max), linear_place_field_max


def reshape_to_segments(time_series, segments):
    df = []
    for row in segments.itertuples():
        row_series = time_series.loc[row.start_time:row.end_time]
        row_series.index = row_series.index - row_series.index[0]
        df.append(row_series)

    return pd.concat(df, axis=0, keys=segments.index).sort_index()


def _get_closest_ind(map_estimate, all_positions):
    map_estimate = np.asarray(map_estimate)
    all_positions = np.asarray(all_positions)
    return np.argmin(np.linalg.norm(
        map_estimate[:, np.newaxis, :] - all_positions[np.newaxis, ...],
        axis=-2), axis=1)


def _get_projected_track_positions(position, track_segments, track_segment_id):
    projected_track_positions = project_points_to_segment(
        track_segments, position)
    n_time = projected_track_positions.shape[0]
    projected_track_positions = projected_track_positions[(
        np.arange(n_time), track_segment_id)]
    return projected_track_positions


def calculate_replay_distance(track_graph, map_estimate, actual_positions,
                              actual_track_segment_ids, position_info):
    '''Calculate the linearized distance between the replay position and the
    animal's physical position for each time point.

    Parameters
    ----------
    track_graph : networkx.Graph
        Nodes and edges describing the track
    map_estimate : ndarray, shape (n_time, n_position_dims)
        Maximum aposterior estimate of the replay
    actual_positions : ndarray, shape (n_time, 2)
        Animal's physical position during the replay
    actual_track_segment_ids : ndarray, shape (n_time,)
        Animal's track segment ID during the replay
    position_info : pandas.DataFrame

    Returns
    -------
    replay_distance_from_actual_position : ndarray, shape (n_time,)

    '''

    actual_track_segment_ids = (
        np.asarray(actual_track_segment_ids).squeeze().astype(int))

    # Find 2D position closest to replay position
    n_position_dims = map_estimate.shape[1]
    if n_position_dims == 1:
        closest_ind = _get_closest_ind(
            map_estimate, position_info.linear_position2)
    else:
        closest_ind = _get_closest_ind(
            map_estimate, position_info.loc[:, ['x_position', 'y_position']])

    df = position_info.iloc[closest_ind]
    replay_positions = df.loc[:, ['x_position', 'y_position']].values
    replay_track_segment_ids = (
        df.loc[:, ['track_segment_id']].values.squeeze().astype(int))

    track_segments = get_track_segments_from_graph(track_graph)

    # Project positions to closest edge on graph
    replay_positions = _get_projected_track_positions(
        replay_positions, track_segments, replay_track_segment_ids)
    actual_positions = _get_projected_track_positions(
        actual_positions, track_segments, actual_track_segment_ids)

    edges = np.asarray(track_graph.edges)
    replay_edge_ids = edges[replay_track_segment_ids]
    actual_edge_ids = edges[actual_track_segment_ids]
    replay_distance_from_actual_position = []

    zipped = zip(
        actual_edge_ids, replay_edge_ids, actual_positions, replay_positions,
        actual_track_segment_ids, replay_track_segment_ids)

    for (actual_edge_id, replay_edge_id, actual_pos, replay_pos,
         actual_id, replay_id) in zipped:
        track_graph1 = track_graph.copy()
        if actual_id != replay_id:
            # Add actual position node
            node_name = 'actual_position'
            node1, node2 = actual_edge_id
            track_graph1.add_path([node1, node_name, node2])
            track_graph1.remove_edge(node1, node2)
            track_graph1.nodes[node_name]['pos'] = tuple(actual_pos)

            # Add replay position node
            node_name = 'replay_position'
            node1, node2 = replay_edge_id
            track_graph1.add_path([node1, node_name, node2])
            track_graph1.remove_edge(node1, node2)
            track_graph1.nodes[node_name]['pos'] = tuple(replay_pos)
        else:
            node1, node2 = actual_edge_id

            track_graph1.add_path(
                [node1, 'actual_position', 'replay_position', node2])
            track_graph1.add_path(
                [node1, 'replay_position', 'actual_position', node2])

            track_graph1.nodes['actual_position']['pos'] = tuple(actual_pos)
            track_graph1.nodes['replay_position']['pos'] = tuple(replay_pos)
            track_graph1.remove_edge(node1, node2)

        # Calculate distance between all nodes
        for edge in track_graph1.edges(data=True):
            track_graph1.edges[edge[:2]]['distance'] = np.linalg.norm(
                track_graph1.node[edge[0]]['pos'] -
                np.array(track_graph1.node[edge[1]]['pos']))

        replay_distance_from_actual_position.append(
            nx.shortest_path_length(
                track_graph1, source='actual_position',
                target='replay_position', weight='distance'))

    return np.asarray(replay_distance_from_actual_position)
