from typing import List

import pandas as pd

from src.Tour.Stop import Stop
from src.Utils.helper import calculate_distance

stops: List[Stop] = []


def setup_distance_matrix():
    df_new_distance_matrix = pd.DataFrame(index=[stop.hash_id for stop in stops],
                                          columns=[stop.hash_id for stop in stops])
    df_new_distance_matrix.fillna(value=0.0, inplace=True)
    return df_new_distance_matrix


distance_matrix = setup_distance_matrix()
capacity_demands = dict()


def calculate_distance_matrix() -> None:
    for stop_i in stops:
        for stop_j in stops:
            distance_i_j = calculate_distance(stop_i, stop_j)
            distance_matrix.at[stop_i.hash_id, stop_j.hash_id] = distance_i_j


def init_capacity_demands() -> None:
    for stop in stops:
        capacity_demands[stop.hash_id] = [stop.demand_weight, stop.demand_volume]


def clear():
    stops.clear()
    distance_matrix.iloc[0:0]


def add_stop(stop: Stop) -> None:
    stops.append(stop)


def get_stop(index: int) -> Stop:
    return stops[index]


def get_list_of_stops():
    return stops


def get_stop_by_hash_id(id: float) -> Stop:
    return [x for x in stops if x.hash_id == id]


def get_length_of_stops() -> int:
    return len(stops)


def get_microhub():
    return stops[0]


def get_distances():
    return distance_matrix


def get_distance_by_matrix(hash_a: float, hash_b: float) -> float:
    return distance_matrix.at[hash_a, hash_b].astype(float)


def get_capacity_demands_as_dict():
    return capacity_demands