#!/usr/bin/env python3
"""
Getting Ships of Star Wars API
"""

import requests


def availableShips(passengerCount):
    """ number of passanger to hold """
    data = requests.get('https://swapi-api.hbtn.io/api/starships/')
    data = data.json()
    my_ships = []
    while(data['next']):
        for result in data['results']:
            capacity = result['passengers']
            capacity = capacity.replace(',', '')
            if capacity.isnumeric():
                if int(capacity) >= passengerCount:
                    my_ships.append(result['name'])
        data = requests.get(data['next'])
        data = data.json()
    if data['next'] is None:
        for result in data['results']:
            capacity = result['passengers']
            capacity = capacity.replace(',', '')
            if capacity.isnumeric():
                if int(capacity) >= passengerCount:
                    my_ships.append(result['name'])
    return my_ships
