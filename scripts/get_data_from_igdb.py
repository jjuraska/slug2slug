import json
import requests
from requests.exceptions import RequestException, Timeout, HTTPError


api_url = 'https://api-endpoint.igdb.com/'
api_key = '50346e2cc40eb442ae6106f0ff9652e1'


def get_game_by_id(id):
    headers = {
        'user-key': api_key,
        'Accept': 'application/json'
    }
    payload = {
        'fields': '*'
    }

    try:
        response = requests.get(api_url + 'games/' + id,
                                params=payload,
                                headers=headers,
                                timeout=1)
        response.raise_for_status()
    except Timeout as err:
        print('IGDB request timed out:\n', err)
        return None
    except HTTPError as err:
        print(err)
        return None
    except RequestException as err:
        print(err)
        return None

    game = response.json()

    # game = verify_response(json.loads(response.read()))

    return game


def verify_response(response):
    if response['status_code'] != 1:
        raise Exception('Request status code {0}: {1}'.format(response['status_code'], response['error']))

    return response['results']


def main():
    # Tomb Raider (2013)
    # game_details = get_game_by_id('1164')

    # Shadow of the Tomb Raider (2018)
    game_details = get_game_by_id('37777')

    print(json.dumps(game_details, indent=4))


if __name__ == '__main__':
    main()
