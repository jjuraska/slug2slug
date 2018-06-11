import json
import urllib.parse
import urllib.request


base_url = 'https://www.giantbomb.com/api/'
api_key = 'abf5b56b933fd2abb13a4294a2ef58ce8897c294'


def get_game_by_id(id):
    values = {
        'api_key': api_key,
        'format': 'json'
    }

    url_values = urllib.parse.urlencode(values)
    req = urllib.request.Request(base_url + 'game/' + id + '/?' + url_values)

    with urllib.request.urlopen(req) as response:
        game = verify_response(json.loads(response.read()))

    return game


def verify_response(response):
    if response['status_code'] != 1:
        raise Exception('Request status code {0}: {1}'.format(response['status_code'], response['error']))

    return response['results']


def main():
    # Tomb Raider (2013)
    game_details = get_game_by_id('3030-27312')
    print(game_details)


if __name__ == '__main__':
    main()
