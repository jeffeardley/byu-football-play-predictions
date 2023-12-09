import cfbd
API_KEY = 'tV0p30nGicAOlqO9NKqAP9TNKAqTUXNS7jr1gUBx5hCXa9uXynEyAvw5IHyL/tuM'

configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = 'Bearer ' + API_KEY
configuration.api_key_prefix['Authorization'] = 'Bearer'

# Create an instance of the API class
api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))

year = 2022
week = 1
season_type = 'regular'
team = 'BYU'
api_response = api_instance.get_games_for_week(year, week, season_type=season_type)

# Filter the results by team
games = [game for game in api_response if game.home_team == team or game.away_team == team]

# Get the game ID for the first game in the list
game_id = games[0].id

# Call the API method
api_response = api_instance.get_play_by_play_data(game_id, year=year, week=week, season_type=season_type, team=team)

# Print the offensive play-by-play data
for play in api_response.offense_plays:
    print(play)
