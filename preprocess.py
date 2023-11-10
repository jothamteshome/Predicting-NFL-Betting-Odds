import numpy as np
import heapq
import sympy as sym
import re
import random
from matplotlib import pyplot as plt
from bisect import bisect_left
import os
import pandas as pd
from pandasql import sqldf

# NFL teams abbreviations
nfl_teams = {'DET' : 'Detroit Lions', 'CHI' : 'Chicago Bears', 'MIN' : 'Minnesota Vikings', 'GB' : 'Green Bay Packers',
             'SF' : 'San Francisco 49ers', 'LAR' : 'Rams', 'SEA' : 'Seattle Seahawks', 'ARI' : 'Arizona Cardinals',
             'NO' : 'New Orleans Saints', 'CAR' : 'Carolina Panthers', 'ATL' : 'Atlanta Falcons', 'TB' : 'Tampa Bay Buccaneers',
             'WAS' : 'Washington', 'DAL' : 'Dallas Cowboys', 'NYG' : 'New York Giants', 'PHI' : 'Philadelphia Eagles',
             'KC' : 'Kansas City Chiefs', 'LAC' : 'Chargers', 'LVR' : 'Raiders', 'DEN' : 'Denver Broncos',
             'CLE' : 'Cleveland Browns', 'BAL' : 'Baltimore Ravens', 'CIN' : 'Cincinnati Bengals', 'PIT' : 'Pittsburgh Steelers',
             'IND' : 'Indianapolis Colts', 'TEN' : 'Tennessee Titans', 'HOU' : 'Houston Texans', 'JAX' : 'Jacksonville Jaguars',
             'NE' : 'New England Patriots', 'NYJ' : 'New York Jets', 'BUF' : 'Buffalo Bills', 'MIA' : 'Miami Dolphins',
             'LV' : 'Raiders', 'PICK' : 'even'}
# teams that underwent name changes
name_changes = {'Chargers', 'Washington', 'Raiders', 'Rams'}

# function to generate data_home.csv and data_away.csv
def load_data():
    data_home = []
    data_away = []
    labels = []
    # loop through year directories in 'data' directory
    for i_year, year in enumerate(os.listdir('data')):
        year_data_home = []
        year_data_away = []
        # loop through team directories for given year
        for i_team, team in enumerate(os.listdir(os.path.join('data', year))):
            f = os.path.join('data', year, team)
            if os.path.isfile(f):
                file = open(f, 'r', encoding='utf8')
                file_data = file.readlines()
                labels = file_data[0].strip().split(',')
                # add column labels for home team, wins, losses, and ties
                labels[9] = "OppTeam"
                labels.append('Team')
                labels.append('Wins')
                labels.append('Losses')
                labels.append('Ties')
                games_home = []
                games_away = []
                # loop through each game for given team and year
                for i, game in enumerate(file_data):
                    if i > 0:
                        g = game.strip().split(',')
                        # only consider game if it is not a bye week or not canceled
                        if g[9] != 'Bye Week' and g[4] != 'canceled':
                            # add home team to game data
                            g.append(team[:-4])
                            # split record into wins, losses, and ties
                            record = g[7].split('-')
                            win, loss, tie = int(record[0]), int(record[1]), 0 if len(record) < 3 else int(record[2])
                            # subtract outcome to get records prior to game
                            outcome = g[5]
                            if outcome == 'W':
                                win -= 1
                            elif outcome == 'L':
                                loss -= 1
                            elif outcome == 'T':
                                tie -= 1
                            # add wins, losses, and ties to game data
                            g.append(str(win))
                            g.append(str(loss))
                            g.append(str(tie))
                            # add game data to home or away data
                            if g[8] == '@':
                                games_away.append(g)
                            else:
                                games_home.append(g)
                # add team's data to year's data
                if i_team == 0:
                    year_data_home = np.array(games_home)
                    year_data_away = np.array(games_away)
                else:
                    year_data_home = np.concatenate((year_data_home, games_home), axis=0)
                    year_data_away = np.concatenate((year_data_away, games_away), axis=0)
        # add year's data to all the data
        if i_year == 0:
            data_home = year_data_home
            data_away = year_data_away
        else:
            data_home = np.concatenate((data_home, year_data_home), axis=0)
            data_away = np.concatenate((data_away, year_data_away), axis=0)

    labels = np.array(labels).reshape(-1, 1).T
    # add labels to data
    labeled_data_home = np.concatenate((labels, data_home), axis=0)
    labeled_data_away = np.concatenate((labels, data_away), axis=0)
    # turn data into df
    df_home, df_away = pd.DataFrame(labeled_data_home), pd.DataFrame(labeled_data_away)

    # make first row columns of df
    df_home.columns = df_home.iloc[0]
    df_home = df_home.drop(df_home.index[0])

    df_away.columns = df_away.iloc[0]
    df_away = df_away.drop(df_away.index[0])

    # save df as csv
    df_home.to_csv('data_home.csv', index=False)
    df_away.to_csv('data_away.csv', index=False)
    return

# function to generate data_betting.csv
def load_betting_data():
    file = open('betting_data/spreadspoke_scores.csv', 'r', encoding='utf8')
    d = file.readlines()
    d = [x.strip().split(',') for x in d]
    data = []
    for i, game in enumerate(d):
        # add home_spred and away_spread columns
        if i == 0:
            game.append("home_spread")
            game.append("away_spread")
            data.append(game)
        # only deal with data from 2002 to 2022
        elif (2002 <= int(game[1]) <= 2022 and game[3] == 'FALSE'):
            # get team name from abbreviation
            team_fav = nfl_teams[game[8]]
            # get favorite's spread
            spread_fav = float(game[9])
            home_fav = True
            # determine if the home team is the favorite
            if team_fav not in name_changes:
                if team_fav == game[4]:
                    home_fav = True
                else:
                    home_fav = False
            else:
                if team_fav in game[4]:
                    home_fav = True
                else:
                    home_fav = False
            # add home and away spreads to game data
            if team_fav == 'even':
                game.append('0')
                game.append('0')
            elif home_fav:
                game.append(str(spread_fav))
                game.append(str(-spread_fav))
            else:
                game.append(str(-spread_fav))
                game.append(str(spread_fav))
            # add game data to total data
            data.append(game)

    # turn data into df
    df = pd.DataFrame(data)
    # make first row the columns
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    # save df to csv
    df.to_csv('data_betting.csv', index=False)
    return

def sql():
    # open home, away, and betting data
    df_home = pd.read_csv("Pre-Processed Data/data_home.csv")
    df_away = pd.read_csv("Pre-Processed Data/data_away.csv")
    df_both = pd.concat([df_home, df_away], axis=0)
    df_bet = pd.read_csv("Pre-Processed Data/data_betting.csv")

    # turn dates into datetime objects
    df_home['Date'] = pd.to_datetime(df_home['Date'])
    df_away['Date'] = pd.to_datetime(df_away['Date'])
    df_both['Date'] = pd.to_datetime(df_both['Date'])
    df_bet['schedule_date'] = pd.to_datetime(df_bet['schedule_date'])

    # run sql query to put data in form we want
    df_query2 = sqldf('''SELECT bet.schedule_season as "season", home.Date, home.Outcome as "home_outcome", home.Wins as "home_wins", home.Losses as "home_losses", home.Ties as "home_ties",
     home.Team as "home_team", bet.home_spread, home.Tm as "home_score", away.Tm as "away_score", bet.away_spread, away.Team as "away_team",
     away.Wins as "away_wins", away.Losses as "away_losses", away.Ties as "away_ties", -(home.Tm - away.Tm) as "acutal_home_spread" 
    FROM df_home as home
    INNER JOIN df_away as away on home.Date = away.Date
    INNER JOIN df_bet as bet on bet.schedule_date = home.Date and bet.team_home = home.Team and bet.team_away = away.Team
    ''')

    df_query3 = sqldf('''
    SELECT bet.schedule_season as "season", both.Date, both.Week as "week", both.Team as "team", 
        both.Tm as "points", 
        both.Opp as "points_against",
        both.TotYd as "total_yards", 
        both.OppTotYd as "total_yards_against",
        both.PassY as "passing_yards", 
        both.OppPassY as "passing_yards_against",
        both.RushY as "rushing_yards", 
        both.OppRushY as "rushing_yards_against",
        both.FD as "first_downs", 
        both.OppFD as "first_downs_against",
        both.OffTO as "offense_turnovers", 
        both.DefTO as "defense_turnovers",      
        both.location, both.OppTeam as "opponent"
    FROM df_both as both
    INNER JOIN df_bet as bet on bet.schedule_date = both.Date and (bet.team_home = both.Team or bet.team_home = both.OppTeam)
    GROUP BY season, team, week, both.Tm
    ORDER BY season, team, week
    ''')
    # and bet.team_home = both.Team and bet.team_away = both.Team
    # convert date to datetime
    df_query3['Date'] = pd.to_datetime(df_query3['Date'])

    df_query4 = sqldf('''
    SELECT curr.season, curr.Date, curr.week, curr.team, curr.opponent, 
        AVG(data.points) as "points_pg", 
        AVG(data.points_against) as "points_against_pg",
        AVG(data.total_yards) as "total_yards_pg", 
        AVG(data.total_yards_against) as "total_yards_against_pg",
        AVG(data.passing_yards) as "passing_yards_pg", 
        AVG(data.passing_yards_against) as "passing_yards_against_pg",
        AVG(data.rushing_yards) as "rushing_yards_pg", 
        AVG(data.rushing_yards_against) as "rushing_yards_against_pg",
        AVG(data.first_downs) as "first_downs_pg", 
        AVG(data.first_downs_against) as "first_downs_against_pg",
        AVG(data.offense_turnovers) as "offense_turnovers_pg", 
        AVG(data.defense_turnovers) as "defense_turnovers_pg"
    FROM df_query3 as curr
    INNER JOIN df_query3 as data on data.season = curr.season and data.team = curr.team and curr.week > data.week
    GROUP BY curr.season, curr.team, curr.week
    ORDER BY curr.season, curr.team
    ''')

    df_query4['Date'] = pd.to_datetime(df_query4['Date'])

    df_query5 = sqldf('''SELECT data.season, data.Date, home_avg.week, 
        data.home_wins, data.home_losses, data.home_ties,
        
        ROUND(home_avg.points_pg, 2) as "home_points_pg", 
        ROUND(home_avg.points_against_pg, 2) as "home_points_against_pg",
        ROUND(home_avg.total_yards_pg,2) as "home_total_yards_pg", 
        ROUND(home_avg.total_yards_against_pg,2) as "home_total_yards_against_pg",
        ROUND(home_avg.passing_yards_pg,2) as "home_passing_yards_pg", 
        ROUND(home_avg.passing_yards_against_pg,2) as "home_passing_yards_against_pg",
        ROUND(home_avg.rushing_yards_pg,2) as "home_rushing_yards_pg", 
        ROUND(home_avg.rushing_yards_against_pg,2) as "home_rushing_yards_against_pg",
        ROUND(home_avg.first_downs_pg,2) as "home_first_downs_pg", 
        ROUND(home_avg.first_downs_against_pg,2) as "home_first_downs_against_pg",
        ROUND(home_avg.offense_turnovers_pg,2) as "home_offense_turnovers_pg", 
        ROUND(home_avg.defense_turnovers_pg,2) as "home_defense_turnovers_pg",
        
        data.home_team, data.home_spread, data.home_score,
        data.acutal_home_spread, 
        data.away_score, data.away_spread, data.away_team,
        data.away_wins, data.away_losses, data.away_ties, 
        
        ROUND(away_avg.points_pg,2) as "away_points_pg", 
        ROUND(away_avg.points_against_pg,2) as "away_points_against_pg",
        ROUND(away_avg.total_yards_pg,2) as "away_total_yards_pg", 
        ROUND(away_avg.total_yards_against_pg,2) as "away_total_yards_against_pg",
        ROUND(away_avg.passing_yards_pg,2) as "away_passing_yards_pg", 
        ROUND(away_avg.passing_yards_against_pg,2) as "away_passing_yards_against_pg",
        ROUND(away_avg.rushing_yards_pg,2) as "away_rushing_yards_pg", 
        ROUND(away_avg.rushing_yards_against_pg,2) as "away_rushing_yards_against_pg",
        ROUND(away_avg.first_downs_pg,2) as "away_first_downs_pg", 
        ROUND(away_avg.first_downs_against_pg,2) as "away_first_downs_against_pg",
        ROUND(away_avg.offense_turnovers_pg,2) as "away_offense_turnovers_pg", 
        ROUND(away_avg.defense_turnovers_pg,2) as "away_defense_turnovers_pg"
    FROM df_query2 as data
    LEFT JOIN df_query4 as home_avg on home_avg.season = data.season and home_avg.Date = data.Date and home_avg.team = data.home_team
    LEFT JOIN df_query4 as away_avg on away_avg.season = data.season and away_avg.Date = data.Date and away_avg.team = data.away_team
    ''')
    df_query5['Date'] = pd.to_datetime(df_query5['Date'])

    x = 10
    # save df as csv
    df_query5.to_csv('preprocessed_additional_data.csv', index=False)


def main():
    # function to generate data_home.csv and data_away.csv
    # load_data()

    # function to generate data_betting.csv
    # load_betting_data()

    # function to generate preprocessed_data.csv, need the three csv files from above
    sql()


if __name__ == "__main__":
    main()