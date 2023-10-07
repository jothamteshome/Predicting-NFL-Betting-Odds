import os
import pandas as pd
import requests

from bs4 import BeautifulSoup, element


# Given a year, collect all the teams from that year
def collectTeams(url):
    # Request the page for a given year using requests library
    res = requests.get(url)

    # Parse content using bs4
    soup = BeautifulSoup(res.content, 'html.parser')

    # Find AFC and NFC tables
    afcTable = soup.find(id='AFC')
    nfcTable = soup.find(id='NFC')

    allTeams = []
    
    # Find all table entrys for a given team
    allTeams.extend(afcTable.findAll('a'))
    allTeams.extend(nfcTable.findAll('a'))

    teamURLs = {}
    # Build dictionary with linking a team's name to a team's url
    for i in range(len(allTeams)):
        teamURLs[allTeams[i].text] = "https://www.pro-football-reference.com" + allTeams[i].attrs['href']

    return teamURLs

# Convert a tableRow to a usable format
def convertToList(tableRow):   
    rowData = [""] * 25

    # Make sure all child elements are of type bs4.element.Tag
    children = [child for child in tableRow.children if type(child) == element.Tag]

    # If week is a bye week, return nearly empty list
    if children[9].text.strip().lower() == "bye week":
        rowData[0], rowData[9] = children[0].text, children[9].text
        return rowData
    
    # If game was cancelled, return list
    elif children[10].text.strip().lower() == "canceled":
        for i, child in enumerate(children):
            rowData[i] = child.text

        return rowData

    # Loop through all child nodes in html
    for i, child in enumerate(children):
        # If the data-stat is the game date, save in yyyy-mm-dd format
        if child.attrs['data-stat'] == 'game_date':
            rowData[i] = child.attrs['csk']
        
        # If the data-stat is the game date, save in hh:mm format
        elif child.attrs['data-stat'] == 'game_time':
            rowData[i] = ":".join(child.attrs['csk'].split('.'))
        else:
            rowData[i] = child.text

    return rowData

# Save team info to Pandas Dataframe and then convert to csv file
def saveTeamInfo(filePath, team, teamURL):
    # Get team page response using requests
    res = requests.get(teamURL)

    # Parse html using bs4
    soup = BeautifulSoup(res.content, 'html.parser')

    # Find table with games and get each table row from it
    games = soup.find(id='games')
    tableRows = games.findAll('tr')

    # Store the column names from the second table row in '#games' div
    columns = [child.text for child in tableRows[1] if type(child) == element.Tag]
    columns[3], columns[4], columns[5], columns[8] = 'Time', 'Boxscore', 'Outcome', 'Location'

    # Initialize team's dataframe with column headers
    teamData = []

    # Loop through weeks of the season (first 18 or less)
    for i in range(2, min(len(tableRows), 19)):
        teamData.append(convertToList(tableRows[i]))

    pd.DataFrame(teamData, columns=columns).to_csv(f"{filePath}/{team}.csv", index=False)

# Collect the NFL data by season for every team
# Makes sure not to collect duplicate data
def collectNFLData():
    os.makedirs('NFL Season Data', exist_ok=True)

    # Check what already exists to make sure not duplicate seasons are created
    seasons = os.listdir('NFL Season Data')
    seasons.sort()

    # Start collecting from the latest season available
    startingSeason = 2002
    if len(seasons) > 1:
        startingSeason = int(seasons[-1])

    # Loop through a range of years
    for year in range(startingSeason, 2023):
        # Make a subdirectory for each year
        os.makedirs(f'NFL Season Data/{year}', exist_ok=True)

        # Collect team urls for each year
        teamURLs = collectTeams(f"https://www.pro-football-reference.com/years/{year}")

        # Set of every team collected so duplicates are not created
        seasonsCollectedTeams = set(os.listdir(f"NFL Season Data/{year}"))

        # Save each teams season data as a csv file
        for team in teamURLs:
            if f"{team}.csv"in seasonsCollectedTeams:
                continue

            saveTeamInfo(f'NFL Season Data/{year}', team, teamURLs[team])