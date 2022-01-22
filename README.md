# press_start

This repository will hold my work on my video game data analysis project.


*PROJECT DESCRIPTION WITH GOALS*
I will work on video game data collected from https://www.kaggle.com/gregorut/videogamesales ( I will do this one after I learn  how to acquire data through webscrapping: https://www.vgchartz.com/). My end goal is to create a model that can predict North American sales based on features discovered through exploratoration. The final result will be a Jupyter Notebook that contains my model to predict future sales of a game in North America.


*INITIAL HYPOTHESIS/QUESTION OF DATA*
Can you use a game's genre to predict its sales in North America?
Can you use a game's developer to predict its sales in North America? 

*DATA DICTIONARY*
rank - Ranking of overall sales
name - Name of the game
platform - Platform of the game's release (i.e. PC, Nintendo Gameboy, Nintendo Switch, Playstation, playstation 2, etc.)
year - Year of the game's release
genre - Genre of the game
publisher - Publisher of the game (i.e. Campcom, Square, Nintendo, etc.)
na_sales - Sales in North America (in millions)
eu_sales - Sales in Europe (in millions)
jp_sales - Sales in Japan (in millions)
other_sales - Sales in the rest of the world (in millions)
global_sales - Total worldwide sales (in millions)
combined_sales - Total of all sales outside of the North American region (in millions)
age_bins - Games that were released from 1980 - 2002 will be classified as 'old_af', 2002 - 2010 as 'middle_aged', and anything released after 2010 will be classified as 'noob'
top_publishers - These are publishers that have 10 or more game titles and the mean sales of these publishers are greater than the mean sales of the population.
over_five_mill - These games have sold over 5 million copies in the North American region.