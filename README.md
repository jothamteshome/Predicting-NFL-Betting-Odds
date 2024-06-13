# Predicting NFL Betting Odds


## 1. Introduction
Over the past two decades, the popularity of sports has grown tremendously. With many
new ways to bring in audiences, such as online videos and social media, the viewership of
major sports leagues in the United States has gone through the roof. Alongside the increase in
the viewership of these sports, there has also been a major increase in sports betting. Following
a 2018 U.S. Supreme Court ruling, sports betting has become legal in some form in 38 states
and the District of Columbia [3]. Additionally, it was projected that 73.5 million American adults
were going to bet on the NFL during the 2023-2024 season, which equates to about 28% of all
adults in America [1]. Clearly, sports betting has become very prevalent and there is a lot of
money involved with it.
A sportsbook is a platform, typically online, that facilitates sports betting. They provide
various different betting markets and they provide odds and lines for every market. Sports
betting markets can be based on many different characteristics of a single game. One of the
most popular bets that can be placed on the points spread, or the amount of points you expect
one team to have over another team. For example, if Team 1 has a spread of ‘-7.5’ and Team 2
ends the game with 17 points, Team 1 would need to have a score of at least 25 for a bet on
Team 1 ‘-7.5’ to be a winning bet. On the contrary, if Team 2 has a spread of ‘+7.5’ and Team 1
ends the game with 17 points, Team 2 would have to score 10 points for a bet on Team 2 ‘+7.5’
to be a winning bet.
In this project, we aim to predict the points spread of an NFL game using many different
factors, including home and away team records and average per game metrics. To make these
predictions effectively, we tested various regression models, including linear regression, ridge
regression, lasso regression, random forest regression, voting regression, support vector
regression, and a neural network. We believe training all of these various models will give us
better insight into where we can improve model performance by introducing a different feature
set, as well as being able to compare the accuracy of a purely linear model with a non-linear
model.
Our main goal with this project is to predict the spreads of NFL games as accurately as
possible. We believe that if we can accurately predict these point spreads, we can leverage our
model against existing sportsbook odds, which can aid us in making more educated predictions.
In the long run, this can be beneficial to users as well, since making more educated predictions
will help users win more bets. Ideally, we aim to train a model that can predict the point spread
of NFL games more accurately than the sportsbooks. While we know this will be difficult, since

sportsbooks have access to more data than we can obtain, if we are able to predict better than
them, it could be very successful in winning bets. It is also worth noting that the spread a
sportsbook predicts opens up at a certain number and moves depending on if there is an
imbalance in the number of people betting on either side. This is due to the sportsbook wanting
a balance of betters on both sides of the spread, allowing for them to profit the most money.
This is an aspect that we will not be considering, so it will be interesting to see how our model's
prediction compares to the sportsbook.


## 2. Methodology
### 2.1 Datasets
For this project, we created our dataset by joining data from two sources: Pro Football
Reference [2] and Kaggle’s “NFL Scores and Betting Data” dataset [4]. The Pro Football
Reference data contains the results of every team’s games, including the score of the game, the
team’s record, offensive stats, defensive stats, and expected points for the offense, defense,
and special teams. The “NFL Scores and Betting Data” dataset contains betting spread data for
all games since sports betting became a thing. Using these two datasets we created our own
with data from both. Our dataset initially consisted of the date of the game, followed by the
home and away team’s name, wins, losses, ties, spread, and score of the game. Using this data
we were not able to obtain great results, since we essentially were only using each team’s
record to predict the spread of the game. As an attempt to create a more robust dataset and
improve our models, we were able to access and calculate more metrics from the Pro Football
Reference source. These metrics included points per game, points against per game, total
yards, per game, total yards against per game, passing yards per game, passing yards against
per game, rushing yards per game, rushing yards against per game, first downs per game, first
downs against per game, offensive turnover, and defensive turnovers for both the home team
and the away team.. By using these additional metrics we created a more robust dataset that
we believed would allow our models to perform better.

### 2.2 Data Collection
To collect the data from Pro Football Reference, we had to create a Python script to
scrape the data from the website. This web scraper grabbed data for every game from every
team between the years 2002 to 2022. Our scraper consists of two parts. The first part collects
the teams for any given year, and the second part takes each team and collects the relevant
data for that team’s season. To collect the teams, we make use of the Python requests module,
which allows us to send a request to the website and receive a response. We also make use of
the BeautifulSoup Python package. Using BeautifulSoup, we parse the web page containing the
teams that played in a given season to find two tables, labeled ‘AFC’ and ‘NFC’, which we can
see in Figure 1.
<br/><br/>
![AFC and NFC tables for the 2022 NFL Season](https://github.com/jothamteshome/Predicting-NFL-Betting-Odds/assets/94427010/e1790cf8-a6d4-4a6c-9ca8-fc9ea71bf752)
<p align="center">Figure 1. AFC and NFC tables for the 2022 NFL Season</p>
<br/><br/><br/>

![Schedule and Game Results for 2022 Detroit Lions](https://github.com/jothamteshome/Predicting-NFL-Betting-Odds/assets/94427010/9db074be-ef15-4c12-96e3-61095ed86306)
<p align="center">Figure 2. Schedule and Game Results for 2022 Detroit Lions</p>
<br/><br/><br/>

We do an additional search on this table to find the table row elements, which contain
stats including team record, home team score, and away team score. Finally, we take each table
row in HTML format and check that the data fits our expectations before manually converting
each HTML row into a list format, which we can then use to create our team data CSV for the
given season. To obtain the data from Kaggle’s “NFL Scores and Betting Data” dataset, we
simply had to download the dataset from Kaggle. This dataset consisted of data dating back to

the 1966 season and playoff game data. Our first step was to convert this data to a pandas data
frame and to remove all the data before the 2002 season and all of the playoff data, since we
are only training on regular season data. The dataset also gave the favorite team as a
three-letter abbreviation and the favorite’s spread. Our next step was to use the abbreviation to
figure out the home spread and away spread. Finally, we saved this data frame as a betting data
CSV file to be used later.
After scraping the data from Pro Football Reference, we were given the data in
directories separated by years and team names. The first step was to combine this data into two
CSV files, one for the home team data and one for the away team data. The next step was to
convert these two files and the betting data file to pandas data frames, which allowed us to use
pandas SQL to join them into one pandas data frame with all the data needed from each game
in one row. The first SQL query we ran combined the home and away data with the home and
away spread, so every row contained the home and away team’s records and spread. The
second query we ran resulted in a data frame where we added the additional metrics from Pro
Football Reference, points, points against, total yards, total yards against, passing yards,
passing yards against, rushing yards, rushing yards against, first downs, first downs against,
offensive turnovers, and defensive turnovers, to the home and away dataframe. The third query
resulted in a dataframe that consisted of the average metrics, from the second query, for a given
team in a given season in the weeks before that game. The final query we ran joined the
average metrics, from the third query to the game and betting data, from the first query. This
final query resulted in our complete dataset where every row corresponds to a given game,
consisting of the home and away team’s record, average metrics (stated above), and betting
spread.

### 2.3 Model Training
For this project, we trained regression models that would predict the home spread of
games being played. The models that were utilized involved both deep learning and non-deep
learning models, and involved both linear models as well as non-linear models. The use of
linear and non-linear models, we believed, would provide insight into the problem space of
spread prediction concerning the team performance metrics.
For non-deep learning regression, we tested Linear Regression, Lasso Regression,
Ridge Regression, Support Vector Regression, and Random Forest Regression, as well as
combining each of these into a single Voting Regression model to determine if it would show
better results than any single model. To set up these models, we used the scikit-learn Python
package.
Initially, we began training and testing our models on the original dataset that we had
compiled. Using the various scikit-learn models, we were able to predict the points spread of
any given NFL game. To do this, we utilized most of the features in our dataset, including home
team record, away team record, home team score, away team score, and the outcome of the
game. We can see the hyperparameters used when initially training our models on our initial
dataset in Table 1 below.
<br/><br/>

<p align="center"><img src="https://github.com/jothamteshome/Predicting-NFL-Betting-Odds/assets/94427010/d9718e69-8cf8-4065-ae4f-4a6ccfbfc4fa"/></p>
<p align="center">Table 1. Initial Model Hyperparameters</p>
<br/><br/><br/>

As we can see, many of these hyperparameters were chosen as a baseline to determine
the performance of our model on the initial set of data. Using these hyperparameters, we were
able to see subpar results, with our best model being the Support Vector Regressor, with the
Voting Regressor coming in at a close second. As our best model, the SVR was able to achieve
an MSE of 17.04, and an R-squared 2 of 0.558. We can see the performance of each model in
Table 2 below.
<br/><br/>

<p align="center"><img src="https://github.com/jothamteshome/Predicting-NFL-Betting-Odds/assets/94427010/0b0b11cf-ae33-445f-8833-d59512b9e9a5"/></p>
<p align="center">Table 2. Model Results on Initial Data w/ Initial Hyperparameters</p>
<br/><br/><br/>

As we can see from the table above, we were not able to achieve very great results on
our initial dataset. Thus, we began testing our models using the improved dataset described in
Section 2.2. When we began using this set of data instead, we decided to begin by using the

same model hyperparameters as we had used initially. Doing this, we were able to see a
significant improvement in the result of every model excluding the SVR model, which took a
massive hit in performance. While the SVR came out on top in our training using our initial data,
we found that it achieved a mean-squared error of 24.93 and an R-squared of 0.347 when we
trained the models on the improved dataset. This was strange to see, as again, every other
model saw an increase in performance. Apart from the SVR model, each of the other models
saw an increase of about 15-20% in terms of their R-squared values and a decrease in their
mean-squared error of 6-7. In Table 3, we can see the results of these models on the improved
dataset.
<br/><br/>

<p align="center"><img src="https://github.com/jothamteshome/Predicting-NFL-Betting-Odds/assets/94427010/2b16c7a7-7305-4c53-929b-9aefb38afa67"/></p>
<p align="center">Table 3. Model Results on Improved Data w/ Initial Hyperparameters</p>
<br/><br/><br/>

We can see from above that when training on the improved data, the R-squared values
of both the Linear Regression and Ridge Regression are best, and the Linear Regression model
also has the lowest MSE overall, though Ridge Regression comes in as a very close second
place. Seeing this, we believed that with some minor adjustments to the model
hyperparameters, we could achieve even better performance across the board. After some
additional testing, we began training our models using different hyperparameters to see if we
were able to find a better fit for our data. In Table 4 below, we can see the model
hyperparameters that best fit the data after training our models using the improved dataset
described in Section 2.2 above.
<br/><br/>

<p align="center"><img src="https://github.com/jothamteshome/Predicting-NFL-Betting-Odds/assets/94427010/11da9f88-7637-4b5f-b3e1-3ef7e3878813"/></p>
<p align="center">Table 4. Final Model Hyperparameters</p>
<br/><br/><br/>

While many of the hyperparameters that we modified were the same, the values of these
saw many different changes. In fact, tuning each of the models was a time-consuming process,
as since we are using a Voting Regression model, minor changes in one model can affect both
models, so it was necessary to try and maximize the results of the R-squared values and
minimize the mean-squared error for all models simultaneously. With the combination of these
improvements to the model hyperparameters as well as the improvement of the dataset, we
ended up with our final models achieving the best results we’d seen. We can see these results
below in Table 5.
<br/><br/>

<p align="center"><img src="https://github.com/jothamteshome/Predicting-NFL-Betting-Odds/assets/94427010/bfcfea41-dcce-4f53-905a-9604a8a9bd4d"/></p>
<p align="center">Table 5. Model Results on Improved Data w/ Final Hyperparameters</p>
<br/><br/><br/>

Above, we can see when training our models using the final hyperparameters from Table
4, our models can perform much better overall than when we used the initial model
hyperparameters on the same data. Apart from the standard Linear Regression model that did
not see an increase in either metric, every other model had some sort of improvement, whether
it be small, such as in the case of the Ridge Regression model and the Random Forest
Regression model, or very large, such as with the SVR model. As we can see, the R-squared
value of Lasso Regression and Voting Regression increased by about 3%, while their MSE
dropped by about 1-1.2 points. The clearest improvement though is from the SVR model. When
using the initial hyperparameters, it had an R-squared score of under 35%, and it jumped up to
over 72% with the final hyperparameters. This massive improvement is likely due to the kernels
being used in either model. Initially, we had been using the radial basis function kernel, while in
our final model, we used a linear kernel, which likely was able to fit the nature of our data better.
Overall though, we see that the Voting Regression model is our best non-deep learning model,
as it performs the best when looking at either metric, having both the greatest R-squared score
and smallest MSE. This is unsurprising, as it had been performing relatively well regardless of
how the models were being trained. Additionally, it has the advantage of being able to take the
results of each of the different models and take the best values between them to make its
prediction, which is why it was so stable, as well as why it performed the best when each model
began performing well.
The construction of the neural network model was carried out using the PyTorch library
in Python. Both the training and testing phases utilized the original and enhanced datasets,
which encompassed features related to football teams' statistics across various seasons. To
optimize the Mean Squared Error (MSE) loss during training, a specified maximum number of
epochs was set, incorporating early stopping when no significant change in validation loss was
observed. The chosen learning rate was 0.001, and the ADAM optimizer was employed.
Besides creating an accurate predictor, our secondary objective when creating our
neural network model was to ensure that the final iteration of this model met or exceeded the
performance achieved by non-deep learning models. In the process of developing the model
architecture, performance metrics such as MSE and R-squared were employed for comparison
with the best-performing non-deep learning models. We continuously altered the model
architecture, incorporating a greater number of connections to better fit the training data until
there was no longer a significant increase in training performance, and then tuning
generalization such that training and testing performance converged. The ensuing architecture
comprised densely connected layers, with a greater number of neurons in the early layers
gradually reducing in deeper layers, ultimately resulting in a single output for prediction.
Following each fully connected layer, Rectified Linear Unit (ReLU) activation functions
were applied to introduce non-linearity, reflecting the project's focus on comparing linear and
non-linear methods. To promote generalization, dropout layers were strategically inserted after
each hidden layer, randomly deactivating neurons during training. Notably, larger dropout rates
were assigned to earlier layers with more neurons, while smaller rates were implemented for
later layers with fewer neurons. The comprehensive details of the model architecture and
hyperparameters can be found in Table 6.
<br/><br/>

<p align="center"><img src="https://github.com/jothamteshome/Predicting-NFL-Betting-Odds/assets/94427010/a3127cd3-e2bb-4f06-bcc1-3027d2a49a1b"/></p>
<p align="center">Table 6. Neural Network Architecture</p>
<br/><br/><br/>

The performance of this final neural network can be seen in Table 7 below. It can be
observed that the testing performance of this final neural network model exceeds that of the
best-performing non-deep learning model, Voting Regression, and thereby fulfilled our
secondary objective for the neural network.
<br/><br/>

<p align="center"><img src="https://github.com/jothamteshome/Predicting-NFL-Betting-Odds/assets/94427010/c455332a-d18f-4b68-b53a-4b416ccb22a5"/></p>
<p align="center">Table 7. Neural Network Testing Performance on Improved Data Compared to Voting Regression w/ Final Hyperparameters</p>
<br/><br/><br/>


## 3. Results
After creating models that were able to predict the home spread values using the
features of our dataset, we utilized these models to similarly predict the point differentials of the
game. While the spread and point differential are very similar, since the spread is the
sportsbook’s prediction of the point differential, there is often a larger variance in the actual
outcome of games, resulting in a larger variance in the point differential. For example, in NFL
regular season games from 2002 to 2022 (data we collected), the home spread ranged from
-26.5 to +18.5, while the home point differential ranged from -59 to +46. Due to this difference,
we wanted to see how our model would perform when changing the target variable to the point
differential in the game instead of the spread. As the point differential reflects the actual
outcome of the game, having our models predict this metric allows us to evaluate the realistic
application of our model, as well as to compare it to the performance of the sportsbook. We can

see the results of our Voting Regression and Neural Network models when predicting home
point differential compared to the sportsbook’s predictions in Table 8 below.
<br/><br/>

<p align="center"><img src="https://github.com/jothamteshome/Predicting-NFL-Betting-Odds/assets/94427010/86c7947f-2be2-43f5-afee-23a3feb01224"/></p>
<p align="center">Table 8. Model Results for Predicting Home Point Differential</p>
<br/><br/><br/>

As we can see from the table above, of our best models, the Voting Regression model
has the lower mean-squared error, but our Neural Network performs better when it comes to the
R-squared value. However, it is apparent that the sportsbook’s predictions surpass those of
either of our models, with the results for both metrics being significantly better than either of our
models. We predict that this is likely due to the data available to both us and the sportsbooks.
While we were able to collect a good subset of the data for this project, sportsbooks are likely
able to collect more in-depth data, as well as having betting data from many previous years that
would not be available to us. Considering this, we believe that our models have performed well
in comparison, though we know we can improve upon these results by searching for and getting
access to more in-depth NFL data.


## 4. Conclusion
From what was observed when training the models using the original data and the
improved data, the most notable conclusion that could be drawn is that the data and the
features that are available for training are most important when creating a predictive model.
While hyperparameter tuning and model architecture played a significant role in improving our
models’ performance, we found the lack of features in the original dataset limited the accuracy
that we could achieve, and once we added more features to our dataset we saw a significant
increase in the performance of our models.
If we were given more time to continue work on this project, we could look into obtaining
even more features to help increase the robustness of our dataset. Additional features could
include, number of significant players injured on each team, weather conditions, record in home
games versus record in away games, more in-game statistics, or record against similar
opponents. Another thing we could experiment with is actually using our predictions, and
potentially a threshold, to decide whether or not we should be on a spread or not. We could then
see if our model or various thresholds of our model could be profitable.
For our team member contributions, Jotham worked on web scraping, linear model
implementation, and linear model tuning, Karn worked on the neural network implementation
and neural network hyperparameter tuning, and Josh worked on data reading, data
preprocessing, and full dataset compilation.


## References
* [1] Greenberg, D. (2023, September 6). Record 73.5m Americans projected to bet on 2023 NFL season. Front Office Sports. https://frontofficesports.com/americans-betting-2023-nfl-season-football/#:~:text=Record%2073.5%20Million%20Americans%20Projected%20to%20Bet%20On%202023%20NFL%20Season&text=About%2028%25%20of%20all%20adults,on%20the%20NFL%20this%20season.
* [2] NFL, AFL and Pro Football History. Pro. (n.d.). https://www.pro-football-reference.com/years/
* [3] Pempus, B. (2023, November 29). States where sports betting is legal. Forbes. https://www.forbes.com/betting/guide/states-where-sports-betting-is-legal/#:~:text=There%20is%20a%20patchwork%20of,states%2C%2029%20permit%20wagering%20online.
* [4] Spreadspoke. (2023, December 10). NFL scores and Betting Data. Kaggle. https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data




