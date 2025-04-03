import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def find_tracking_df_index(gameId):
    num = gameId - 2022000000
    if num<=91200 :
         return 0
    elif num>=91500 and num<=91901:
         return 1
    elif num>=92200 and num<=92600:
         return 2
    elif num>=92900 and num<=100300:
         return 3
    elif num>=100600 and num<=101000:
         return 4
    elif num>=101300 and num<=101700:
         return 5
    elif num>=102000 and num<=102400:
         return 6
    elif num>=102700 and num<=103100:
         return 7
    elif num>=110300 and num<=110700:
         return 8

def get_direction_category(left_angle, right_angle):
     if left_angle < 0: 
          left_angle = left_angle + 360
     if right_angle < 0:
          right_angle = right_angle + 360

     left_above = True
     left_below = True
     right_above = True
     right_below = True

     if left_angle > 85 and left_angle < 265:
          left_above = True
          left_below = False
     elif left_angle <= 85 or left_angle >= 265:
          left_above = False
          left_below = True
     else:
          print("problem with left angle: " + str(left_angle))
     
     if right_angle <= 95 or right_angle >= 275:
          right_above = True
          right_below = False
     elif right_angle > 95 and right_angle < 275:
          right_above = False
          right_below = True
     else:
          print("problem with right angle: " + str(right_angle))
     
     if left_above and right_above:
          return 1
     elif left_below and right_above:
          return 2
     elif left_below and right_below:
          return 3
     elif left_above and right_below:
          return 4
     else:
          print("problem with angles")
          return -1

def decide_player_in_cone(right_eqn, left_eqn, defy, receiver_dir_category):

     if receiver_dir_category == 1:
          if defy >= right_eqn and defy >= left_eqn:
               return True
          else:
               return False
     
     elif receiver_dir_category == 2:
          if defy >= right_eqn and defy <= left_eqn:
               return True
          else:
               return False

     elif receiver_dir_category == 3:
          if defy <= right_eqn and defy <= left_eqn:
               return True
          else:
               return False

     elif receiver_dir_categoyr == 4:
          if defy <= right_eqn and defy >= left_eqn:
               return True
          else:
               return False


tracking_csvs = ["tracking_week_1.csv", "tracking_week_2.csv", "tracking_week_3.csv", "tracking_week_4.csv", "tracking_week_5.csv", "tracking_week_6.csv", "tracking_week_7.csv", "tracking_week_8.csv", "tracking_week_9.csv"]
columns_to_use = ["gameId", "playId", "nflId", "frameId", "club","playDirection", "x", "y", "s", "a", "dis", "o", "dir", "event"]
#tracking dfs only contains the tracking data from the frame the pass was received on. It contains this data for every player on the field
tracking_dfs = []
for file in tracking_csvs:
    df = pd.read_csv(file, usecols=columns_to_use)
    df = df[df["event"] == "pass_arrived"]
    tracking_dfs.append(df)

#(18791, 13) (18538, 13) (20125, 13) (17227, 13) (18607, 13) (16445, 13) (16330, 13) (16652, 13) (13984, 13)
for df in tracking_dfs:
    print(df.shape)

player_play_data = pd.read_csv("player_play.csv", usecols = ["gameId","playId","nflId","passingYards","hadPassReception","receivingYards","wasTargettedReceiver","yardageGainedAfterTheCatch"])
plays_data = pd.read_csv("plays.csv", usecols = ["gameId","playId","down","yardsToGo","yardlineNumber","passLength","targetX","targetY","yardsGained"])

#need to filter player_play file to find only plays that hadPassreception
player_plays_with_passes = player_play_data[player_play_data["hadPassReception"] == 1]
#5625 records found
player_plays_with_passes = player_plays_with_passes.reset_index(drop=True)
print(player_plays_with_passes.shape)
print(player_plays_with_passes.head(3))

#get the nflID of the player who had a pass reception from player_play entry
receiving_players = player_plays_with_passes[["nflId"]]
print(receiving_players.head(3))

model_input_columns = ["receiverx", "receivery", "receivers", "receivera", "receiverdis", "receivero", "receiverdir", "distance_to_nearest_def", 
    "defenders_in_path","pass_length", "yards_to_go", "yardline_num", "yards_gained"]
model_input_df = pd.DataFrame(columns = model_input_columns)

#find the position of the receiver and all other players around them on that frame (use nflID for the receiver and get all other player positional data and somehow determine
#   who was closest and what their position was in the game)
for _, entry in player_plays_with_passes.iterrows():
    row_data = {}

    #Id info for the game and play
    playerId = entry["nflId"]
    gameId = entry["gameId"]
    playId = entry["playId"]
    #which tracking file is this game in
    tracking_df_index = find_tracking_df_index(int(gameId))

    print("found receiverID: " + str(playerId) + "for game and play: " + str(gameId) + " " + str(playId))

    #the tracking data file we need
    tracking_data = tracking_dfs[tracking_df_index]

    #the play entry containing yards to go, yardline num, and yards gained for this play
    #usecols = ["gameId","playId","down","yardsToGo","yardlineNumber","targetX","targetY","yardsGained"]
    play_entry = plays_data[(plays_data["gameId"] == gameId) & (plays_data["playId"] == playId)]

    #filter to get only entries with the necessary gameId, playId, and event
    filtered_tracking_data = tracking_data[(tracking_data["gameId"] == gameId) & (tracking_data["playId"] == playId) & (tracking_data["event"] == "pass_arrived")]
    #get the receiver's data
    #skip if it's empty, that happens sometimes 
    receiver_data = filtered_tracking_data[filtered_tracking_data["nflId"] == playerId]
    if receiver_data.empty:
          continue
    print("RECEIVER DATA: " + str(receiver_data.shape))

    #get the entries for all defenders from this frame. This is any player that is not on the same team (club) as the receiver
    defender_data = filtered_tracking_data[ (filtered_tracking_data["club"] != receiver_data["club"].values[0])  & (filtered_tracking_data["club"] != "football")]

    print(defender_data.shape)
    print(defender_data)

    receiverx = receiver_data["x"].iloc[0]
    receivery = receiver_data["y"].iloc[0]
    receiverdir = receiver_data["dir"].iloc[0]

    #170 degree cone of vision for the receiver
    cone_right = (receiverdir + 85)
    cone_left = (receiverdir - 85)

    #equation for the right side of the cone
    right_side_theta = -1 * (cone_right - 90)
    slope_right = math.tan(right_side_theta * math.pi / 180)
          
    #equation for the left side of the cone
    left_side_theta = -1 * (cone_left - 90)
    slope_left = math.tan(left_side_theta * math.pi / 180)

    print("cone right: " + str(cone_right) + "cone left: " + str(cone_left) + "right theta: " + str(right_side_theta) + " left theta: " + str(left_side_theta))

    #what direction is the player facing? need this for defenders_in_cone
    receiver_dir_category = get_direction_category(left_side_theta, right_side_theta)

    print("game: " + str(gameId) + " play: " + str(playId) + " receiverId: " + str(playerId) + " tracking data index: " + str(tracking_df_index) + 
    " receiver direction: " + str(receiverdir) + " receiver dir cat: " + str(receiver_dir_category))

    #calculated distance cannot be more than 1000
    row_data["distance_to_nearest_def"] = 1000

    row_data["defenders_in_path"] = 0

    for defender in defender_data.iloc[0:].values:
          #defender position
          defx = defender[6]
          defy = defender[7] 

          #vector between two players
          vector = [receiverx - defx, receivery - defy]

          #euclidean distance between rec and def, check if this is the closest defender
          distance = math.sqrt( vector[0]**2 + vector[1]**2 )
          row_data["distance_to_nearest_def"] = min(row_data["distance_to_nearest_def"], distance)

          right_eqn = slope_right * (defx - receiverx) + receivery
          left_eqn = slope_left * (defx - receiverx) + receivery

          print("slope right: " + str(slope_right) + " slope left: " + str(slope_left) + " right_eqn: " + str(right_eqn) + " left_eqn: " + str(left_eqn) + " defender y: " + str(defy))

          is_player_in_cone = decide_player_in_cone(right_eqn, left_eqn, defy, receiver_dir_category)

          if is_player_in_cone:
               print(str(defender[2]) + " defender is in cone")
               row_data["defenders_in_path"] = row_data["defenders_in_path"] + 1


    #columns_to_use = ["gameId", "playId", "nflId", "frameId", "club","playDirection", "x", "y", "s", "a", "dis", "o", "dir", "event"]
    row_data["receiverx"] = receiverx
    row_data["receivery"] = receivery
    row_data["receivers"] = receiver_data["s"].iloc[0]
    row_data["receivera"] = receiver_data["a"].iloc[0]
    row_data["receiverdis"] = receiver_data["dis"].iloc[0]
    row_data["receivero"] = receiver_data["o"].iloc[0]
    row_data["receiverdir"] = receiverdir
    row_data["pass_length"] = play_entry["passLength"].iloc[0]
    row_data["yards_to_go"] = play_entry["yardsToGo"].iloc[0]
    row_data["yardline_num"] = play_entry["yardlineNumber"].iloc[0]
    row_data["yards_gained"] = entry["yardageGainedAfterTheCatch"]

    #add entry to dataframe
    model_input_df = pd.concat([model_input_df, pd.DataFrame([row_data])], ignore_index=True)

print(player_plays_with_passes.shape)
print(model_input_df.shape)
print(model_input_df.head(3))
model_input_df.to_csv("model_input_df.csv", index=False)