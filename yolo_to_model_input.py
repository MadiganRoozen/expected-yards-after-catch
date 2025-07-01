import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

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


yolo_data = pd.read_csv("tracking_output_with_qb.csv", usecols = ["playID", "playerID", "team", "frame_num", "x", "y", "position", "speed", "acceleration", "distance", "direction","event"])

#find quarterback on frame 0, most reliable frame
quarterback = yolo_data[(yolo_data["position"] == "qb") & (yolo_data["frame_num"] == 0)]
#find the quarterback ID specifically
qbID = quarterback["playerID"].iloc[0]
print("QBID " + str(qbID))

#random frame number for now, we can change this to generate a separate model input file for every frame if we want to
frame = 100
flag = 0
while(flag != 1):
     qb_in_frame = yolo_data[(yolo_data["playerID"] == qbID) & (yolo_data["frame_num"] == frame)]
     print(qb_in_frame)
     print(str(frame))
     if qb_in_frame.empty:
          frame = frame + 1
     else:  
          flag = 1

#get all data from the same frame
yolo_data = yolo_data[yolo_data["frame_num"] == frame]

#all possible players to throw to, excludes quarterback
teammates = yolo_data[(yolo_data["team"] == "offense") & (yolo_data["playerID"] != qbID)]

#defenders
defenders = yolo_data[yolo_data["team"] == "defense"]

#need to adjust
model_input_columns = ["receiverID", "receiverx", "receivery", "receivers", "receivera", "receiverdis", "receiverdir", "distance_to_nearest_def", "defenderx",
     "defendery", "defenders", "defendera", "defenderdis", "defenderdir", "defenders_in_path","friends_in_path", "pass_length"]
model_input_df = pd.DataFrame(columns = model_input_columns)

for _, target in teammates.iterrows():

    row_data = {}

    receiverx = target["x"]
    receivery = target["y"]
    receiverdir = target["direction"]
    receiverID = target["playerID"]

    #calculate pass length
    qbX = quarterback["x"]
    passLength = qbX - receiverx
    if passLength.iloc[0] < 0:
     passLength = passLength * (-1)

    #170 degree cone of vision for the receiver
    cone_right = (receiverdir + 85)
    cone_left = (receiverdir - 85)

    #equation for the right side of the cone
    right_side_theta = -1 * (cone_right - 90)
    slope_right = math.tan(right_side_theta * math.pi / 180)
          
    #equation for the left side of the cone
    left_side_theta = -1 * (cone_left - 90)
    slope_left = math.tan(left_side_theta * math.pi / 180)

    #print("cone right: " + str(cone_right) + "cone left: " + str(cone_left) + "right theta: " + str(right_side_theta) + " left theta: " + str(left_side_theta))

    #what direction is the player facing? need this for defenders_in_cone
    receiver_dir_category = get_direction_category(left_side_theta, right_side_theta)

    #initialize variables for defenders in cone and closest defender
    row_data["distance_to_nearest_def"] = 100
    row_data["defenders_in_path"] = 0
    closest_defender_entry = defenders.iloc[0]

    for defender in defenders.iloc[0:].values:
        #defender position
        defx = defender[4]
        defy = defender[5]

        #vector between two players
        vector = [receiverx - defx, receivery - defy]

        #euclidean distance between rec and def, check if this is the closest defender
        distance = math.sqrt( vector[0]**2 + vector[1]**2 )

        if distance < row_data["distance_to_nearest_def"]:
          row_data["distance_to_nearest_def"] = distance

          #save the closest defender
          closest_defender_entry = defender

          right_eqn = slope_right * (defx - receiverx) + receivery
          left_eqn = slope_left * (defx - receiverx) + receivery

          #print("slope right: " + str(slope_right) + " slope left: " + str(slope_left) + " right_eqn: " + str(right_eqn) + " left_eqn: " + str(left_eqn) + " defender y: " + str(defy))

          is_player_in_cone = decide_player_in_cone(right_eqn, left_eqn, defy, receiver_dir_category)

          if is_player_in_cone:
               row_data["defenders_in_path"] = row_data["defenders_in_path"] + 1

    row_data["friends_in_path"] = 0

    for friend in teammates.iloc[0:].values:
        #ignore the current receiver
        if friend[1] == receiverID:
            continue
        #defender position
        fx = friend[4]
        fy = friend[5] 

        #vector between two players
        vector = [receiverx - fx, receivery - fy]

        right_eqn = slope_right * (fx - receiverx) + receivery
        left_eqn = slope_left * (fx - receiverx) + receivery

        is_player_in_cone = decide_player_in_cone(right_eqn, left_eqn, defy, receiver_dir_category)

        if is_player_in_cone:
          row_data["friends_in_path"] = row_data["friends_in_path"] + 1

    #columns_to_use = ["gameId", "playId", "nflId", "frameId", "club","playDirection", "x", "y", "s", "a", "dis", "o", "dir", "event"]
    row_data["receiverID"] = receiverID
    row_data["receiverx"] = receiverx
    row_data["receivery"] = receivery
    row_data["receivers"] = target["speed"]
    row_data["receivera"] = target["acceleration"]
    row_data["receiverdis"] = target["distance"]
    row_data["receiverdir"] = receiverdir
    row_data["pass_length"] = passLength.iloc[0]
    #row_data["yards_to_go"] = play_entry["yardsToGo"].iloc[0]
    #row_data["yardline_num"] = play_entry["yardlineNumber"].iloc[0]
    #row_data["yards_gained"] = entry["yardageGainedAfterTheCatch"]
    row_data["defenderx"] = closest_defender_entry[4]
    row_data["defendery"] = closest_defender_entry[5]
    row_data["defenders"] = closest_defender_entry[7]
    row_data["defendera"] = closest_defender_entry[8]
    row_data["defenderdis"] = closest_defender_entry[9]
    row_data["defenderdir"] = closest_defender_entry[10]

    #add entry to dataframe
    model_input_df = pd.concat([model_input_df, pd.DataFrame([row_data])], ignore_index=True)

print(model_input_df.shape)
print(model_input_df.head(3))
model_input_df.to_csv("model_input_from_yolo.csv", index=False)

