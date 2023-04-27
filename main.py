#Import required libraries
import cv2
import numpy as np

#Initialize variables
score = [0, 0]
fps = 0
video = cv2.VideoCapture("pedra-papel-tesoura.mp4")

#Converts the input image from BGR to HSV
def convert_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Creates masks for hand colors (pink and yellow) and merges them
def find_hand_colors(hsv):
    pink_hsv_mask = cv2.inRange(hsv, np.array([0, 20, 0]), np.array([255, 255, 255]))
    yellow_hsv_mask = cv2.inRange(hsv, np.array([0, 30, 0]), np.array([255, 255, 255]))
    return cv2.bitwise_or(yellow_hsv_mask, pink_hsv_mask)

#Finds contours of the hands based on their color masks
def find_contours(hsv):
    contours, _ = cv2.findContours(find_hand_colors(hsv), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse= True)

#Calculates the area of the input contour
def find_countour_area(contour):
    return cv2.contourArea(contour)

#Draws contours around the hands on the input image
def draw_contour(rgb, contours):
    contours_img = rgb.copy()
    return cv2.drawContours(contours_img, [contours[0], contours[1]], -1, (255,255,255), 0)

#Identifies the move (Rock, Paper, or Scissors) based on the hand contour area
def ident_move(area):
    if(50000 < area < 52000):
        return 'Pedra'
    if(63000  < area < 64000):
        return 'Papel'
    if(48000 < area < 49500):
        return 'Tesoura'


#Calculates the center of mass of an object (contour)
def object_center(contour):
    return int(cv2.moments(contour)['m10']/cv2.moments(contour)['m00']) + int(cv2.moments(contour)['m01']/cv2.moments(contour)['m00'])

#Identifies player positions based on the center of mass
def ident_players(contours):
    if (object_center(contours[0]) < object_center(contours[1])):
        player1 = contours[0]
        player2 = contours[1]
    else:
        player1 = contours[1]
        player2 = contours[0]        
    return player1, player2

#Determines the winner based on player's moves
def ident_winner(move1, move2):
    if(move1 == move2):
        return 'Empate'
    elif(move1 == 'Pedra'):
        if move2 == "Tesoura":
            return 'Jogador 1 ganhou!'
        else:
            return 'Jogador 2 ganhou!'
    elif(move1 == 'Papel'):
        if move2 == "Pedra":
            return 'Jogador 1 ganhou!'
        else:
            return 'Jogador 2 ganhou!'
    elif(move1 == 'Tesoura'):
        if move2 == "Papel":
            return 'Jogador 1 ganhou!'
        else:
            return 'Jogador 2 ganhou!'

#  if(fps % 90 == 0):: This condition checks if the current frame number (fps)
#     is divisible by 85 without any remainder. Since the default frame rate for most
#     videos is around 30 frames per second (fps), this condition will be true every
#     3 seconds (90 / 30). This is used to regulate the rate at which the score updates
#     to avoid counting multiple times for a single move.   
  
#Updates the score every 3 seconds based on the game result 
def calculate_score(result):
    if(fps % 90 == 0):  
        if(result == 'Jogador 1 ganhou!'):
            score[0] += 1
        if(result == 'Jogador 2 ganhou!'):
            score[1] += 1

#Check if the video is opened correctly and read the first frame.
#For each frame, perform the following steps:
if video.isOpened(): 
    rval, frame = video.read()
else:
    rval = False
while rval:
    fps += 1
    #Convert the frame to HSV color space.
    hsv = convert_to_hsv(frame)

    #Find and draw hand contours.
    contours = find_contours(hsv)
    draw_contours = draw_contour(frame, contours)

    #Identify player positions.
    player1, player2 = ident_players(contours)

    #Calculate hand contour areas
    area1 = find_countour_area(player1)
    area2 = find_countour_area(player2)

    #identify moves
    move1 = ident_move(area1)
    move2 = ident_move(area2)

    #Determine the winner
    result = ident_winner(move1,move2)
    #Update the score
    calculate_score(result)

    #Add text overlays to display moves, scores, and the result
    cv2.putText(draw_contours, 'Player 1 jogou ' + move1 +' e '+ 'Player 2 jogou ' + move2 , (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
    cv2.putText(draw_contours, str(score) , (900, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
    cv2.putText(draw_contours, result , (800, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    #Show the processed frame
    cv2.imshow("checkpoint", draw_contours)

    #Read the next frame and wait for a key press
    rval, frame = video.read()
    key = cv2.waitKey(20)

    #Exit the loop if the 'Esc' key is pressed
    if key == 27:
        break

#Close the video display window and release the video capture object
cv2.destroyWindow()
video.release()