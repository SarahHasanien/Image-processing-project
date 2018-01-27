# Image-processing-project
-A real time game with 2 levels written in python using openCV. 
-The user can play it by opening his webcam. 
-Game sequence:
  1) User is given the option to choose between 2 games.
   He selects the game by swapping his hand right to choose the first game or left to choose the second game.
   Five seconds are given to the user to be ready for moving his then we start taking the data.
  2) If he moved his hand right the first game is loaded.
  The second game generates random numbers.
  displays 2 numbers of them on the screen.
  ask the user to raise his hand with the true answer by enetring 2 digits with his hand(the most digit then the least digit).
  between taking the 2 digits the user is given 8 seconds to be ready then we take the answer.
  if he answered true , score will increase by 1 else the lives will be decreased by one.
  when the lives are 0 the game terminates and a game over screen appears.
  3) If he moved his hand right the second game is loaded.
  An image contains random different shapes appears.
  the user is asked to press any key to start playing.
  the game detects the types of shapes in the image and count their number.
  we ask the user about the number of each shape.
   if it is equivalent to the number calculated by our program then we print true else we print false.
   we give the user 8 seconds to be ready before taking the answer. 
  the second game terminates when the user answer the 8 questions:
  (How many squares are in the image?)
  (How many rectangles are in the image?)
  (How many triangles are in the image?)
  (How many circles are in the image?)
  (How many rhombuses are in the image?)
  (How many hexagons are in the image?)
  (How many pentagons are in the image?)
  (How many ellipses are in the image?)
-Note that at any point you can terminate the game by pressing ‘c’.
