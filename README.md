# Sudoku Recognizer
**Scan a sudoku** with your camera and **solve** it with the press of a buttom.

Program in action:

![Gif with program in action](https://media.giphy.com/media/ULL310i8Ktvk1oTSYQ/giphy.gif)

Output on console:

<img atl="output from program" src="https://i.imgur.com/fMYgJ5U.png" width="25%" height="25%"> 



## Objectives
Create a program which uses a camera and is able to recognize and solve a Sudoku.

## How to use
Point your camera towards a sudoku. There will be a **red outline** indicating that the sudoku can be recognized. After 30 frames of uninterruptly detection, the image will freeze.
**Press "S"** to go on with the current frame, or anything else to continue scanning. Then a **input promt** will be opened explaining what to do. If you want to exit the program, you
can exit by pressing "Q".

## Accomplishments
Recognizes **big squares** inside certain area and is able to crop the numbers individually. This detection uses exclusively OpenCV.

Manages to predict the numbers **90%** of the time.

You can **modify the sudoku** and make up for the 10% of the times it fails.

Given a valid Sudoku, it uses **Backtraking algorithm**. I didn't use any reference for my implementation.

## How it's done
Uses **OpenCV** to everything related to the sudoku recognition, the grid and the cropped numbers. A lot of **image processing** such as adaptative threshold or noise
reduction has been implemented
as well as cusotm techniques used to clean even more the image and get only the desired number, removing artifacts from the image.

Once the numbers are cleared, **keras** is used to load a **pre-trained model for digit recognition** is used to predict all the numbers.

## Problems
It does not detect only Sudokus, but **any kind of rectangular shape**. If you use a rectangular shape as the input, you will get impredictable behaviour.

Everything **looks ugly** and the program is not intuitive to use.

The **modification process of the sudoku** after the predictions have been made is **poorly made** and uses console commands.

Having to modify **individual** numbers is a pain in the ass.

## Possible improvements
**Train** the prediction model with my samples. Since this model have been probably made with handrwritting images instead of computer-font-digit images, this would **greatly
improve** the prediction.

## Conclusion
Although **usable**, it's not a finished product (especially in its **user-friendliness**). All the things I wanted to do are done, mainly using OpenCV instead of PIL and
integrating a prediction model. 




