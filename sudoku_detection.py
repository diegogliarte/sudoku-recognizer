import glob
from time import sleep

import cv2
import numpy as np
from numbers_detection import *
from Backtracking import *
import re


def process_image(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adaptive_treshold = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                              cv2.THRESH_BINARY_INV, 9, 5)
    return adaptive_treshold


def find_max_contourn(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area, max_idx = 0, 0
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_idx = idx

    return contours, max_idx


def get_rectangle_points(image, percentage):
    h = image.shape[0]
    w = image.shape[1]
    side = h * percentage
    h1 = (1 - percentage) / 2
    h2 = h1 + percentage

    p1 = (int((w - side) / 2), int(h * h1))
    p2 = (int(((w - side) / 2) + side), int(h * h2))

    return p1, p2


def get_four_corners(contours, max):
    epsilon = 0.01 * cv2.arcLength(contours[max], True)
    corners = cv2.approxPolyDP(contours[max], epsilon, True)

    return corners


def draw_contours(image, contours, id, color=(0, 0, 255), thickness=5):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.drawContours(image, contours, id, color, thickness)
    return image


def has_four_corners(contours, max):
    if not contours:
        return False

    corners = get_four_corners(contours, max)
    return len(corners) == 4 and cv2.contourArea(contours[max]) > 45000


def crop_grid(image):
    contours, max_idx = find_max_contourn(image)
    corners = get_four_corners(contours, max_idx)

    if (corners[0, 0, 0] < corners[2, 0, 0]):
        src = np.float32([corners[0], corners[3], corners[1], corners[2]])
    else:
        src = np.float32([corners[1], corners[0], corners[2], corners[3]])
    dst = np.float32([[0, 0], [512, 0], [0, 512], [512, 512]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(image, matrix, (512, 512))

    return result


def empty_images(image):
    p1, p2 = get_rectangle_points(image, 0.5)
    crop = image[p1[1]:p2[1], p1[0]:p2[0]]
    np_crop = np.asarray(crop)
    if np.all(np_crop == 0):
        return np.zeros(image.shape)
    else:
        return image


def fix_contours(corners):
    if (corners[0, 0, 0] < corners[2, 0, 0]):
        copy = corners
        corners[0] = copy[0]
        corners[1] = copy[3]
        corners[2] = copy[2]
        corners[3] = copy[1]
    return corners


def crop_numbers(image):
    preds = []
    images = []
    margin = 2
    for i in range(9):
        for j in range(9):
            cropped = image[i * (512 // 9) + margin:(i + 1) * (512 // 9) - margin + 3,
                      j * (512 // 9) + margin + 3:(j + 1) * (512 // 9) - margin]
            img = cv2.resize(cropped, (28, 28))
            mask = np.ones(img.shape[:2], dtype="uint8") * 255
            contours, max_idx = find_max_contourn(img)

            for ctn in contours:
                if np.all(ctn != contours[max_idx]):
                    cv2.drawContours(mask, [ctn], -1, 0, -1)
            img = cv2.bitwise_and(img, img, mask=mask)

            #cv2.imwrite(f"crops/crop{i}{j}_raw.png", img)

            img = empty_images(img)

            #cv2.imwrite(f"crops/crop{i}{j}.png", img)

            images.append(img)

            img_pred = img.reshape(1, 28, 28, 1).astype(np.float32)
            img_pred /= 255
            preds.append(img_pred)

    return preds, images


def draw_translucid(image, p1, p2):
    output = np.zeros_like(image, dtype=np.uint8)
    output[:, :, ] = 200
    output[p1[1]:p2[1], p1[0]:p2[0]] = image[p1[1]:p2[1], p1[0]:p2[0]]
    cv2.addWeighted(image, 0.5, output, 1 - .5, 0, output)
    return output


def remove_artifacts(grids):
    coordinates = np.zeros((81))
    for grid in grids:
        for idx, number in enumerate(grid):
            if (np.all(number == 0)):
                coordinates[idx] += 1

    mask = np.where(coordinates > len(grids) / 2, True, False)
    mask = np.resize(mask, (1, 28, 28, 1))
    result = np.array(grids[-1])
    result = result * mask

    return result


def predict_crops(crops_preds):
    preds = []
    for crop in crops_preds:
        if np.all(crop == 0):
            preds.append([0])
        else:
            preds.append(predict_image(crop))
    return preds


def get_crops(frames):
    crops_preds = []
    for frame in frames:
        grid = crop_grid(frame)
        pred, img = crop_numbers(grid)
        crops_preds.append(pred)
    return crops_preds


def create_focus(contours, focus):
    mask = np.ones(focus.shape[:2], dtype="uint8") * 255
    for ctn in contours:
        if cv2.contourArea(ctn) < 25:
            cv2.drawContours(mask, [ctn], -1, 0, -1)
    focus = cv2.bitwise_and(focus, focus, mask=mask)
    return focus


def modify_sudoku(array, solver):
    answer = input(
        "If you want to change the sudoku, use this syntax: 0, 8 = 3 (This will change the last number of the first row to a 3). Use 0 to solve and -1 to scan again.\n")
    while ((answer != "0" or not solver.check_valid_sudoku()) and answer != "-1"):
        validation = r"^\(?([0-8])\,\s*([0-8])\)?\s*=\s*([0-9])$"
        if match := re.match(validation, answer):
            y = int(match.group(1))
            x = int(match.group(2))
            n = int(match.group(3))
            array[y, x] = n
            solver.set_sudoku(array)
            promt = "Use the syntax '0, 8 = 3' (without the quotes) to change the sudoku once again, 0 to solve and -1 to leave\n"
        elif answer == "0":
            promt = "Your sudoku is invalid! Make sure you inputed it correctly. Remember, use the syntax '0, 8 = 3' (without the quotes) to change the sudoku, 0 to solve and -1 to leave\n"
        else:
            promt = "Invalid input. Remember, use the syntax '0, 8 = 3' (without the quotes) to change the sudoku, 0 to solve and -1 to leave\n"
        print(solver)
        answer = input(promt)

    if answer == "0":
        print("\nSolving Sudoku\n")
        solver.solve()


def video():
    cap = cv2.VideoCapture(0)
    solver = Backtracking()
    frames = []

    while (True):

        ret, frame = cap.read()

        processed = process_image(frame)
        p1, p2 = get_rectangle_points(frame, 0.8)
        output = draw_translucid(processed, p1, p2)

        focus = processed[p1[1]:p2[1], p1[0]:p2[0]]
        contours, max_idx = find_max_contourn(focus)
        focus = create_focus(contours, focus)

        cv2.imshow("output", output)

        if has_four_corners(contours, max_idx):
            focus_drawn = draw_contours(focus, contours, max_idx)
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
            output[p1[1]:p1[1] + focus_drawn.shape[0], p1[0]:p1[0] + focus_drawn.shape[1]] = focus_drawn
            frames.append(focus)
            cv2.imshow("output", output)

            if len(frames) >= 30 and cv2.waitKey(0) & 0xFF == ord("s"):
                crops_preds = get_crops(frames)

                crops_preds = remove_artifacts(crops_preds[:10])

                preds = predict_crops(crops_preds)

                array = np.array(preds).reshape((9, 9))
                solver.set_sudoku(array)
                print(solver)

                cv2.imshow("output", output)
                modify_sudoku(array, solver)

        else:
            frames = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)


if __name__ == "__main__":
    video()
