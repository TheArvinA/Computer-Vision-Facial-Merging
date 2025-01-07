import gradio as gr
import cv2
import numpy as np
import dlib
import imutils

from scipy.spatial import Delaunay
from matplotlib import pyplot as plt


def get_triangles(points):
    return Delaunay(points).simplices


def affine_transform(input_image, input_triangle, output_triangle, size):
    warp_matrix = cv2.getAffineTransform(
        np.float32(input_triangle), np.float32(output_triangle)
    )
    output_image = cv2.warpAffine(
        input_image,
        warp_matrix,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return output_image


def face_detect(img):
    image = imutils.resize(img, width=512)

    # Detect face within image
    face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    # To draw a rectangle in a face
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    return image


def image_points(img):
    # Detect features within image
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    try:
        detected_face = detector(img, 1)[0]
    except:
        print("No face detected in image {}".format(img))

    landmarks = predictor(img, detected_face)
    points = []

    for p in landmarks.parts():
        points.append([p.x, p.y])

    # Add 8 image frame coordinate points
    x = img.shape[1] - 1
    y = img.shape[0] - 1
    points.append([0, 0])
    points.append([x // 2, 0])
    points.append([x, 0])
    points.append([x, y // 2])
    points.append([x, y])
    points.append([x // 2, y])
    points.append([0, y])
    points.append([0, y // 2])

    return np.array(points)


def feature_detect(img):
    image = imutils.resize(img, width=512)
    img_points = image_points(image)

    for idx, point in enumerate(img_points):
        pos = (point[0], point[1])
        cv2.putText(
            image,
            str(idx),
            pos,
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            fontScale=0.1,
            color=(0, 255, 0),
        )

    return image


def triangle_fig(img):
    image = imutils.resize(img, width=512)
    points = image_points(image)
    triangles = get_triangles(points)

    fig, ax = plt.subplots()
    ax.triplot(points[:, 0], points[:, 1], triangles, linewidth=1, color="b")
    ax.imshow(image)

    return fig


def feature_detector_triangle(img):
    feature_img = feature_detect(img)
    triangle_img = triangle_fig(img)

    return feature_img, triangle_img


def merge_faces(img1, img2):
    points1 = image_points(img1)
    points2 = image_points(img2)

    # Calculate the average coordinates of points in two images
    alpha = 0.5
    points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)
    triangles = get_triangles(points)

    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)

    for i in triangles:

        # Calculate the frame of triangles
        x = i[0]
        y = i[1]
        z = i[2]

        tri1 = [points1[x], points1[y], points1[z]]
        tri2 = [points2[x], points2[y], points2[z]]
        tri = [points[x], points[y], points[z]]

        rect1 = cv2.boundingRect(np.float32([tri1]))
        rect2 = cv2.boundingRect(np.float32([tri2]))
        rect = cv2.boundingRect(np.float32([tri]))

        tri_rect1 = []
        tri_rect2 = []
        tri_rect_warped = []

        for i in range(0, 3):
            tri_rect_warped.append(((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
            tri_rect1.append(((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
            tri_rect2.append(((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))

        # Accomplish the affine transform in triangles
        img1_rect = img1[rect1[1] : rect1[1] + rect1[3], rect1[0] : rect1[0] + rect1[2]]
        img2_rect = img2[rect2[1] : rect2[1] + rect2[3], rect2[0] : rect2[0] + rect2[2]]

        size = (rect[2], rect[3])
        warped_img1 = affine_transform(img1_rect, tri_rect1, tri_rect_warped, size)
        warped_img2 = affine_transform(img2_rect, tri_rect2, tri_rect_warped, size)

        # Calculate the result based on alpha
        img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

        # Generate the mask
        mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

        # Accomplish the mask in the merged image
        img_morphed[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]] = (
            img_morphed[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
            * (1 - mask)
            + img_rect * mask
        )

    return img_morphed


face_detector = gr.Interface(
    fn=face_detect,
    inputs=gr.Image(),
    outputs=["image"],
    description="This aims to detect faces.",
)

feature_detector = gr.Interface(
    fn=feature_detector_triangle,
    inputs=gr.Image(),
    outputs=["image", gr.Plot()],
    description="This aims show features within a facial analysis.",
)

face_merger = gr.Interface(
    fn=merge_faces,
    inputs=[gr.Image(), gr.Image()],
    outputs=["image"],
    description="This aims to employ facial detection and feature analysis to merge two faces.",
)

demo = gr.TabbedInterface(
    [face_merger, face_detector, feature_detector],
    ["Face Merger", "Face Detector", "Feature Detector"],
    title="CPS843: Face Merge",
)

if __name__ == "__main__":
    demo.launch()
