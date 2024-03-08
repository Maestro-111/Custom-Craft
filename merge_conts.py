import numpy as np
import cv2
import os
import pytesseract
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
from math import factorial
from sklearn.metrics import silhouette_score
import random
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re

def get_mean_dist(clustering_algo, data):
    cluster_coords = defaultdict(list)
    cluster_mean_dist = {}

    N = len(clustering_algo.labels_)

    for i in range(N):
        label = clustering_algo.labels_[i]
        coords = data[i]
        cluster_coords[label].append(coords)

    for cluster in cluster_coords:
        coords = cluster_coords[cluster]
        cluster_mean_dist[cluster] = mean_distance_between_rectangles(coords)

    return cluster_mean_dist


def apply_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def enhance_sharpness(img, factor=1):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img


def apply_clahe(img,clipLimit):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(10, 10))
    final_img = clahe.apply(img)
    rgb_image = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
    return rgb_image

def apply_erosion(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    result = cv2.erode(image, kernel, iterations=1)
    return result


def apply_opening(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return result

def apply_blackhat(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    result = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return result


def roi_process(roi):
    if len(roi.shape) == 2:  # Check if the image is already grayscale
        gray = roi
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    cl = gray
    cl = cv2.threshold(cl, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cl = cv2.normalize(cl, cl, 0, 1.0, cv2.NORM_MINMAX)
    cl = (cl * 255).astype("uint8")
    cl = cv2.threshold(cl, 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    #cl = cv2.morphologyEx(cl, cv2.MORPH_OPEN, kernel)

    cnts, hierarchy = cv2.findContours(cl.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        chars.append(c)

    chars = np.vstack([chars[i] for i in range(0, len(chars))])
    hull = cv2.convexHull(chars)

    mask = np.zeros(roi.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.dilate(mask, None, iterations=2)

    cl = cv2.bitwise_and(cl, cl, mask=mask)

    return cl 


def rectangle_to_contour(x1, y1, x2, y2, x3, y3, x4, y4):
    rect_pts = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    rect_pts = rect_pts.reshape((-1, 1, 2))
    return rect_pts


def is_contour_inside(contour1, contour2,eps=10): # check if contour1 in contour2; eps - allow small deviation
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)

    return x2 <= x1+eps and y2 <= y1 + eps and x2 + w2 + eps >= x1 + w1 and y2 + h2 + eps >= y1 + h1

def remove_contours_inside(contours):
    filtered_contours = []

    # Iterate through each contour
    for i in range(len(contours)):
        is_inside = False

        # Compare with other contours
        for j in range(len(contours)):
            if i != j and is_contour_inside(contours[i], contours[j]):
                is_inside = True
                break

        # If the contour is not inside any other, add it to the result
        if not is_inside:
            filtered_contours.append(contours[i])

    return filtered_contours

def centroids_distance(cx1,cy1,cx2,cy2):
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

def calculate_rectangle_centroid(x1, y1, x2, y2,local_transform=False):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if local_transform:
        return np.array([cx, cy]).reshape(1,-1)
    else:
        return cx, cy

def contour_distance(contour1, contour2):
    M1 = cv2.moments(contour1)
    M2 = cv2.moments(contour2)

    if M1['m00'] == 0 or M2['m00'] == 0:
        return float('inf')  # Avoid division by zero

    cx1 = int(M1['m10'] / M1['m00'])
    cy1 = int(M1['m01'] / M1['m00'])

    cx2 = int(M2['m10'] / M2['m00'])
    cy2 = int(M2['m01'] / M2['m00'])

    distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    return distance

def merge_close_contours_2(contours, kmeans,cluster_mean_dist,standard_scaler):

    '''
    not using currently
    '''

    merged_contours = []
    seen = set()

    for i in range(len(contours)):

        x_left,y_left,w,h = cv2.boundingRect(contours[i])
        x_right,y_right = x_left+w,y_left+h

        if (x_left,y_left,x_right,y_right) in seen:
            continue

        current_centroid_1 = calculate_rectangle_centroid(x_left, y_left, x_right, y_right,local_transform=True)
        scaled_current_centroid_1 = standard_scaler.transform(current_centroid_1)
        centr_x_1, centr_y_1 = scaled_current_centroid_1[0]

        cluster_label  = kmeans.predict(np.array([[centr_x_1, centr_y_1]]))[0]
        current_dist = cluster_mean_dist[cluster_label]

        close_contours = [contours[i]]
        merged = False

        for j in range(i + 1, len(contours)):

            dx1, dy1,w,h = cv2.boundingRect(contours[i])
            dx3, dy3 = dx1+w,dy1+h
            current_centroid_2 = calculate_rectangle_centroid(dx1, dy1, dx3, dy3,local_transform=True)

            scaled_current_centroid_2 = standard_scaler.transform(current_centroid_2)
            centr_x_2, centr_y_2 = scaled_current_centroid_2[0]

            d = centroids_distance(centr_x_1, centr_y_1, centr_x_2, centr_y_2)

            if d < current_dist:
                close_contours.append(contours[j])
                merged = True

        if merged:
            list_of_pts = []
            for ctr in close_contours:
                list_of_pts += [pt[0] for pt in ctr]

            ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
            ctr = cv2.convexHull(ctr)
            merged_contours.append(ctr)
        else:
            merged_contours.append(contours[i])

        tups = []
        for cnt in close_contours:
            x_left, y_left, w, h = cv2.boundingRect(cnt)
            x_right, y_right = x_left + w, y_left + h
            tups.append((x_left,y_left,x_right,y_right))

        seen.update(tups)

    return merged_contours


def merge_close_contours_3(points,centrs,cluster_mean_dist):

    merged_contours = []
    seen = set()

    for i in range(len(points)):
        x1, y1, x2, y2, x3, y3, x4, y4 = points[i]

        centr_x_1, centr_y_1,label = centrs[i]

        if (x1, y1, x2, y2, x3, y3, x4, y4) in seen:
            continue

        current_threshold = cluster_mean_dist[label]

        close_contours = [points[i]]
        merged = False

        for j in range(i+1, len(points)):
            dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4 = points[j]
            centr_x_2, centr_y_2, label = centrs[j]

            d = centroids_distance(centr_x_1,centr_y_1,centr_x_2,centr_y_2)

            if d < current_threshold:
                close_contours.append(points[j])
                merged = True

        if merged:
            ctr = np.array(close_contours).reshape((-1, 1, 2)).astype(np.int32)
            ctr = cv2.convexHull(ctr)
            merged_contours.append(ctr)
        else:
            merged_contours.append(rectangle_to_contour(x1, y1, x2, y2, x3, y3, x4, y4))

        tups = []
        for point in close_contours:
            x1, y1, x2, y2, x3, y3, x4, y4 = point
            tups.append((x1, y1, x2, y2, x3, y3, x4, y4))
        seen.update(tups)

    return merged_contours




def read_coords(txt_file):

    f = open(txt_file, "r")

    points = []

    for line in f:
        line = line.replace('\n', '')
        if line:
            line = list(map(int,line.split(',')))
            points.append(line)

    return points

def distance_between_contours(contour1, contour2):
    # Calculate the distance between closest points of two contours
    min_dist = float('inf')
    for pt1 in contour1:
        for pt2 in contour2:
            dist = np.linalg.norm(pt1 - pt2)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def merge_rectangles(rect1, rect2): # rect1 = x1,y1,x2,y2,x3,y3,x4,y4

    if not rect1:
        return rect2

    x1_min = min(rect1[0], rect2[0])
    y1_min = min(rect1[1], rect2[1])
    x1_max = max(rect1[4], rect2[4])
    y1_max = max(rect1[5], rect2[5])

    w,h = x1_max-x1_min,y1_max-y1_min

    return [x1_min, y1_min,x1_min+w,y1_min,x1_max, y1_max,x1_min,y1_min+h]

def merge_close_contours(contours, threshold):

    merged_contours = []
    merged_indices = set()

    for i, contour1 in enumerate(contours):

        if i in merged_indices:
            continue

        x1, y1, w, h = cv2.boundingRect(contour1)
        x2, y2 = x1 + w, y1
        x3, y3 = x1 + w, y1 + h
        x4, y4 = x1, y1 + h


        current = [x1,y1,x2,y2,x3,y3,x4,y4]

        for j, contour2 in enumerate(contours):

            if i != j:
                dist = distance_between_contours(contour1, contour2)
                if dist < threshold:
                    dx1, dy1, dw, dh = cv2.boundingRect(contour2)
                    dx2, dy2 = dx1 + dw, dy1
                    dx3, dy3 = dx1 + dw, dy1 + dh
                    dx4, dy4 = dx1, dy1 + dh

                    current = merge_rectangles(current, [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4])
                    merged_indices.add(j)

        current = np.array(current, dtype=np.int32).reshape((-1, 1, 2))
        merged_contours.append(current)


    return merged_contours



def get_centroids_from_points(points,edge=10):

    centroids = []

    for point in points:

        x1, y1, x2, y2, x3, y3, x4, y4 = point

        new_centroid = calculate_rectangle_centroid(x1, y1, x3, y3)
        centroids.append(new_centroid)

    #plt.scatter([cnt[0]for cnt in centroids],[cnt[1]for cnt in centroids])
    #plt.show()

    new_centroids = np.array(centroids)
    standard_scaler = StandardScaler()
    normalized_new_centroids_z_score = standard_scaler.fit_transform(centroids)

    metric = []

    for cluster_num in range(4, edge):

        kmeans = BisectingKMeans(n_clusters=cluster_num, random_state=0, n_init=20, algorithm='lloyd').fit(normalized_new_centroids_z_score)


        if len(set(kmeans.labels_)) < 2:
            metric.append((cluster_num, -1))
            continue

        metric.append((cluster_num, silhouette_score(centroids, kmeans.labels_)))

    #plt.plot(range(4, edge), [i[1] for i in metric])
    #plt.xlabel("Cluster N")
    #plt.ylabel("Silhouette score for N")
    #plt.show()

    optimal_k = max(metric, key=lambda x: x[1])[0]

    kmeans = BisectingKMeans(n_clusters=optimal_k, random_state=0, n_init=20, algorithm='lloyd').fit(normalized_new_centroids_z_score)
    cluster_mean_dist = get_mean_dist(kmeans, normalized_new_centroids_z_score)
    centroids_with_labels_column = np.concatenate((normalized_new_centroids_z_score, np.array(kmeans.labels_).reshape(-1, 1)), axis=1)


    return centroids_with_labels_column,cluster_mean_dist



def get_boxes(merged,copy,img,lst):
    count = 0

    for cnt in merged:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_c = img[y:y + h, x:x + w]


        new_width = 4 * roi_c.shape[1]  # Increase width by a factor of 2
        new_height = 4 * roi_c.shape[0]
        resized_roi = cv2.resize(roi_c, (new_width, new_height))
        lst.append(resized_roi)

        image_filename = f"roi_{count}_.png"  # You can customize the filename as needed
        image_path = os.path.join('contours', image_filename)
        cv2.imwrite(image_path, roi_c)
        count+= 1

    plt.imshow(copy)
    plt.show()

    for file in os.listdir('contours'):
        file_path = os.path.join('contours', file)
        os.remove(file_path)

def save_cnts(folder,img,img_path,cnts,count=0):

    img_path = img_path.split('\\')
    img_path = img_path[-1]

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        roi_c = img[y:y + h, x:x + w]
        new_width = 4 * roi_c.shape[1]  # Increase width by a factor of 2
        new_height = 4 * roi_c.shape[0]

        image_filename = f"roi_{count}_{img_path}"  # You can customize the filename as needed
        image_path = os.path.join(f'{folder}', image_filename)
        cv2.imwrite(image_path, roi_c)
        count += 1






def merge_intersecting_rectangles(rectangles, n_iterations):

    def intersects(rect1, rect2): # rect2 inters rect1

        def intersect(p_left, p_right, q_left, q_right):
            return min(p_right, q_right) > max(p_left, q_left)

        x1, y1, x2, y2, x3, y3, x4, y4 = rect1
        dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4 = rect2

        if y1<=dy1 <= y3 and (x1<=dx1<=x2 or x1<=dx2<=x2):
            return True
        if y1<=dy3 <= y3  and (x1<=dx1<=x2 or x1<=dx2<=x2):
            return True

        return False

    merged_rectangles = rectangles

    for _ in range(n_iterations):
        processed_indices = set()
        current_merged = []
        for i, rect1 in enumerate(merged_rectangles):
            if i in processed_indices:
                continue
            merged_rect = [rect1]
            for j, rect2 in enumerate(merged_rectangles):

                if i == j:
                    continue


                if intersects(rect1, rect2):
                    merged_rect.append(rect2)
                    processed_indices.add(j)

            figure = []

            for rec in merged_rect:
                figure = merge_rectangles(figure,rec)
            current_merged.append(figure)
        merged_rectangles = current_merged

    return merged_rectangles


def rectangles_to_contours(rectangles):
    contours = []
    for rect in rectangles:
        x1, y1, x2, y2, x3, y3, x4, y4 = rect
        contour = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        contours.append(contour)
    return contours


def custom_cnt(points_list):
    contours = []
    for points in points_list:
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        contours.append(contour)
    return contours


def vis_cnts(copy,merged_rectangles):
    for contour in merged_rectangles:
        x1, y1, w, h = cv2.boundingRect(contour)
        x3, y3 = x1+w, y1+h
        cv2.rectangle(copy, (x1, y1), (x3, y3), (0, 255, 0), 2)

    plt.imshow(copy)
    plt.show()


def merge_existing_boxes(path,points):
    img = cv2.imread(path)

    copy1 = img.copy()
    copy2 = img.copy()
    copy3 = img.copy()
    copy4 = img.copy()

    merged_rectangles = merge_intersecting_rectangles(points, 5) # lst of points
    merged_rectangles = custom_cnt(merged_rectangles) # convert to cnts

    vis_cnts(copy1,merged_rectangles)

    merged_rectangles = merge_close_contours(merged_rectangles,100)
    merged_rectangles = remove_contours_inside(merged_rectangles)

    vis_cnts(copy2, merged_rectangles)

    merged_rectangles = merge_close_contours(merged_rectangles,100)
    merged_rectangles = remove_contours_inside(merged_rectangles)

    vis_cnts(copy3, merged_rectangles)

    merged_rectangles = merge_close_contours(merged_rectangles,100)
    merged_rectangles = remove_contours_inside(merged_rectangles)


    ROI = []

    get_boxes(merged_rectangles, copy4, img, ROI)

    return ROI




def mean_distance_between_rectangles(centroids_array):

    num_centroids = len(centroids_array)

    # Compute pairwise distances between centroids
    distances = np.zeros((num_centroids, num_centroids))
    for i in range(num_centroids):
        for j in range(i + 1, num_centroids):
            distances[i, j] = np.linalg.norm(centroids_array[i] - centroids_array[j])
            distances[j, i] = distances[i, j]

    # Calculate mean distance
    mean_distance = np.mean(distances[distances != 0])  # Exclude distances between the same centroids

    return mean_distance


def process_text_from_tesseract(text_data):

    text_data = re.split(r'\n', text_data)
    text_data = ' '.join(text_data)

    print(text_data)


def run(info_dir,tes_mode:str,to_save,text_out_path:str):

    txts_path = []
    images_path = []

    for folder in os.listdir(info_dir):
        path = os.path.join(info_dir,folder)
        for content in os.listdir(path):
            if folder == 'coords':
                txts_path.append(os.path.join(path,content))
            else:
                images_path.append(os.path.join(path,content))

    img_txt = list(zip(txts_path,images_path))


    for txt_path,image_path in img_txt:
        points = read_coords(txt_path)


        ROI = merge_existing_boxes(image_path,points)

        file = text_out_path

        for roi in ROI:

            if tes_mode == '6':
                text = pytesseract.image_to_string(roi, config='--psm 6')
            elif tes_mode == '4':
                text = pytesseract.image_to_string(roi, config='--psm 4')
            else:
                text = pytesseract.image_to_string(roi, config='--psm 3')

            if to_save:
                with open(file, 'a') as f:
                    f.write(text + '\n')  # Add a newline after each text

            if text:
                process_text_from_tesseract(text)



if __name__ == '__main__':
    run('tets_boxes_from_craft')