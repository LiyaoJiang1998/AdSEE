import numpy as np
import pandas as pd
import torch
import copy
import math

import cv2
import matplotlib.pyplot as plt
import PIL.Image
import scipy
import scipy.ndimage
import dlib

from face_swap import face_swap_algorithm

def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return PIL.Image.fromarray(var.astype('uint8'))

def latents_to_images(latents, generator):
    latents = torch.from_numpy(latents).float().cuda()
    
    with torch.no_grad():
        images, _ = generator([latents], randomize_noise=False, input_is_latent=True)
    
    final_images = []
    for img in images:
        final_images.append(tensor2im(img))
    return final_images

def solution_to_faces_images(solution, data, edit_directions, style_vector_dim, generator):
    solution_data = pd.DataFrame(copy.deepcopy(data.to_dict()))
    reshaped_solution = np.reshape(solution, ((int(solution_data["face_count"][0]), len(edit_directions))))
    # Apply the edits in solution to the face_latent_raw
    solution_face_latent = np.reshape(solution_data['face_latents_raw'][0], \
                    (int(solution_data["face_count"][0]), style_vector_dim[0], style_vector_dim[1])) 
    for xid in range(solution_face_latent.shape[0]):     # for each face
        for yid in range(reshaped_solution.shape[1]): # for each edit direction
            solution_face_latent[xid, :, :] = solution_face_latent[xid, :, :] \
                                + reshaped_solution[xid, yid] * edit_directions[yid,:,:]
    
    solution_face_latent = np.reshape(solution_face_latent, \
                    (int(solution_data["face_count"][0]), style_vector_dim[0], style_vector_dim[1])) 
    
    final_images = latents_to_images(solution_face_latent, generator)
    return final_images
    
    
def get_landmark(img, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

#     img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def reverse_quad_transform(image, background_image, quad_to_map_to):
    # https://stackoverflow.com/questions/37525264/how-to-map-rectangle-image-to-quadrilateral-with-pil
    # forward mapping, for simplicity
    result = PIL.Image.new("RGB",image.size)
    result_pixels = result.load()
    background_image_pixels = background_image.load()

    width, height = result.size

    for y in range(height):
        for x in range(width):
#             result_pixels[x,y] = (0,0,0)
           result_pixels[x,y] = background_image_pixels[x,y]

    p1 = (quad_to_map_to[0],quad_to_map_to[1])
    p2 = (quad_to_map_to[2],quad_to_map_to[3])
    p3 = (quad_to_map_to[4],quad_to_map_to[5])
    p4 = (quad_to_map_to[6],quad_to_map_to[7])

    p1_p2_vec = (p2[0] - p1[0],p2[1] - p1[1])
    p4_p3_vec = (p3[0] - p4[0],p3[1] - p4[1])

    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x,y))

            y_percentage = y / float(height)
            x_percentage = x / float(width)

            # interpolate vertically
            pa = (p1[0] + p1_p2_vec[0] * y_percentage, p1[1] + p1_p2_vec[1] * y_percentage) 
            pb = (p4[0] + p4_p3_vec[0] * y_percentage, p4[1] + p4_p3_vec[1] * y_percentage)

            pa_to_pb_vec = (pb[0] - pa[0],pb[1] - pa[1])

            # interpolate horizontally
            p = (pa[0] + pa_to_pb_vec[0] * x_percentage, pa[1] + pa_to_pb_vec[1] * x_percentage)

            try:
                result_pixels[p[0],p[1]] = (pixel[0],pixel[1],pixel[2])
            except Exception:
                pass

    return result


def inverse_align_face(face_img, person_img, predictor):
    # make the inverse of alignment, stitch edited face back to orignal image
    """
    :param filepath: str
    :return: PIL Image
    """
    try:
        lm = get_landmark(person_img, predictor)
    except Exception as e:
        print(e)
        # if cannot find face, return the orignal person image
#         img = PIL.Image.fromarray(person_img.astype('uint8'), 'RGB') 
        return None

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    
    # read image
    img = PIL.Image.fromarray(person_img.astype('uint8'), 'RGB')
    face_img = PIL.Image.fromarray(face_img.astype('uint8'), 'RGB')
    
    output_size = 256
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    
    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        # disable bluring for inverse transform
#         h, w, _ = img.shape
#         y, x, _ = np.ogrid[:h, :w, :1]
#         mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
#                           1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
#         blur = qsize * 0.02
#         img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
#         img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    
    # Resize face image to match person_image.
    face_img = face_img.resize(img.size, PIL.Image.ANTIALIAS)
    
    # find the inverse transformation from the face image rectangle to a QUAD
    result = reverse_quad_transform(face_img, img, (quad + 0.5).flatten())
    
    # undo padding from results
    if enable_padding and max(pad) > border - 4:
        result = result.crop((pad[0], pad[1], result.size[0]-pad[2], result.size[1]-pad[3]))
    
    # undo cropping, paste the cropped face portion of results back into original
    background_img = PIL.Image.fromarray(person_img.astype('uint8'), 'RGB')
    original_size = background_img.size
    if shrink > 1:
        background_img = background_img.resize(rsize, PIL.Image.ANTIALIAS)
    background_img.paste(result, (crop[0], crop[1]))
    
    # undo shrinking:
    if shrink > 1:
        background_img = background_img.resize(original_size, PIL.Image.ANTIALIAS)
    
    # Return inverse transformed person image.
    return background_img

def merge_persons_with_mask(original_img, result_person_img_list, 
                            original_person_img_list, mask_list, face_align_predictor, erode=3):
    # merge the masked out persons with editted face, back into the original image.
    original_img = PIL.Image.fromarray(original_img.astype('uint8'), 'RGB')
    any_swap_made = False
    for i, (result_person_img, original_person_img) in enumerate(zip( \
                                            result_person_img_list, original_person_img_list)):
        if result_person_img is None:
            continue # inverse face alignment failed for result person image, so skip
        mask = PIL.Image.fromarray(mask_list[i]).convert("RGB")
        # applying cv2.Erode to erode and smooth the mask edges
        mask_cv2 = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
        kernel = np.ones((erode, erode), np.uint8)
        mask_cv2 = cv2.erode(mask_cv2, kernel)
        mask = PIL.Image.fromarray(cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2RGB))
        
        # face swap from face detected in result_person_img, and in original_person_img
        swapped_person_img = face_swap(src_img=result_person_img, dst_img=original_person_img,
                                      predictor=face_align_predictor)
        if swapped_person_img is not None:
            # composite each masked person image with original image
            original_img = PIL.Image.composite(swapped_person_img, original_img, mask.convert("L"))
            any_swap_made = True
        
    return original_img, any_swap_made

def face_swap(src_img, dst_img, predictor):
    '''
    Swapped the face in src_img, to the dst_img.
    Output:
        Return the dst_img with the face from src_img.
        Or Return None when cannot swap
    '''
    try:
        src_points = get_landmark(np.array(src_img), predictor) # np.array shape=(68, 2)
        dst_points = get_landmark(np.array(dst_img), predictor) # np.array shape=(68, 2)
        
        # Converts PIL image to cv2 image
        src_img_cv2 = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
        dst_img_cv2 = cv2.cvtColor(np.array(dst_img), cv2.COLOR_RGB2BGR)
        swapped_img = face_swap_algorithm(src_img_cv2, dst_img_cv2, src_points, dst_points)
        if swapped_img is None:
            return None
        swapped_img = PIL.Image.fromarray(cv2.cvtColor(swapped_img, cv2.COLOR_BGR2RGB))
    
    except Exception as e:
        # if cannot find face, give up swapping
        print(e)
        return None
    
    return swapped_img

def images_side_by_side(images, resize_dims, concat_axis):
    '''
    concat_axis: 0 for vertical, 1 for horizontal
    '''
    images_resized = [i.copy() for i in images]
    for i in range(len(images_resized)):
        images_resized[i].resize(resize_dims)
        
    res = np.concatenate([np.array(image) for image in images_resized], axis=concat_axis)
    return PIL.Image.fromarray(res)