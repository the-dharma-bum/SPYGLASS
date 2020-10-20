import os
import numpy as np
import cv2
from tqdm import tqdm


def get_index_from_string(string: str) -> int:
	""" ugly trick to extract a number from a path (string) 
		1. replace '/' and '.' by ' '.
		2. for each word test if it's a digit and stores the result in a list.
		3. as we now there's one and one digit only, return the first element.
	"""
	string = string.replace("/", " ")
	string = string.replace(".", " ")
	return [int(word) for word in string.split() if word.isdigit()][0]


def cut_video(input_path: str, target_depth: int, output_dir: str)-> None:
    capture = cv2.VideoCapture(input_path)
    depth   = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nb_videos = int(depth / target_depth) + 1
    for i in range(nb_videos-1):
        output_path = output_dir +f'{get_index_from_string(input_path)}_{i}.mp4'
        output_params = output_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (height, width)
        output  = cv2.VideoWriter(*output_params)
        for t in range(i*target_depth, (i+1)*target_depth):
            _, frame = capture.read()
            output.write(frame)
    last_output_path = input_path[:-5] + f'_{nb_videos}_last.mp4'
    output_params = last_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (height, width)
    last_output = cv2.VideoWriter(*output_params)
    for t in range((nb_videos-1)*target_depth, depth):
        _, frame = capture.read()
        last_output.write(frame)
    for t in range(depth, nb_videos*target_depth):
        last_output.write(np.zeros((height,width), dtype='uint8'))
    capture.release()


def cut_dataset(input_dir: str, output_dir: str, target_depth: int) -> None:
    input_list = sorted(os.listdir(input_dir))
    for i in tqdm(range(len(input_list))):
        input_path = os.path.join(input_dir, input_list[i])
        cut_video(input_path, target_depth, output_dir)

