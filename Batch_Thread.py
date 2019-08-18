
import cv2
import random
import threading

import numpy as np

from Define import *
from Utils import *
from DataAugmentation import *

class Batch_Thread(threading.Thread):
    end = False
    ready = False
    min_data_size = 1
    max_data_size = 50

    batch_size = 0

    batch_data_length = 0
    batch_data_list = []

    def __init__(self, txt_path, min_data_size = 1, max_data_size = 50, batch_size = BATCH_SIZE):
        self.batch_size = batch_size
        self.min_data_size = min_data_size
        self.max_data_size = max_data_size

        self.data_list = read_txt(txt_path, REPLACE_DIR)

        threading.Thread.__init__(self)

    def get_batch_data(self):
        batch_image_data = []
        batch_label_data = []
        
        if self.ready:
            batch_image_data, batch_label_data = self.batch_data_list[0]

            del self.batch_data_list[0]
            self.batch_data_length -= 1

            if self.batch_data_length < self.min_data_size:
                self.ready = False

        return batch_image_data, batch_label_data

    def run(self):
        while not self.end:
            while self.batch_data_length >= self.max_data_size:
                continue
            
            batch_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)
            batch_label_data = np.zeros((BATCH_SIZE, CLASSES), dtype = np.float32)
            batch_data = random.sample(self.data_list, self.batch_size)

            for index, data in enumerate(batch_data):
                image_path, label = data

                image = cv2.imread(image_path)
                image = DataAugmentation(image)
                image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

                batch_image_data[index] = image.astype(np.float32)
                batch_label_data[index] = one_hot(label, CLASSES)
            
            self.batch_data_list.append([batch_image_data, batch_label_data])
            self.batch_data_length += 1

            if self.batch_data_length >= self.min_data_size:
                self.ready = True
            else:
                self.ready = False

if __name__ == '__main__':
    train_thread = Batch_Thread('./dataset/train.txt', max_data_size = 100)
    train_thread.start()

    # while True:
    #    pass

    train_thread.join()