import os
import sys
import cv2
import time
import torch
import socket
import struct
import signal
import numpy as np  
from _thread import *
from datetime import datetime

COCO_TO_NYU40_MAPPING = {'__background__': 'void',
                         'person': 'person', 'bicycle': 'void', 'car': 'void', 'motorcycle': 'void', 'airplane': 'void',
                         'bus': 'void', 'train': 'void', 'truck': 'void', 'boat': 'void', 'traffic light': 'void',
                         'fire hydrant': 'void', 'stop sign': 'void',  'parking meter': 'void', 'bench': 'table', 'bird': 'void',
                         'cat': 'void', 'dog': 'void', 'horse': 'void', 'sheep': 'void', 'cow': 'void',
                         'elephant': 'void', 'bear': 'void', 'zebra': 'void', 'giraffe': 'void', 'backpack': 'bag',
                         'umbrella': 'void', 'handbag': 'void', 'tie': 'void', 'suitcase': 'box', 'frisbee': 'void',
                         'skis': 'void', 'snowboard': 'void', 'sports ball': 'void', 'kite': 'void', 'baseball bat': 'void',
                         'baseball glove': 'void', 'skateboard': 'void', 'surfboard': 'void', 'tennis racket': 'void', 'bottle': 'void',
                         'wine glass': 'void', 'cup': 'void', 'fork': 'void', 'knife': 'void', 'spoon': 'void',
                         'bowl': 'void', 'banana': 'void', 'apple': 'void', 'sandwich': 'void', 'orange': 'void',
                         'broccoli': 'void', 'carrot': 'void', 'hot dog': 'void', 'pizza': 'void', 'donut': 'void',
                         'cake': 'void', 'chair': 'chair', 'couch': 'sofa', 'potted plant': 'void', 'bed': 'bed',
                         'dining table': 'table',  'toilet': 'void', 'tv': 'television', 'laptop': 'void', 'mouse': 'void',
                         'remote': 'void',  'keyboard': 'void', 'cell phone': 'void', 'microwave': 'box', 'oven': 'box',
                         'toaster': 'box', 'sink': 'sink', 'refrigerator': 'refridgerator', 'book': 'books', 'clock': 'void',
                         'vase': 'void', 'scissors': 'void', 'teddy bear': 'void', 'hair drier': 'void', 'toothbrush': 'void'}

client_socket_list = []
client_socket_addr = []
thread_control_list = []
HOST = ""
PORT = 12003

def stop_handler(signum,frame):
    print("Received stop signal")
    cleanup()
    exit()

def cleanup():
    print("Cleaning up resources...")
    print("Please wait for 5s...")
    
    for i in range(len(thread_control_list)): 
        thread_control_list[i] = False
    
    time.sleep(5) # for threads to return 
    server_socket.close()
    
    for client_socket in client_socket_list:
        client_socket.close()

def receiveAll(client_socket, length, control = [True], thread_id= 0, start_time = None): # 必要 不定時讀就不會block 要加timeout
    data = b""
    
    while (len(data) < length) and control[thread_id]:
        if (datetime.now() - start_time).total_seconds() > 10:
            break
        # print(len(data))
        new_data = client_socket.recv(length-len(data))
        data += new_data
        if new_data: 
            start_time = datetime.now()
    
    if len(data) != length: 
        # print(len(data),length,control[thread_id],datetime.now(),start_time)
        raise Exception
    
    return data

def threaded_client_cam(client_socket, client_addr, control=None, thread_id=None):
    payload_size = struct.calcsize("<I")
    pic_or_end_word_len = 3
    
    while True and control[thread_id]:
        try:
            # Check if received data is "pic" or "end"
            # print("Receiving image...")
            data = receiveAll(client_socket, pic_or_end_word_len, control, thread_id, start_time=datetime.now())
            msg_str = data.decode("utf-8")
            
            # Close thread if client disconnected
            if msg_str == "end":
                for i in range(len(client_socket_addr)):
                    try:
                        if client_socket_addr[i] == client_addr:
                            print(f"Client {client_addr[0]}:{client_addr[1]} disconnected")
                            break
                    except Exception as e:
                        print(e)
                        break
                break
            
            # Receive image byte length
            data = receiveAll(client_socket, payload_size, control, thread_id, start_time=datetime.now())
            msg_size = struct.unpack("<I", data)[0]
            
            # Receive actual image
            data = receiveAll(client_socket, msg_size, control, thread_id, start_time=datetime.now())
            frame = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            frame = cv2.flip(frame, 0)

            # Detect chair
            results = yolov5(frame)
            df = results.pandas().xyxy[0]
            df = df[df.confidence > 0.7]
            
            # Send data back to client
            if "chair" not in set(df["name"]):
                print(f"No chair detected in {client_addr[0]}:{client_addr[1]}")
                data_out = str.encode("0")
                client_socket.sendall(struct.pack("<i", len(data_out)) + data_out)
            else:
                df = df.rename(columns={'class': 'cls_id', 'name': 'class'})
                df['bbox'] = df.values[:, :4].tolist()
                df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1, inplace=True)
                df['class'] = df['class'].map(COCO_TO_NYU40_MAPPING)

                height, width, _ = frame.shape
                main_index = -1
                min_distance = 999999999999999

                for i in range(len(df)):
                    if (str(df["class"][i]) != "chair"):
                        continue
                    
                    distance = ((df["bbox"][i][0] + df["bbox"][i][2])/2 - width/2) * \
                        ((df["bbox"][i][0] + df["bbox"][i][2])/2 - width/2) + \
                        ((df["bbox"][i][1] + df["bbox"][i][3])/2 - height/2) * \
                        ((df["bbox"][i][1] + df["bbox"][i][3])/2 - height/2)
                    
                    if distance < min_distance:
                        main_index = i
                        min_distance = distance

                    temp = []
                    for j in range(4):
                        temp.append(int(df["bbox"][main_index][j]))
                    cv2.rectangle(frame, (temp[0], temp[1]), (temp[2], temp[3]), (255,0,0), 2)
                    cv2.imwrite(f"live_demo/inputs/img-box-{client_addr[1]}.jpg", frame)

                    bboxmsg = str(df["bbox"][main_index][0])+","+str(df["bbox"][main_index][1])+","+str(df["bbox"][main_index][2])+","+str(df["bbox"][main_index][3])
                    
                    print(f"Chair detected for port {client_addr[1]}: {bboxmsg}")
                    # data_out = str.encode(bboxmsg)
                    bboxmsg = bboxmsg.encode("utf-8")
                    client_socket.send(bboxmsg)

        except Exception as e:
            print(e)
            print(f"Client {client_addr[0]}:{client_addr[1]} lost connection")
            break

    client_socket.close()
    print(f"Client thread {thread_id} dead")

def server_main():
    thread_count = 0

    server_socket.bind((HOST,PORT))
    print("Socket bind complete")
    server_socket.listen(25) # n by n sockets
    print("Socket now listening")

    while True:
        try:
            client_socket, addr = server_socket.accept()
            client_socket_list.append(client_socket)
            client_socket_addr.append(addr)
            print(f"Connected to: {addr[0]}:{addr[1]}")

            # Receive byte length
            payload_size = struct.calcsize("<I")
            data = receiveAll(client_socket,payload_size,start_time=datetime.now())

            # Receive data of above byte length
            msg_size = struct.unpack("<I", data)[0]
            data = receiveAll(client_socket,msg_size,start_time=datetime.now())
            data_str = data.decode("utf-8")

            # Start new thread
            thread_control_list.append(True)
            if data_str == "start": 
                start_new_thread(
                    threaded_client_cam, 
                    (client_socket, addr, thread_control_list, thread_count)
                )

            print(f"=============== Thread Number: {thread_count} =============== {data_str} ===============")
            thread_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by keyboard.")
            break
        
        except Exception as e:
            print(e)
            break
    
    cleanup()

yolov5 = torch.hub.load(
    'ultralytics/yolov5', 
    'custom', 
    'yolov5m_finetuned.pt', 
    force_reload=True
)

signal.signal(signal.SIGTSTP, stop_handler)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
print("Socket created")

if __name__ == '__main__':
    server_main()