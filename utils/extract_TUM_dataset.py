import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path

bag_path = Path("dataset-outdoors4_512_16.bag")
output = "tum_dataset"

cam0_topic = "/cam0/image_raw"
cam1_topic = "/cam1/image_raw"
imu_topic = "/imu0"

# create folders
os.makedirs(f"{output}/cam0/data", exist_ok=True)
os.makedirs(f"{output}/cam1/data", exist_ok=True)
os.makedirs(f"{output}/imu0", exist_ok=True)

cam0_ts = open(f"{output}/cam0/timestamps.txt", "w")
cam1_ts = open(f"{output}/cam1/timestamps.txt", "w")

imu_file = open(f"{output}/imu0/data.csv", "w", newline="")
imu_writer = csv.writer(imu_file)
imu_writer.writerow([
    "timestamp",
    "gyro_x","gyro_y","gyro_z",
    "acc_x","acc_y","acc_z"
])

typestore = get_typestore(Stores.ROS1_NOETIC)

cam0_idx = 0
cam1_idx = 0

with AnyReader([bag_path], default_typestore=typestore) as reader:

    connections = [x for x in reader.connections if x.topic in [cam0_topic, cam1_topic, imu_topic]]

    for connection, timestamp, rawdata in tqdm(reader.messages(connections)):

        msg = reader.deserialize(rawdata, connection.msgtype)
        t = timestamp * 1e-9  # convert to seconds

        # cam0
        if connection.topic == cam0_topic:

            if msg.encoding == "mono16":
                img = np.frombuffer(msg.data, dtype=np.uint16)
                img = img.reshape(msg.height, msg.width)

            elif msg.encoding == "mono8":
                img = np.frombuffer(msg.data, dtype=np.uint8)
                img = img.reshape(msg.height, msg.width)

            else:
                # fallback
                img = np.frombuffer(msg.data, dtype=np.uint8)
                img = img.reshape(msg.height, msg.width, -1)

            filename = f"{cam0_idx:06d}.png"
            cv2.imwrite(f"{output}/cam0/data/{filename}", img)

            cam0_ts.write(f"{t}\n")
            cam0_idx += 1

        # cam1
        elif connection.topic == cam1_topic:

            if msg.encoding == "mono16":
                img = np.frombuffer(msg.data, dtype=np.uint16)
                img = img.reshape(msg.height, msg.width)

            elif msg.encoding == "mono8":
                img = np.frombuffer(msg.data, dtype=np.uint8)
                img = img.reshape(msg.height, msg.width)

            else:
                # fallback
                img = np.frombuffer(msg.data, dtype=np.uint8)
                img = img.reshape(msg.height, msg.width, -1)

            filename = f"{cam1_idx:06d}.png"
            cv2.imwrite(f"{output}/cam1/data/{filename}", img)

            cam1_ts.write(f"{t}\n")
            cam1_idx += 1

        # imu
        elif connection.topic == imu_topic:

            imu_writer.writerow([
                t,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])

cam0_ts.close()
cam1_ts.close()
imu_file.close()

print("Extraction complete.")