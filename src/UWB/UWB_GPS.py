import subprocess
import time
import json
import numpy as np
from scipy.optimize import minimize
import serial

baudrate = 115200
port0 = '/dev/ttyACM0';
port1 = '/dev/ttyACM1';
port2 = '/dev/ttyACM2';
port3 = '/dev/ttyACM3';
port4 = '/dev/ttyACM4'

# ser0 = serial.Serial(port0, baudrate); command = 'initf\n'; ser0.write(command.encode('utf-8')) # Target board 
# ser1 = serial.Serial(port1, baudrate); command = 'respf\n'; ser1.write(command.encode('utf-8')) # Anchor 1
# ser2 = serial.Serial(port2, baudrate); command = 'respf\n'; ser2.write(command.encode('utf-8')) # Anchor 2 
# ser3 = serial.Serial(port3, baudrate); command = 'respf\n'; ser3.write(command.encode('utf-8')) # Anchor 3
# ser4 = serial.Serial(port4, baudrate); command = 'respf\n'; ser4.write(command.encode('utf-8')) # Anchor 4 
# time.sleep(2)

# Define anchors and distances
anchors = np.array([
    [10, 0, 0],  # Anchor 1
    [0, 10, 0],  # Anchor 2
    [0, 0, 10],  # Anchor 3
    [0, 0, 0]    # Anchor 4
    ])
# Initial guess
initial_guess = 10**3*np.ones(3)

def target_location(target):
    return np.linalg.norm([np.linalg.norm(anchor - target)-d for anchor, d in zip(anchors, dist)])

try:
    while True:
        try:
            # data = ser1.readline().decode('utf-8').strip(); res = json.loads(data); d1 = int(res['results'][0]['D_cm'])
            # data = ser2.readline().decode('utf-8').strip(); res = json.loads(data); d2 = int(res['results'][0]['D_cm'])
            # data = ser3.readline().decode('utf-8').strip(); res = json.loads(data); d3 = int(res['results'][0]['D_cm'])
            # data = ser4.readline().decode('utf-8').strip(); res = json.loads(data); d4 = int(res['results'][0]['D_cm'])

            # dist = [d1,d2,d3,d4]
            spoofPosition = [1e2 * np.random.rand() for _ in range(3)]
            dist = [np.linalg.norm(i-spoofPosition) for i in anchors]

            result = minimize(target_location, initial_guess)
            # Extract the solution
            location = result.x
            print("Estimated location:", location)
            print("Actual location:", spoofPosition)
            initial_guess = location
            time.sleep(1)
        except Exception as e:
            print(f"ERROR! in locating or reading{e}")
            continue
except KeyboardInterrupt:
    print("Stopping modes on board and exiting")
    # command = "python ExitAndClean.py"
    # subprocess.run(command, shell=True)
