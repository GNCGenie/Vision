import serial
import json
import time

baudrate = 115200
port1 = "/dev/ttyACM3"
port2 = "/dev/ttyACM4"
port3 = "/dev/ttyACM5"
# port4 = "/dev/ttyACM3"
# port5 = "/dev/ttyACM4"

ser1 = serial.Serial(port1, baudrate)
command = "initf 4 2400 200 25 2 42 01:02:03:04:05:06:07:08 1 0 0 1 2 3 4 \n" 
ser1.write(command.encode('utf-8'))

ser2 = serial.Serial(port2, baudrate)
command = "respf 4 2400 200 25 2 42 01:02:03:04:05:06:07:08 1 0 0 1\n"
ser2.write(command.encode('utf-8'))

ser3 = serial.Serial(port3, baudrate)
command = "respf 4 2400 200 25 2 42 01:02:03:04:05:06:07:08 1 0 0 2\n"
ser3.write(command.encode('utf-8'))

# ser4 = serial.Serial(port4, baudrate)
# command = "respf 4 2400 200 25 2 42 01:02:03:04:05:06:07:08 1 0 0 3\n"
# ser4.write(command.encode('utf-8'))

# ser5 = serial.Serial(port5, baudrate)
# command = "respf 4 2400 200 25 2 42 01:02:03:04:05:06:07:08 1 0 0 4\n"
# ser5.write(command.encode('utf-8'))
time.sleep(1)

numBoards = 2
try:
    while True:
        try:
            data = ser1.readline().decode('utf-8').strip()
            res = json.loads(data)
            for i in range(numBoards):
                print(f"Anchor : {res['results'][i]['Addr']} : Distance {res['results'][i]['D_cm']}")
        except Exception as e:
            print(f"ERROR! in reading {e}")
            continue
except KeyboardInterrupt:
    print("Stopping modes on board and exiting")
    command = 'stop\n'
    ser1.write(command.encode('utf-8'))
    ser2.write(command.encode('utf-8'))
    ser3.write(command.encode('utf-8'))
