import GPUtil
from threading import Thread
import time

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.result=GPUtil.getGPUs()
        self.gpu_num=len(GPUtil.getGPUs())
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            self.result = GPUtil.getGPUs()
            #print(self.result[0].memoryUtil)
            time.sleep(self.delay)
    def get_result(self):
        try:
            return self.result
        except:
            return None
    def get_gpu_num(self):
        try:
            return self.gpu_num
        except:
            return 0
            

    def stop(self):
        self.stopped = True
        
# Instantiate monitor with a 10-second delay between updates
monitor = Monitor(10)

# Train, etc.

# Close monitor
