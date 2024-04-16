import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.total_time = 0

    def time(self):
        return self

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.total_time += elapsed_time

    def get_and_reset(self):
        total = self.total_time
        self.total_time = 0
        return total


if __name__ == '__main__':
    from time import sleep
    timer_1 = Timer()

    with timer_1.time():
        sleep(1)

    # Block 2: This block will not be timed
    sleep(1)

    with timer_1.time():
        # Block 3: This block will also be timed and its time will add to the timer
        sleep(1)

    # Block 4: This block will not be timed
    sleep(1)

    total_time_timer_1 = timer_1.get_and_reset()
    print(f"Total timed duration: {total_time_timer_1} seconds")  # 2 seconds
